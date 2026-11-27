"""Pack signed reference shards during acquisition and reuse verified v2.2 buckets."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path

import duckdb

from .footprint_intake_v24 import select_pilots
from .io import digest, save_json


def read(path):
    return json.loads(path.read_text())


def groups(cells, size=32):
    pilots = set(select_pilots(cells))

    def order(label):
        box = cells[label]
        ra = sum(box["ra"]) / 2
        dec = sum(box["dec"]) / 2
        return (not (abs(dec) < 1.6 and (ra < 53 or ra >= 307)), ra, dec, label)

    ordered = sorted(cells.keys() - pilots, key=order)
    return [sorted(pilots)] + [ordered[i : i + size] for i in range(0, len(ordered), size)]


def location(work, label):
    revised = work / "resolved" / (label + ".json")
    return Path(read(revised)["shard"]) if revised.is_file() else work / "input" / label


def pack(work, label, shards, db, *, verify=False):
    input_rows = 0
    signatures = {}
    for name, shard in shards:
        receipt = shard / "v22_reference_input_core_summary.json"
        if not receipt.exists():
            return None
        rec = read(receipt)
        signatures[name] = rec["parquet_sha256"]
        input_rows += int(rec["measurements"])
    signature = sha256(json.dumps([duckdb.__version__, signatures], sort_keys=True).encode()).hexdigest()
    dest = work / "reference-packs" / label / signature
    dest.mkdir(exist_ok=True, parents=True)
    marker = dest / "private_pack.json"
    with (dest / "lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if marker.exists():
            r = read(marker)
            if r["shards"] != signatures or r["rows"] != input_rows:
                raise ValueError("Changed signed reference pack")
            if verify and any(digest(dest / k) != v for k, v in r["files"].items()):
                raise ValueError("Reference pack data changed")
            return marker
        paths = []
        for name, shard in shards:
            source = shard / "reference.parquet"
            if digest(source) != signatures[name]:
                raise ValueError("Changed reference input")
            paths.append(str(source))
        # A new attempt directory preserves any interrupted COPY.
        version = dest / "parts"
        for i in range(1, 1000000):
            if not version.exists():
                break
            version = dest / f"parts-{i}"
        version.mkdir()
        db.from_parquet(paths).create_view("pack_input", replace=True)
        if input_rows:
            db.execute(
                f"""COPY (SELECT *,hash(detection_id)%64 detection_bucket FROM pack_input)
                       TO '{version}' (FORMAT PARQUET,PARTITION_BY(detection_bucket),COMPRESSION ZSTD,ROW_GROUP_SIZE 32768)"""
            )
            n = db.execute(f"SELECT count(*) FROM read_parquet('{version}/*/*.parquet')").fetchone()[0]
            if n != input_rows:
                raise ValueError("Reference pack count changed")
        files = {str(p.relative_to(dest)): digest(p) for p in version.glob("*/*.parquet")}
        save_json(
            marker, dict(rows=input_rows, shards=signatures, engine=duckdb.__version__, files=files)
        )
    return marker


def available(base, workers):
    work = base / "v2.4"
    cells = read(work / "registration/private_full_footprint_plan.json")["new"]
    db = duckdb.connect()
    db.execute(f"SET threads={workers}")
    db.execute("SET memory_limit='3GB'")
    completed = rows = 0
    for i, labels in enumerate(groups(cells)):
        path = pack(work, f"new-{i:04d}", [(n, location(work, n)) for n in labels], db)
        if path:
            completed += len(labels)
            rows += read(path)["rows"]
    db.close()
    save_json(
        work / "registration/v24_reference_pack_progress_summary.json",
        dict(
            cores=completed,
            reference_rows=rows,
            engine=duckdb.__version__,
            meaning="Hash-partitioned signed measurements before global identity and reference conflict resolution.",
        ),
    )


def assemble(root, dest, binding_sha, db, plan):
    work = root.parent
    marker = dest / "private_detection_buckets.json"
    if marker.exists():
        if read(marker)["binding_sha256"] != binding_sha:
            raise ValueError("Changed physical graph inputs")
        return
    prior = work.parent / "v2.2"
    old_graph = prior / "graph"
    old_plan = prior / "registration/private_full_coordinate_plan.json"
    old_ticket = read(old_graph / "private_input_binding.json")
    old_marker = read(old_graph / "private_detection_buckets.json")
    if old_marker["binding_sha256"] != digest(old_graph / "private_input_binding.json") or old_ticket[
        "plan_sha256"
    ] != digest(old_plan):
        raise ValueError("Predecessor hash buckets are not bound to the archived registration")
    old_labels = [v["core"] for v in read(old_plan)["cores"]]
    if len(old_labels) != len(old_ticket["shards"]):
        raise ValueError("Predecessor reference registration count changed")
    changed = {
        n
        for n, sha in zip(old_labels, old_ticket["shards"])
        if read(root / "input" / n / "v22_reference_input_core_summary.json")["parquet_sha256"] != sha
    }
    if old_ticket["engine_version"] != duckdb.__version__:
        changed = set(old_labels)  # Hash algorithms from other engine versions are not interchangeable.
    new_cells = read(work / "registration/private_full_footprint_plan.json")["new"]
    if set(old_labels) | set(new_cells) != {e["core"] for e in plan["cores"]}:
        raise ValueError("Reference pack footprint differs from final registered plan")
    files = {}

    def install(p, bucket, label, checksum):
        if digest(p) != checksum:
            raise ValueError("Damaged reference bucket")
        target = dest / "detection_buckets" / bucket / (label + "-" + p.name)
        target.parent.mkdir(exist_ok=True, parents=True)
        if target.exists() and target.resolve() != p.resolve():
            raise ValueError("Inconsistent prior bucket link")
        if not target.exists():
            target.symlink_to(p.resolve())
        files[str(target.relative_to(dest))] = checksum

    if len(changed) < len(old_labels):
        from pandas import DataFrame

        db.register("replaced_core_ids", DataFrame({"source_core": list(changed) or [""]}))
        for name, sha in old_marker["files"].items():
            p = old_graph / name
            bucket = p.parent.name
            if digest(p) != sha:
                raise ValueError("Predecessor archive bucket changed")
            if changed:
                corrected = dest / "old-filtered" / bucket / p.name
                corrected.parent.mkdir(exist_ok=True, parents=True)
                db.execute(
                    f"""COPY (SELECT x.* EXCLUDE(detection_bucket) FROM read_parquet('{p}',hive_partitioning=true) x
                      ANTI JOIN replaced_core_ids USING(source_core)) TO '{corrected}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
                )
                p = corrected
                sha = digest(p)
            install(p, bucket, "previous", sha)
    packed = [(f"new-{i:04d}", g) for i, g in enumerate(groups(new_cells))]
    changed = sorted(changed)
    packed.extend((f"changed-{i:05d}", changed[i : i + 32]) for i in range(0, len(changed), 32))
    for label, names in packed:
        proof = pack(work, label, [(n, root / "input" / n) for n in names], db)
        if proof is None:
            raise ValueError("Unclosed reference input")
        r = read(proof)
        for name, sha in r["files"].items():
            p = proof.parent / name
            install(p, p.parent.name, label, sha)
    save_json(
        marker,
        dict(
            binding_sha256=binding_sha,
            files=files,
            bucket_count=64,
            reused_predecessor_cores=len(old_labels) - len(changed),
            replaced_predecessor_cores=len(changed),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= a.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Use assigned CPUs")
    available(a.base, a.workers)
