"""Group-correlated uncertainty of closed, strictly common faint spectroscopic samples."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .astrometry import years
from .continuous_reference import verify_directory
from .full_coordinate_sources import reference_context, saved_reference_response
from .io import digest, save_json
from .stream_json import array_items


def read(path):
    return json.loads(path.read_text())


def one(task):
    root, cache, entry, chosen, pairing = task
    core = entry["core"]
    folder = root / "sources" / core
    science = folder / "private_standard_motion.json.gz"
    proof = folder / "v22_source_core_summary.json"
    target = cache / (core + ".json")
    ticket = dict(pairing=pairing, source=digest(proof), implementation=digest(Path(__file__)))
    if target.exists():
        if read(target)["ticket"] != ticket:
            raise ValueError("Changed physical group diagnostic")
        return str(target)
    if digest(science) != read(proof)["private_output_sha256"][science.name]:
        raise ValueError("Changed source member history")
    records = []
    for j, row in enumerate(array_items(science)):
        if j in chosen:
            q, low = chosen.pop(j)
            records.append((q, low, row))
            if not chosen:
                break
    if chosen:
        raise ValueError("Missing accepted spectral member slot")
    wanted = {k for _, _, row in records for k in row["detection_ids"]}
    source = Path(entry["source"])
    verify_directory(source, ("detections.csv",))
    columns = [
        "detection_id",
        "ra_deg",
        "dec_deg",
        "sigma_east_mas",
        "sigma_north_mas",
        "observation_time",
    ]
    data = pd.concat(
        [
            chunk.loc[chunk.detection_id.isin(wanted)]
            for chunk in pd.read_csv(source / "detections.csv", usecols=columns, chunksize=150000)
        ]
    )
    data = data.set_index("detection_id", drop=False, verify_integrity=True)
    if len(data) != len(wanted):
        raise ValueError("Missing immutable member measurement")
    contributions = {}
    members = []
    for q, low, row in records:
        ids = row["detection_ids"]
        d = data.loc[ids]
        h, ix = saved_reference_response(
            row,
            d,
            years(d.observation_time),
            np.hypot(d[["sigma_east_mas", "sigma_north_mas"]].to_numpy(), 50.0),
        )
        if not np.array_equal(ix, np.arange(len(ids))):
            raise ValueError("Frozen influence and ordered source members disagree")
        graph = (
            root
            / "reference"
            / f"component-{int(row['reference_component']):04d}"
            / "private_c0_unfiltered.npz"
        )
        center, solution, _, _, sha = reference_context(str(graph))
        if sha != row["reference_model_sha256"] or not np.allclose(
            center, row["projection_center_deg"], rtol=0, atol=1e-12
        ):
            raise ValueError("Changed physical reference or tangent chart")
        labels = row["physical_groups"]
        state = -np.einsum("pni,dni->dp", h, np.stack([solution[g][1] for g in labels], axis=1))
        draw = (state @ np.asarray(row["state_jacobian"]).T)[:, [1, 3]]
        draw -= draw.mean(axis=0)
        np.testing.assert_allclose(
            np.cov(draw, rowvar=False),
            np.asarray(row["reference_icrs_covariance"])[np.ix_([1, 3], [1, 3])],
            rtol=2e-7,
            atol=2e-7,
        )
        independent = np.asarray(row["independent_icrs_covariance"])[np.ix_([1, 3], [1, 3])]
        key = f"{low}|{sha}"
        if key not in contributions:
            contributions[key] = dict(
                n=0, draw=np.zeros((48, 2)), independent=np.zeros((2, 2)), sum_pm=np.zeros(2)
            )
        r = contributions[key]
        r["draw"] += draw
        r["independent"] += independent
        r["sum_pm"] += row["pm_icrs_masyr"]
        r["n"] += 1
        runs = sorted({exp for exp in row["exposures"] if exp.startswith("SDSS:")})
        if len(runs) != row["information"]["sdss_physical_runs"] or not runs:
            raise ValueError("Physical SDSS run membership did not close")
        members.append((q, runs))
    save_json(target, dict(ticket=ticket, contributions=contributions, members=members))
    return str(target)


def group_sensitivity(sample, membership):
    n = len(sample)
    groups = sorted({g for q in sample.q for g in membership[q]})
    if n < 5 or len(groups) < 2:
        return dict(n=n, sdss_runs=len(groups))
    lookup = {v: i for i, v in enumerate(groups)}
    rr = []
    cc = []
    vv = []
    for i, q in enumerate(sample.q):
        labels = membership[q]
        rr.extend([i] * len(labels))
        cc.extend(lookup[g] for g in labels)
        vv.extend([1 / len(labels)] * len(labels))
    incidence = csr_matrix((vv, (rr, cc)), shape=(n, len(groups)))
    sorts = {}
    radii = {}
    for m in ("ours", "xiao", "nsc"):
        r = np.hypot(sample[m + "_pm_e"], sample[m + "_pm_n"]).to_numpy()
        sorts[m] = np.argsort(r)
        radii[m] = r[sorts[m]]
    rng = np.random.default_rng(240082)
    sensitivity = {m: [] for m in radii}
    # Bound temporary memory, while aligning each physical-run perturbation across catalogues.
    for _ in range(8):
        weight = incidence @ rng.exponential(size=(8, len(groups))).T
        for m in radii:
            cum = np.cumsum(weight[sorts[m]], axis=0)
            sensitivity[m].extend(radii[m][np.argmax(cum >= 0.68 * cum[-1], axis=0)])
    improve = {
        m: 100 * (1 - np.asarray(sensitivity["ours"]) / sensitivity[m])
        for m in ("xiao", "nsc")
        if min(sensitivity[m]) > 0
    }
    return dict(
        n=n,
        sdss_runs=len(groups),
        draws=64,
        P68_reweight_95={m: np.quantile(x, [0.025, 0.975]) for m, x in sensitivity.items()},
        improvement_percent_reweight_95={m: np.quantile(x, [0.025, 0.975]) for m, x in improve.items()},
    )


def run(root, validation, output, workers):
    proof = validation / "v24_spectral_pairing_summary.json"
    spectral = validation / "private_final_spectral_pairing.parquet"
    registered = root / "registration/private_full_coordinate_plan.json"
    owner = root / "catalogue/v22_catalogue_ownership_summary.json"
    pairing = digest(proof)
    s = read(proof)
    if s["private_pair_sha256"] != digest(spectral) or s["closure_sha256"] != digest(owner):
        raise ValueError("Spectral labels belong to another blind catalogue")
    columns = ["q", "core", "slot", "reference_r"] + [
        m + "_pm_" + axis for m in ("ours", "xiao", "nsc") for axis in "en"
    ]
    frame = pd.read_parquet(spectral, columns=columns).dropna(subset=columns)
    frame = frame.loc[frame.reference_r.ge(21) & frame.reference_r.lt(23)].copy()
    entries = {e["core"]: e for e in read(registered)["cores"]}
    cache = output / "physical-errors"
    cache.mkdir(exist_ok=True, parents=True)
    tasks = []
    for core, part in frame.groupby("core", sort=True):
        chosen = {int(v.slot): (int(v.q), int(v.reference_r)) for v in part.itertuples()}
        if len(chosen) != len(part):
            raise ValueError("Duplicate final blind source in strict QSO sample")
        tasks.append((root, cache, entries[core], chosen, pairing))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(one, tasks, chunksize=1))
    sums = {}
    members = {}
    for path in paths:
        item = read(Path(path))
        for q, runs in item["members"]:
            if q in members:
                raise ValueError("Duplicate spectral source after global ownership")
            members[q] = runs
        for key, r in item["contributions"].items():
            s = sums.setdefault(
                key, dict(n=0, sum_pm=np.zeros(2), draw=np.zeros((48, 2)), independent=np.zeros((2, 2)))
            )
            for k in s:
                s[k] += r[k]
    results = {}
    for lo in (21, 22):
        sample = frame.loc[frame.reference_r.ge(lo) & frame.reference_r.lt(lo + 1)]
        pieces = [v for k, v in sums.items() if k.startswith(str(lo) + "|")]
        n = sum(r["n"] for r in pieces)
        if len(sample) != n:
            raise ValueError("Population error denominator changed")
        if not n:
            results[str(lo)] = dict(n=0)
            continue
        # Aligned labels within a physical solution, independent draws between disconnected graphs.
        shared = sum(np.cov(r["draw"], rowvar=False) for r in pieces) / n**2
        ind = sum(r["independent"] for r in pieces) / n**2
        results[str(lo)] = dict(
            n=n,
            physical_reference_components=len(pieces),
            mean_pm=sum(r["sum_pm"] for r in pieces) / n,
            mean_measurement_covariance=ind,
            mean_common_reference_covariance=shared,
            modeled_mean_se=np.sqrt(np.diag(ind + shared)),
            physical_run_sensitivity=group_sensitivity(sample, members),
        )
    save_json(
        output / "v24_common_qso_population_summary.json",
        dict(
            closure_sha256=digest(owner),
            pairing_sha256=pairing,
            magnitude=results,
            scope="Unweighted mean of strict 3-way QSO motions. Same-reference perturbations summed before "
            "covariance; disconnected components contribute independent matrices. Field residual/colour systematics are additional. "
            "Positive physical SDSS run reweighting is descriptive sensitivity; equal run share for a multi-run star.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "validation", "output"):
        p.add_argument(k, type=Path)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= a.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Use assigned CPUs")
    run(a.root, a.validation, a.output, a.workers)
