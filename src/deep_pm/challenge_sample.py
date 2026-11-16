"""Freeze independent sky regions around Gaia-selected high-motion targets."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from . import access
from .__main__ import tomllib
from .astrometry import epoch_halo_deg
from .io import save_json
from .provenance import run_stage


def partition_candidates(cache, config):
    spec = config["challenge"]
    lo, hi = spec["pm_edges"][0], spec["pm_edges"][-1]
    quality = (f"pm >= {lo} AND pm < {hi} AND astrometric_params_solved IN (31,95)"
               f" AND ruwe < {config['gaia_max_ruwe']} AND phot_g_mean_mag < {spec['max_g']}"
               f" AND pmra_error < {config['gaia_max_pm_error']}"
               f" AND pmdec_error < {config['gaia_max_pm_error']}"
               " AND parallax IS NOT NULL AND (abs(parallax)<10 OR parallax_error BETWEEN 0 AND 2)")
    for field in spec["search_fields"]:
        where = access.q3c(field["ra"], field["dec"], field["radius"]) + " AND " + quality
        frame = cache.checked("nsc", "source_id,ra,dec,pm,parallax", "gaia_dr3.gaia_source", where)
        yield frame.assign(split=field["split"])


def choose(frame, config):
    spec = config["challenge"]
    frame = frame.drop_duplicates("source_id").copy()
    frame["key"] = frame.source_id.map(lambda source: sha256(
        f"{config['seed']}/{source}".encode()).hexdigest())
    chosen = []
    halo = epoch_halo_deg(config)
    cone = spec["radius"] + halo
    # Do not use a challenge if its retrieval disc overlaps a field whose
    # outcomes can be inspected in the regression or development work.
    centers = [(r["ra"], r["dec"], max(spec["minimum_separation_deg"],
                                      cone + r["radius"] + epoch_halo_deg(dict(
                                          config, **r.get("overrides", {})))))
               for r in config.get("regions", {}).values()
               if r["role"] in {"regression", "historical_challenge", "new_development",
                                "independent_representative", "geometry_stress"}]
    sep = max(spec["minimum_separation_deg"], 2 * cone)
    for split in ("dev", "test"):
        for low, high in zip(spec["pm_edges"], spec["pm_edges"][1:]):
            group = frame.loc[frame.split.eq(split) & frame.pm.ge(low) & frame.pm.lt(high)].sort_values("key")
            population = 0
            for row in group.itertuples():
                ra = round(row.ra, 2) % 360
                dec = round(row.dec, 2)
                if centers:
                    pt = np.asarray(centers)
                    sky = np.deg2rad(pt[:, :2])
                    z = np.sin(sky[:, 1]) * np.sin(np.deg2rad(dec)) + np.cos(sky[:, 1]) * np.cos(
                        np.deg2rad(dec)) * np.cos(sky[:, 0] - np.deg2rad(ra))
                    if np.any(z > np.cos(np.deg2rad(pt[:, 2]))):
                        continue
                population += 1
                centers.append((ra, dec, sep))
                chosen.append(dict(region=f"hm_{split}_{low}_{population:02d}", split=split,
                                   bin_low=low, bin_high=high, source_id=str(row.source_id),
                                   ra=ra, dec=dec, radius=spec["radius"]))
                if population >= spec["sources_per_bin_per_split"]:
                    break
    return pd.DataFrame(chosen, columns=["region", "split", "bin_low", "bin_high", "source_id",
                                         "ra", "dec", "radius"])


def run(work, config_path, config):
    cache = access.Cache(work / "validation" / "cache")
    sample = choose(pd.concat(list(partition_candidates(cache, config)), ignore_index=True), config)
    validation = work / "validation"
    validation.mkdir(exist_ok=True)
    generated = [config_path.read_text()]
    role_prefix = config["challenge"].get("role_prefix", "gaia_challenge")
    for row in sample.itertuples():
        generated.extend(["", f"[regions.{row.region}]", f'role = "{role_prefix}_{row.split}"',
                          f"ra = {row.ra}", f"dec = {row.dec}", f"radius = {row.radius}",
                          "", f"[regions.{row.region}.overrides]", 'truth_mode = "annual_parallax"'])
        path = work / row.region / "validation"
        path.mkdir(parents=True, exist_ok=True)
        (path / "challenge_source.txt").write_text(row.source_id + "\n")
    (work / "challenge_regions.toml").write_text("\n".join(generated) + "\n")
    sample.to_csv(validation / "challenge_targets.csv", index=False)
    counts = {f"{split}/{lo}-{hi}": len(g) for (split, lo, hi), g in
              sample.groupby(["split", "bin_low", "bin_high"])}
    save_json(work / "challenge_sample_summary.json", dict(counts=counts, bins=config["challenge"]["pm_edges"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-a-v02.toml"))
    args = parser.parse_args()
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    args.work.mkdir(parents=True, exist_ok=True)
    outputs = ["challenge_regions.toml", "validation/challenge_targets.csv", "challenge_sample_summary.json"]
    with run_stage(args.work, args.config, "challenge", "prepare-challenge", outputs=outputs) as execute:
        if execute:
            run(args.work, args.config, config)
