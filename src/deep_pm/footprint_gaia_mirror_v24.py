"""Read original Gaia DR3 CSV values for sources located using the complete position-only FITS index."""
from bisect import bisect_left

from astropy.io import fits
import healpy as hp
import numpy as np
import pandas as pd

from .exposure_inputs import GAIA_COLUMNS

COLUMNS = (GAIA_COLUMNS + ",visibility_periods_used").split(",")


def sky_index(root, spec, halo):
    west, east = spec["ra"]
    south, north = spec["dec"]
    lo_dec, hi_dec = round(south - halo, 6), round(north + halo, 6)
    width = east - west + 2 * halo
    lo = (west - halo) % 360
    lo_ra = round(lo, 9)
    high = lo + width
    hi_ra = round(high if high <= 360 else high - 360, 9)
    radius = np.hypot((east-west)/2+halo, (north-south)/2+halo)+.002
    center = hp.ang2vec(((west+east)/2) % 360, (south+north)/2, lonlat=True)
    pixels = hp.query_disc(32, center, np.deg2rad(radius), nest=True, inclusive=True)
    selections = []
    paths = []
    for pixel in pixels:
        path = root / "healpix" / f"healpix-{pixel:05d}.fits"
        paths.append(path)
        with fits.open(path, memmap=True) as hdus:
            hdu = hdus[1]
            if hdu.header["HPXNSIDE"] != 32 or not hdu.header["HPXNEST"]:
                raise ValueError("Unexpected Gaia positional index")
            ra = np.asarray(hdu.data["RA"], dtype=float)
            dec = np.asarray(hdu.data["DEC"], dtype=float)
            ids = np.asarray(hdu.data["SOURCE_ID"], dtype="int64")
        if not np.all(hp.ang2pix(32,ra,dec,nest=True,lonlat=True) == pixel):
            raise ValueError("Published Gaia positions disagree with the local sky partition")
        longitude = (ra >= lo_ra) & (ra < hi_ra) if high <= 360 else (ra >= lo_ra) | (ra < hi_ra)
        keep = longitude & (dec>=lo_dec) & (dec<hi_dec)
        selections.append(pd.DataFrame({"source_id":ids[keep],
                                        "ra":ra[keep], "dec":dec[keep]}))
    result = pd.concat(selections,ignore_index=True).set_index("source_id").sort_index()
    if not result.index.is_unique:
        raise ValueError("Repeated Gaia positional ID")
    return result, paths


def original_files(folder, ids):
    """The ESA bulk-file names encode the first and last nested level-eight source-ID pixel."""
    files = sorted(folder.glob("GaiaSource_*.csv.gz"))
    ranges = [tuple(map(int,p.name[len("GaiaSource_"):-len(".csv.gz")].split("-"))) for p in files]
    if not ranges or any(l > h for l,h in ranges) or any(r[0]<=p[1] for p,r in zip(ranges,ranges[1:])):
        raise ValueError("Missing or overlapping original Gaia file index")
    maxima = [v[1] for v in ranges]
    chosen = set()
    for pix in np.unique(ids.to_numpy(dtype="int64") >> 43):
        i = bisect_left(maxima,pix)
        if i == len(files) or ranges[i][0] > pix:
            raise FileNotFoundError("No original source-ID range for a selected star")
        chosen.add(files[i])
    return sorted(chosen)


def original_rows(path, ids):
    selected = []
    for chunk in pd.read_csv(path,comment="#",usecols=COLUMNS,dtype={"source_id":"int64"},chunksize=80000):
        rows = chunk.loc[chunk.source_id.isin(ids),COLUMNS]
        if len(rows):
            selected.append(rows)
    if selected:
        return pd.concat(selected,ignore_index=True).set_index("source_id")
    return pd.DataFrame(columns=COLUMNS).set_index("source_id")


def aligned(index, rows):
    rows = rows.sort_index()
    if not rows.index.is_unique or not rows.index.equals(index.index):
        raise ValueError("The original Gaia release and spatial index contain different source IDs")
    for column in ("ra","dec"):
        if np.any(~np.isfinite(rows[column])) or not np.allclose(index[column],rows[column],rtol=0,atol=1e-10):
            raise ValueError("Original positions changed between the spatial index and source release")
    return rows.reset_index()[COLUMNS]


def compare(old, new):
    old = old.set_index("source_id").sort_index()
    new = new.set_index("source_id").sort_index()
    common = old.index.intersection(new.index)
    equal_ids = old.index.is_unique and new.index.is_unique and old.index.equals(new.index)
    accepted = bool(equal_ids)
    detail = {}
    for k in COLUMNS[1:]:
        a = old.loc[common,k].to_numpy(dtype=float)
        b = new.loc[common,k].to_numpy(dtype=float)
        missing = int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        delta = np.abs(a[finite] - b[finite])
        if k in ("ra","dec"):
            good = np.all(delta <= .001/3600000)
        elif k in ("ref_epoch","visibility_periods_used","astrometric_params_solved"):
            good = np.all(delta == 0)
        elif k in ("phot_g_mean_mag","phot_rp_mean_mag"):
            good = np.all(delta <= 1e-5)
        else:
            good = np.all((delta <= 1e-4) & (delta <= 1e-7 + 1e-6*np.abs(a[finite])))
        detail[k] = dict(missing_mask_differences=missing, maximum_absolute_difference=float(delta.max()) if len(delta) else None,
                         agrees=bool(good and not missing))
        accepted &= detail[k]["agrees"]
    return dict(archive_rows=len(old),mirror_rows=len(new),common_ids=len(common),columns=detail,passes=bool(accepted))


def read_signed_bulk(task):
    from .io import digest
    path, ids = task
    return original_rows(path, ids), (path.name, digest(path))


def run(work, mirror, workers, pending):
    """Compare the original four fields before making any additional candidate payloads available."""
    from concurrent.futures import ProcessPoolExecutor
    import json
    import os
    from pathlib import Path
    from .footprint_intake_v24 import select_pilots, SETTINGS
    from .io import digest, save_json

    reg = work / "registration"
    manifest = reg / "private_full_footprint_plan.json"
    footprint = digest(manifest)
    if footprint != json.loads((reg/"v24_footprint_inventory_summary.json").read_text())["signed_manifest_sha256"]:
        raise ValueError("Changed registered footprint")
    cells = json.loads(manifest.read_text())["new"]
    controls = select_pilots(cells)
    if len(controls) != 4 or set(controls) & set(pending) or len(set(pending)) != len(pending):
        raise ValueError("Invalid mirror validation scope")
    indexes, sky_files, source_files = {}, {}, {}
    for name in controls + list(pending):
        indexes[name], sky_files[name] = sky_index(mirror, cells[name], SETTINGS["halo_deg"])
        source_files[name] = original_files(mirror/"csv", indexes[name].index)
    tasks = {}
    for name, paths in source_files.items():
        for path in paths:
            tasks.setdefault(path,[]).append(indexes[name].index.to_numpy())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(read_signed_bulk, [(p,np.unique(np.concatenate(ids))) for p,ids in tasks.items()]))
    all_rows = pd.concat([v for v,_ in results])
    original_shas = dict(s for _,s in results)
    if all_rows.index.has_duplicates:
        raise ValueError("An original source ID occurs in more than one source file")
    tables = {n: aligned(ix,all_rows.loc[all_rows.index.isin(ix.index)]) for n,ix in indexes.items()}
    comparisons = []
    for name in controls:
        resolved = work/"resolved"/(name+".json")
        src = Path(json.loads(resolved.read_text())["source"]) if resolved.exists() else work/"intake"/name
        old = pd.read_csv(src/"validation/gaia.csv",dtype={"source_id":"int64"})
        comparisons.append(compare(old,tables[name]))
    usable = all(v["passes"] for v in comparisons)
    out = dict(footprint=footprint,slurm_job=os.environ["SLURM_JOB_ID"],controls=comparisons,
               original_files=len(tasks),position_shards=len(set(p for x in sky_files.values() for p in x)),
               valid_controls=sum(v["passes"] for v in comparisons),candidate_core_count=len(pending),
               scope="Four original intake controls, native DR3 source IDs and original publication nulls; no proper-motion outcome selected.")
    save_json(reg/"v24_gaia_original_controls_summary.json",out)
    if usable:
        hashes = {p:digest(p) for p in {p for paths in sky_files.values() for p in paths}}
        for name, df in tables.items():
            dest = work/"gaia-original"/name
            dest.mkdir(exist_ok=True,parents=True)
            tmp = dest/("gaia-"+os.environ["SLURM_JOB_ID"]+".part")
            output = dest/"gaia.csv"
            df.to_csv(tmp,index=False)
            checksum = digest(tmp)
            if output.exists() and digest(output)!=checksum:
                raise ValueError("A verified Gaia subset changed during replay")
            tmp.replace(output)
            save_json(dest/"private_provenance.json",dict(core=name,footprint=footprint,halo=SETTINGS["halo_deg"],
                      original_files={p.name:original_shas[p.name] for p in source_files[name]},
                      position_shards={p.name:hashes[p] for p in sky_files[name]},gaia_sha256=checksum,
                      rows=len(df),controls_sha256=digest(reg/"v24_gaia_original_controls_summary.json")))


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("mirror",type=Path)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--pending",nargs="*",default=[])
    args=parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=args.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use the allocated Slurm CPUs")
    run(**vars(args))
