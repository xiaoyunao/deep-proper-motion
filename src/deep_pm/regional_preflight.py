"""Lock independent regional pilot footprints using only survey geometry."""
import argparse
from collections import Counter
from pathlib import Path
import os

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon
from scipy.spatial import cKDTree

from . import access
from .astrometry import project, unproject, MAS_PER_RADIAN
from .footprints import sdss_visits
from .io import save_json
from .nsc_comparison import configuration
from .provenance import run_stage
from .qso_fields import spectra


def sample_cap(center, radius, n, seed):
    rng = np.random.default_rng(seed)
    a, u = rng.uniform(0, 2*np.pi, n), rng.random(n)
    cap = 1 - np.cos(np.deg2rad(radius))
    r = np.arccos(1-cap*u)
    xy = np.column_stack([np.cos(a),np.sin(a)]) * np.tan(r[:,None]) * MAS_PER_RADIAN
    return xy, unproject(xy, center), 2*np.pi*cap*(180/np.pi)**2


def span_update(mask, t, first, last):
    first[mask] = np.minimum(first[mask], t)
    last[mask] = np.maximum(last[mask], t)


CHIP_VERTICES = [f"vertex_{axis}{i}" for axis in ("ra","dec") for i in range(1,5)]


def r_chips(cache, center, radius):
    columns = ["exposure","instrument","mjd"] + CHIP_VERTICES
    where = access.q3c(*center,radius+.20) + " AND filter='r'"
    return access.checked_spatial(cache,"nsc",",".join(columns),"nsc_dr2.chip",where,
                                  center[1]-radius-.200001,center[1]+radius+.200001)


def geometry(cache, center, radius, xy, sky, window):
    sdss = np.zeros(len(xy), dtype=int)
    first = np.full(len(xy), np.inf)
    last = np.full(len(xy), -np.inf)
    for scan in sdss_visits(window, center, radius, rerun=301, vectorized=True):
        mask = scan.coverage(xy,sky)
        sdss[mask] += 1
        span_update(mask,scan.year,first,last)
    delta_sdss = np.maximum(0, last-first)
    chips = r_chips(cache,center,radius)
    ra = chips[[f"vertex_ra{i}" for i in range(1,5)]].to_numpy()
    dec = chips[[f"vertex_dec{i}" for i in range(1,5)]].to_numpy()
    ok = np.isfinite(ra).all(axis=1) & np.isfinite(dec).all(axis=1) & chips.mjd.between(50000,62000).to_numpy()
    chip_ids = np.flatnonzero(ok)
    points = project(ra[ok].ravel(), dec[ok].ravel(), center).reshape(-1,4,2)
    tree = cKDTree(xy)
    first.fill(np.inf)
    last.fill(-np.inf)
    observers = {}
    for row,corners in zip(chip_ids,points):
        centroid = corners.mean(axis=0)
        extent = np.linalg.norm(corners-centroid,axis=1).max()
        neighbors = np.asarray(tree.query_ball_point(centroid,extent),dtype=int)
        if not len(neighbors):
            continue
        inside = neighbors[Polygon(corners).contains_points(xy[neighbors])]
        if len(inside):
            span_update(inside,chips.mjd.iloc[row]/365.25,first,last)
            observers[chips.exposure.iloc[row]] = str(chips.instrument.iloc[row]).strip()
    return sdss, delta_sdss, np.isfinite(first), np.maximum(0,last-first), Counter(observers.values())


def preflight(work, spec, settings, catalogue, window):
    center = spec["ra"], spec["dec"]
    radius = spec["radius"]
    options = settings["regional"]
    xy, sky, area = sample_cap(center,radius,options["preflight_points"],options["seed"])
    qso = spectra(catalogue,center,radius,0.)
    qso = qso.loc[qso.reference_r.between(21,23,inclusive="left")]
    pos = qso[["ra","dec"]].to_numpy()
    index = len(xy)
    cache = access.Cache(work/"cache")
    sdss, delta_sdss, nsc, span, instruments = geometry(cache,center,radius,
                np.vstack([xy,project(*pos.T, center)]), np.vstack([sky,pos]),window)
    joint = (sdss>0) & nsc
    timely = joint & (span>=options["coverage_years"])
    outer = radius+settings["calibration_buffer_deg"]+.003
    optical = "ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)"
    count = {}
    for key,r in (("core",radius),("buffered",outer)):
        where = access.q3c(*center,r) + " AND "+optical
        count[key] = int(cache.query("nsc","SELECT COUNT(*) AS n FROM nsc_dr2.object WHERE "+where,expected=1).iloc[0,0])
    first = slice(None,index)
    fraction = lambda a: float(np.mean(a[first]))
    needs = dict(joint_fraction=fraction(joint),nsc_span_fraction=fraction(timely),
                 faint_qso_positions=int(timely[index:].sum()),core_archive_candidates=count["core"],
                 buffered_archive_candidates=count["buffered"])
    return dict(area_sqdeg=area, **needs,sampled_positions=index, faint_spectra=len(qso),
                nscr_exposures_by_instrument=dict(instruments),
                single_sdss_fraction=fraction(sdss==1), repeated_sdss_fraction=fraction(sdss>=2),
                long_repeat_fraction=fraction(delta_sdss>=2),
                short_repeat_fraction=fraction((sdss>=2)&(delta_sdss<2)),
                joint_nsc_span_median=float(np.median(span[first][joint[first]])) if joint[first].any() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("catalogue",type=Path)
    p.add_argument("window",type=Path)
    p.add_argument("regions",nargs="+")
    p.add_argument("--config",type=Path,default=Path("configs/regional-v07.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    _,settings=configuration(args.config)
    stage = "regional-preflight-" + settings.get("run_series", "v07")
    failed = []
    for name in args.regions:
        work=args.root/name
        work.mkdir(parents=True,exist_ok=True)
        try:
            with run_stage(work,args.config,name,stage,
                           outputs=["regional_preflight_summary.json"]) as go:
                if go:
                    item=preflight(work,settings["fields"][name],settings,args.catalogue,args.window)
                    save_json(work/"regional_preflight_summary.json",item)
                    print(name,item,flush=True)
        except (RuntimeError,ValueError) as exc:
            failed.append(name)
            print(name,type(exc).__name__,str(exc)[:220],flush=True)
    if failed:
        raise SystemExit("Incomplete preflight: "+", ".join(failed))


if __name__=="__main__":
    main()
