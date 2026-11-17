"""Read the released proper-motion and spectroscopic tables on a Slurm node."""
import argparse
from collections import Counter
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .io import save_json
from .provenance import run_stage


def inside(ra, dec, box, halo=0):
    lo, hi = box["ra"]
    south, north = box["dec"]
    return ((ra - lo + halo) % 360 < hi-lo+2*halo) & (dec >= south-halo) & (dec < north+halo)


def coordinates(ra, dec):
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(d)*np.cos(a), np.cos(d)*np.sin(a), np.sin(d)))


def optical_qsos(sdss_file, desi_file):
    with fits.open(sdss_file, memmap=True) as handle:
        table = handle[1].data
        if len(table) != 750414:
            raise ValueError("Unexpected DR16Q parent row count")
        a = pd.DataFrame(dict(ra=table["RA"].astype(float), dec=table["DEC"].astype(float),
                              reference_r=(22.5-2.5*np.log10(np.where(table["PSFFLUX"][:,2]>0,
                                                                   table["PSFFLUX"][:,2], np.nan))),
                              z=table["Z"].astype(float)))
        a["origin"] = "DR16Q"
    with fits.open(desi_file, memmap=True) as handle:
        t = handle["QSO_CAT"].data
        ok = ((t["Z"]>.5)&(t["Z"]<6)&(t["ZWARN"]==0)&(t["FLUX_R"]>0)&(t["FLUX_IVAR_R"]>0))
        b = pd.DataFrame(dict(ra=t["TARGET_RA"][ok].astype(float), dec=t["TARGET_DEC"][ok].astype(float),
                              reference_r=22.5-2.5*np.log10(t["FLUX_R"][ok].astype(float)),
                              z=t["Z"][ok].astype(float), snr=t["TSNR2_QSO"][ok].astype(float),
                              target=t["TARGETID"][ok].astype(np.int64)))
        b = b.sort_values("snr", ascending=False).drop_duplicates("target").reset_index(drop=True)
        b["origin"] = "DESI"
    a = a.loc[np.isfinite(a[["ra","dec"]]).all(axis=1)].reset_index(drop=True)
    b = b.loc[np.isfinite(b[["ra","dec"]]).all(axis=1)].reset_index(drop=True)
    return a,b


def merge_qsos(a,b):
    # A reciprocal unique 1 arcsec pair retains the DESI position and r flux.
    # The union of ambiguous candidates is separately vetoed for training.
    xyz_a = coordinates(a.ra,a.dec)
    xyz_b = coordinates(b.ra,b.dec)
    radius = np.deg2rad(1/3600)
    tree_a,tree_b = cKDTree(xyz_a),cKDTree(xyz_b)
    near_a = tree_a.query_ball_point(xyz_b,radius,return_length=True)
    near_b = tree_b.query_ball_point(xyz_a,radius,return_length=True)
    distance,index = tree_a.query(xyz_b)
    unique = (near_a==1)&(near_b[index]==1)&(distance<=radius)
    overlap = index[unique]
    combined = pd.concat((a.loc[near_b==0],b.loc[near_a<=0],b.loc[unique].assign(origin="both")),
                         ignore_index=True,sort=False)
    return combined,dict(dr16q=len(a),desi_quality=len(b),overlap=len(overlap),
                         ambiguous_dr16q=int((near_b>0).sum()-len(overlap)),
                         ambiguous_desi=int((near_a>0).sum()-len(overlap)),union=len(combined))


def catalogue_audit(work, legacy, sdss, desi, boxes):
    qa,qb = optical_qsos(sdss,desi)
    combined,counts = merge_qsos(qa,qb)
    pd.concat([qa,qb],ignore_index=True).to_csv(work/"spectra_veto.csv",index=False)
    combined.to_csv(work/"spectra_union.csv",index=False)
    all_flags = Counter()
    chunks = {name:[] for name in boxes}
    columns = ("objID","RA","DEC","PM_SOURCE","pmRA","pmDEC","pmRA_err","pmDEC_err")
    with fits.open(legacy,memmap=True) as handle:
        hdu=handle[1]
        table=hdu.data
        identifiers={k.casefold():k for k in table.names}
        print("Xiao FITS",len(table),list(table.names),flush=True)
        retain=[identifiers[k.casefold()] for k in columns if k.casefold() in identifiers]
        if not {"ra","dec","pm_source"} <= identifiers.keys():
            raise ValueError("Xiao FITS is missing coordinate or source columns")
        n=len(table)
        for lo in range(0,n,1_000_000):
            sl=slice(lo,lo+1_000_000)
            ra,dec=table[identifiers["ra"]][sl],table[identifiers["dec"]][sl]
            raw=table[identifiers["pm_source"]][sl]
            values,frequency=np.unique(raw,return_counts=True)
            all_flags.update({str(v).strip():int(k) for v,k in zip(values,frequency)})
            for name,box in boxes.items():
                rows=np.flatnonzero(inside(ra,dec,box,.06))+lo
                if len(rows):
                    chunks[name].append(pd.DataFrame({k:table[k][rows].astype(
                        table[k].dtype.newbyteorder("="),copy=False) for k in retain}))
            if lo%30_000_000==0:
                print(f"Xiao rows scanned: {min(lo+1_000_000,n):,}/{n:,}",flush=True)
        schema={c.name:dict(format=c.format,unit=c.unit) for c in hdu.columns}
    coverage={}
    for name,box in boxes.items():
        ground=pd.concat(chunks[name],ignore_index=True) if chunks[name] else pd.DataFrame(columns=retain)
        ground.to_csv(work/f"xiao_{name}.csv",index=False)
        selected=inside(combined.ra,combined.dec,box)
        subset=ground.loc[inside(ground[identifiers["ra"]],ground[identifiers["dec"]],box)]
        coverage[name]=dict(xiao_sources=subset[identifiers["pm_source"]].value_counts().to_dict(),
                            spectra=combined.loc[selected,"origin"].value_counts().to_dict(),
                            reference_r_21_23=int((selected & combined.reference_r.between(21,23,inclusive="left")).sum()))
    save_json(work/"catalogue_audit.json",dict(rows=n,bytes=legacy.stat().st_size,xiao_columns=schema,
        xiao_sources=all_flags,spectra=counts,boxes=coverage))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("work","legacy","sdss","desi","config"):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        parser.error("Run on a Slurm compute node")
    args.work.mkdir(parents=True,exist_ok=True)
    boxes=tomllib.loads(args.config.read_text())["boxes"]
    outputs=["catalogue_audit.json","spectra_union.csv","spectra_veto.csv",*(f"xiao_{k}.csv" for k in boxes)]
    with run_stage(args.work,args.config,"catalogues","exposure-catalogues-v09",outputs=outputs) as go:
        if go:
            catalogue_audit(args.work,args.legacy,args.sdss,args.desi,boxes)


if __name__=="__main__":
    main()
