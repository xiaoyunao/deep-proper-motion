"""Census noncanonical archive coordinates; retain actual identifiers only on PScratch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.time import Time

from .access import Cache
from .stripe82_external_v24 import pack
from .io import digest,save_json


def run(base):
    root=base/"v2.4/meridian";root.mkdir(exist_ok=True,parents=True)
    cache=Cache(root/"queries")
    invalid="(ra<0 OR ra>=360)"
    count=cache.query("nsc","SELECT count(*) AS n FROM nsc_dr2.object WHERE "+invalid,expected=1)
    n=int(count.n.iat[0])
    q="id,ra,dec,mjd,pmra,pmdec,pmraerr,pmdecerr,deltamjd,ndet"
    bins=cache.query("nsc","SELECT floor(ra)::int a, floor((dec+90)/2)::int d, count(*) AS n FROM nsc_dr2.object WHERE "+invalid+" GROUP BY 1,2")
    if bins.n.sum()!=n or bins.isna().any().any():
        raise ValueError("Noncanonical sky histogram failed its denominator")
    def fetch(row):
        a,d,count=int(row.a),2*int(row.d)-90,int(row.n)
        where=f"ra>={a} AND ra<{a+1} AND dec>={d} AND dec<{d+2} AND "+invalid
        if -360<=a<720 and -90<=d<=88:
            where=f"q3c_radial_query(ra,dec,{(a+.5)%360},{d+1},1.51) AND "+where
        if count>60000:
            from .access import checked_spatial
            return checked_spatial(cache,"nsc",q,"nsc_dr2.object",where,d,d+2)
        return cache.query("nsc","SELECT "+q+" FROM nsc_dr2.object WHERE "+where,expected=count)
    workers=min(4,int(os.getenv("SLURM_CPUS_PER_TASK","1")))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        df=pd.concat(pool.map(fetch,bins.sort_values("n",ascending=False).itertuples()),ignore_index=True)

    if len(df)!=n or df.id.isna().any() or df.id.duplicated().any():
        raise ValueError("Noncanonical objects do not close with global server count")
    actual=df.ra.between(-360,720)&df.dec.between(-90,90)
    df["ra_normalized"]=np.mod(df.ra,360.)
    df.to_parquet(root/"private_noncanonical_object.parquet",index=False)
    s=df[actual & (abs(df.dec)<=2.2) & ((df.ra_normalized<53)|(df.ra_normalized>=307))].copy()
    epoch=np.full(len(s),np.nan);valid=s.mjd.between(50000,62000).to_numpy()
    epoch[valid]=Time(s.mjd.to_numpy()[valid],format="mjd",scale="utc").tt.jyear
    official=pack(s.id,s.ra_normalized,s.dec,s.pmra,s.pmdec,s.pmraerr,s.pmdecerr,epoch,s.deltamjd.to_numpy()>0)
    name=root/"private_nsc_longitude_supplement.parquet"
    official.to_parquet(name,index=False,compression="zstd")
    result=dict(noncanonical_total=n,legal_representations=int(actual.sum()),s82_supplement=len(official),
                longitude_range_deg=[float(df.ra.min()),float(df.ra.max())] if n else None,
                epoch_supplement_sha256=digest(name),input_sha256=digest(root/"private_noncanonical_object.parquet"))
    save_json(root/"v24_archive_longitude_summary.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm node")
    run(a.base)
