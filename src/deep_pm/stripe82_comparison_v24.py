"""Same-HLC, common-epoch stellar validation of an explicitly versioned frozen catalogue."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

from .exposure_catalogues import coordinates, inside
from .io import digest, save_json
from .stream_json import array_items
from .year_opportunities import utc_year

EDGES = (16, 18, 19, 20, 21, 21.5, 22, 22.5, 23, 24)
METHODS = ("ours", "xiao", "nsc")
ARCSEC = 180 * 3600 / np.pi


def nearest(query, catalogue):
    """Two angular distances and nearest index, including the one-object case."""
    distances, indices = cKDTree(catalogue).query(query, k=2)
    good = np.isfinite(distances)
    distances[good] = 2 * np.arcsin(np.minimum(1., distances[good] / 2)) * ARCSEC
    return indices[:, 0], distances


PAIR_STATES = (
    "outside_radius", "forward_competition", "not_mutual", "reverse_competition",
    "invalid_catalogue_motion", "matched",
)


def pair(stars, other, *, diagnostics=False):
    """Apply position competition to every candidate before motion validity."""
    status = np.zeros(len(stars), dtype=np.uint8)
    i = j = np.empty(0, dtype=int)
    if len(stars) and len(other):
        h = coordinates(stars.ra, stars.dec)
        c = coordinates(other.ra, other.dec)
        forward, fd = nearest(h, c)
        reverse, rd = nearest(c, h)
        eligible = fd[:, 0] <= .7
        status[eligible] = 1
        eligible &= fd[:, 1] - fd[:, 0] >= .3
        status[eligible] = 2
        eligible &= reverse[forward] == np.arange(len(stars))
        status[eligible] = 3
        eligible &= rd[forward, 1] - rd[forward, 0] >= .3
        status[eligible] = 4
        eligible &= other.valid.to_numpy()[forward]
        status[eligible] = 5
        i = np.flatnonzero(eligible)
        j = forward[i]
    return (i, j, status) if diagnostics else (i, j)


def stage(db, directory, name, query):
    dest = directory / name
    db.execute(f"COPY (SELECT *,floor(ra)::INTEGER sector FROM ({query}) q) "
               f"TO '{dest}' (FORMAT PARQUET, PARTITION_BY (sector), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)")
    return dest


def sector(directory, centre):
    files = [p for i in (-1, 0, 1)
             for p in (directory / f"sector={(centre+i)%360}").glob("*.parquet")]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    # The equatorial strip has cos(dec)>0.99. 36 arcsec padding includes competitors
    # in both directions that can affect either the 0.7 or the 0.3 arcsec decision.
    return df.loc[abs((df.ra - centre - .5 + 180) % 360 - 180) <= .51].reset_index(drop=True)


def binomial(k, n):
    if n == 0:
        return dict(n=0, k=0)
    z = norm.ppf(.975)
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return dict(n=int(n), k=int(k), rate=p, wilson95=[max(0.,centre-width),min(1.,centre+width)])


def stats(df, method):
    d = df[[method+"_e",method+"_n"]].to_numpy() - df[["pm_e","pm_n"]].to_numpy()
    if not len(d):
        return dict(n=0)
    r = np.hypot(*d.T)
    med = np.median(d,axis=0)
    return dict(n=len(d),median=med.tolist(),nmad=(1.4826*np.median(abs(d-med),axis=0)).tolist(),
                radial_p68=float(np.quantile(r,.68)),tail_gt20=binomial(int((r>20).sum()),len(r)))


def comparison(df):
    ok = {m:df[m+"_e"].notna() for m in METHODS}
    common = df.loc[ok["ours"] & ok["xiao"] & ok["nsc"]]
    return dict(parent=len(df),matched={m:binomial(int(g.sum()),len(df)) for m,g in ok.items()},
                identity_outcomes={m:dict(zip(PAIR_STATES, np.bincount(df[m+"_status"].to_numpy(), minlength=len(PAIR_STATES)).tolist())) for m in METHODS},
                common=len(common),delta_to_hlc={m:stats(common,m) for m in METHODS},
                hlc_median_pm_errors=(common[["se_e","se_n"]].median().tolist() if len(common) else None))


def run_years(sample, source_root):
    """Read only member records that enter the frozen common stellar sample."""
    common = sample.ours_e.notna() & sample.xiao_e.notna() & sample.nsc_e.notna() & sample.in_nominal
    sample["sdss_years"] = np.nan
    groups = {}
    calendar = {}
    for name, subset in sample[common].groupby("origin_core"):
        folder = source_root / str(name)
        path = folder / "private_standard_motion.json.gz"
        receipt = json.loads((folder/"v22_source_core_summary.json").read_text())
        if digest(path) != receipt["private_output_sha256"][path.name]:
            raise ValueError("Changed signed source members")
        slots = dict(zip(subset.origin_slot.astype(int), subset.index))
        for slot, row in enumerate(array_items(path)):
            if slot in slots:
                idx = slots.pop(slot)
                years = row["information"].get("sdss_unique_UTC_years")
                if years is None:
                    times = [t for e,t in zip(row["exposures"],row["observation_mjd_tt"]) if e.startswith("SDSS:")]
                    unseen = sorted(set(times)-calendar.keys())
                    if unseen:
                        calendar.update(zip(unseen,utc_year(np.asarray(unseen))))
                    years = len({calendar[t] for t in times})
                sample.at[idx,"sdss_years"] = years
                runs = sorted({v for v in row["exposures"] if v.startswith("SDSS:")})
                if len(runs) != row["information"]["sdss_physical_runs"]:
                    raise ValueError("SDSS member identity does not close")
                groups[idx] = runs
        if slots:raise ValueError("Missing signed source slot")
    return groups


def run_sensitivity(sample, membership):
    if len(sample)<20:return dict(n=len(sample))
    from scipy.sparse import csr_matrix
    labels={g:i for i,g in enumerate(sorted({v for k in sample.index for v in membership[k]}))}
    if len(labels)<2:return dict(n=len(sample),runs=len(labels))
    rr=[];cc=[];vv=[]
    for i,k in enumerate(sample.index):
        for g in membership[k]:
            rr.append(i);cc.append(labels[g]);vv.append(1/len(membership[k]))
    incidence=csr_matrix((vv,(rr,cc)),shape=(len(sample),len(labels)))
    rng=np.random.RandomState(240082)
    weights=incidence@rng.exponential(size=(len(labels),64))
    result={}
    for method in METHODS:
        radius=np.hypot(sample[method+"_e"]-sample.pm_e,sample[method+"_n"]-sample.pm_n).to_numpy()
        order=np.argsort(radius)
        cumulative=weights[order].cumsum(axis=0)
        quantile=radius[order][np.argmax(cumulative>=.68*cumulative[-1],axis=0)]
        result[method]=np.quantile(quantile,[.025,.975]).tolist()
    return dict(n=len(sample),runs=len(labels),positive_run_reweight_P68_95=result,
                meaning="Sensitivity to common physical SDSS run weights, not an independent-error interval")


def catalogue_closure(catalogue, plan):
    directory=catalogue.parent
    closure=directory/"v22_catalogue_ownership_summary.json"
    binding=directory/"private_closure_binding.json"
    ownership=json.loads(closure.read_text())
    if (catalogue.resolve()!=(directory/"catalogue.duckdb").resolve()
            or digest(binding)!=ownership["private_input_binding_sha256"]
            or json.loads(binding.read_text())["plan_sha256"]!=digest(plan)):
        raise ValueError("Stripe 82 catalogue ownership and physical registration disagree")
    return digest(closure),ownership["retained_common_epoch_solutions"]


def run(base, catalogue, plan, output, version, workers):
    work=base/"v2.4"
    output.mkdir(exist_ok=True,parents=True)
    closure_sha, retained=catalogue_closure(catalogue,plan)
    receipts=[work/"s82/v24_s82_parent_summary.json",work/"s82-nsc/v24_nsc_epoch_input_summary.json",
              work/"s82-xiao-epoch/v24_xiao_epoch_input_summary.json", plan,work/"meridian/v24_archive_longitude_summary.json"]
    opportunity=work/"s82-opportunity"
    opportunity_receipt=opportunity/"v24_hlc_geometric_opportunity_summary.json"
    receipts.append(opportunity_receipt)
    table=opportunity/"private_hlc_opportunity.parquet"
    if digest(table)!=json.loads(opportunity_receipt.read_text())["sha256"]:
        raise ValueError("Changed HLC moving-position footprint")
    archive=json.loads(receipts[2].read_text())["outputs"]
    native=json.loads(receipts[1].read_text())["parts"]
    checked=[(work/"s82-xiao-epoch"/name,sha) for name,sha in archive.items()]
    checked.extend((work/"s82-nsc"/entry["file"],entry["sha256"]) for entry in native)
    for folder,pattern,files in ((work/"s82-xiao-epoch","private_xiao_*.parquet",set(archive)),
                                 (work/"s82-nsc","private_nsc_*.parquet",{row["file"] for row in native})):
        if {p.name for p in folder.glob(pattern)}!=files:raise ValueError("Unexpected external catalogue shard set")
    for path,sha in checked:
        if digest(path)!=sha:raise ValueError("Changed signed external catalogue shard")
    ticket=dict(closure_sha256=closure_sha,inputs=[digest(p) for p in receipts],catalogue_stat=[catalogue.stat().st_size,catalogue.stat().st_mtime_ns],
                protocol=json.loads(receipts[0].read_text())["protocol_sha256"],implementation=digest(Path(__file__)),version=version)
    state=output/"private_input_signature.json"
    if state.exists() and json.loads(state.read_text())!=ticket:
        raise ValueError("Changed scientific comparison inputs, use a separate version directory")
    save_json(state,ticket)
    complete=output/"v24_stripe82_comparison_summary.json"
    if complete.exists() and digest(output/"private_pairs.parquet")==json.loads(complete.read_text())["pair_sha256"]:
        return
    import duckdb
    db=duckdb.connect()
    db.execute(f"SET threads={workers}")
    db.execute("SET memory_limit='6GB'")
    db.execute(f"ATTACH '{catalogue}' AS scientific (READ_ONLY)")
    if db.execute("SELECT count(*) FROM scientific.accepted").fetchone()[0]!=retained:
        raise ValueError("Stripe 82 comparison index disagrees with its accepted source total")
    queries={"hlc":f"SELECT * FROM read_parquet('{work}/s82/private_hlc_*.parquet') WHERE has_position",
             "ours":"SELECT src AS external,ra_deg AS ra,dec_deg AS dec,TRUE AS valid,pm_e,pm_n,se_e,se_n,core,slot,sdss_runs,nsc_span,joint_span,nsc_nights FROM scientific.accepted WHERE abs(dec_deg)<=2.2 AND (ra_deg<53 OR ra_deg>=307)",
             "xiao":f"SELECT * FROM read_parquet('{work}/s82-xiao-epoch/private_xiao_*.parquet')",
             "nsc":f"SELECT * FROM read_parquet('{work}/s82-nsc/private_nsc_*.parquet')"}
    supplement=work/"meridian/private_nsc_longitude_supplement.parquet"
    if digest(supplement)!=json.loads(receipts[4].read_text())["epoch_supplement_sha256"]:
        raise ValueError("Unverified NSC meridian supplement")
    queries["nsc"] += f" UNION ALL SELECT * FROM read_parquet('{supplement}')"
    duplicate=db.execute(f"SELECT external FROM ({queries['nsc']}) x GROUP BY external HAVING count(*)>1 LIMIT 1").fetchone()
    if duplicate:raise ValueError("Nonunique external NSC ID after longitude closure")
    directories={k:stage(db,output,k,q) for k,q in queries.items()}
    spec=[row["spec"] for row in json.loads(plan.read_text())["cores"]]
    parts=[]
    for centre in sorted(int(d.name.split("=")[1]) for d in directories["hlc"].glob("sector=*")):
        stars=sector(directories["hlc"],centre)
        if not len(stars):continue
        h=coordinates(stars.ra,stars.dec)
        _,separations=nearest(h,h)
        stars["neighbour"]=separations[:,1]
        for m in METHODS:
            other=sector(directories[m],centre)
            i,j,stars[m+"_status"]=pair(stars,other,diagnostics=True)
            for source,target in (("pm_e",m+"_e"),("pm_n",m+"_n"),("se_e",m+"_se"),("se_n",m+"_sn")):
                stars[target]=np.nan
                if len(i):stars.loc[i,target]=other[source].to_numpy()[j]
            if m=="ours":
                stars["origin_core"]=None
                stars["origin_slot"]=np.nan
                if len(i):
                    stars.loc[i,"origin_core"]=other.core.to_numpy()[j]
                    stars.loc[i,"origin_slot"]=other.slot.to_numpy()[j]
                for key in ("sdss_runs","nsc_span","joint_span","nsc_nights"):
                    stars[key]=np.nan
                    if len(i):stars.loc[i,key]=other[key].to_numpy()[j]
        stars=stars.loc[stars.parent & (np.floor(stars.ra)==centre)].copy()
        region=np.zeros(len(stars),bool)
        for s in spec:
            if s["dec"][0]<=2.2 and s["dec"][1]>=-2.2 and ((s["ra"][0]-centre+180)%360-180)<2 and ((s["ra"][1]-centre+180)%360-180)>-2:
                region |= inside(stars.ra,stars.dec,s).to_numpy()
        stars["in_nominal"]=region
        parts.append(stars)
    all_stars=pd.concat(parts,ignore_index=True)
    if all_stars.identity.duplicated().any() or len(all_stars)!=json.loads(receipts[0].read_text())["parent"]:
        raise ValueError("Stripe 82 HLC parent count or unique source identity did not close")
    all_stars=all_stars.merge(pd.read_parquet(table),on="identity",how="left",validate="one_to_one")
    if all_stars.time_opportunity.isna().any():raise ValueError("HLC opportunity lookup incomplete")
    pairpath=output/"private_pairs.parquet"
    membership=run_years(all_stars,catalogue.parents[1]/"sources")
    all_stars.to_parquet(pairpath,index=False,compression="zstd")
    nominal=all_stars[all_stars.in_nominal]
    sample=nominal[nominal.time_opportunity]
    result=dict(scope=version,closure_sha256=closure_sha,area_definition="Frozen owner rectangles intersected with moving-source native field/CCD and time opportunity. Completeness conditional on HLC stellar parent.",
                input_signature=digest(state),pair_sha256=digest(pairpath),all_hlc_parent=len(all_stars),in_nominal=len(nominal),
                in_nominal_geometry=int(nominal.geometry.sum()),in_nominal_time=len(sample),nominal_comparison=comparison(nominal),
                total=comparison(sample),magnitude={},condition={})
    for lo,hi in zip(EDGES,EDGES[1:]):
        sub=sample[(sample.r>=lo)&(sample.r<hi)]
        entry=comparison(sub)
        common=sub[sub.ours_e.notna() & sub.xiao_e.notna() & sub.nsc_e.notna()]
        entry["common_run_sensitivity"]=run_sensitivity(common,membership)
        result["magnitude"][f"{lo:g}-{hi:g}"]=entry
    for key,edges in (("sdss_runs",(1,2,5,np.inf)),("sdss_years",(1,2,3,np.inf)),("joint_span",(0,10,15,20,np.inf)),
                      ("nsc_span",(0,2,4,6,np.inf)),("nsc_nights",(0,5,10,30,np.inf)),
                      ("colour",(-np.inf,.5,1,2,np.inf)),("neighbour",(0,2,5,10,np.inf))):
        result["condition"][key]={f"{lo:g}-{hi:g}":comparison(sample[(sample[key]>=lo)&(sample[key]<hi)]) for lo,hi in zip(edges,edges[1:])}
    strict=sample[sample.ours_e.notna() & sample.xiao_e.notna() & sample.nsc_e.notna()]
    ed=np.linspace(-50,50,101)
    result["density"]={"edges_masyr":ed.tolist(),"sample":len(strict),"counts":{}}
    for axis in ("e","n"):
        for method in METHODS:
            histogram,_,_=np.histogram2d(strict["pm_"+axis],strict[method+"_"+axis],bins=(ed,ed))
            result["density"]["counts"][method+"_"+axis]=histogram.astype(int).tolist()
    save_json(complete,result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("base","catalogue","plan","output"):p.add_argument(name,type=Path)
    p.add_argument("version")
    args=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm compute node")
    run(**vars(args),workers=int(os.environ["SLURM_CPUS_PER_TASK"]))
