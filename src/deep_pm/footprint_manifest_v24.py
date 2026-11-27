"""Pre-register full sky geometrical opportunities without opening any source motion."""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from .io import digest,save_json
from .continuous_summary import area


def read(path):
    return json.loads(path.read_text())


def layers(geometry,kind,bins):
    ny=180*bins;nx=360*bins
    lo=np.full(ny*nx,np.inf);hi=np.full(ny*nx,-np.inf)
    blo=np.full((180,360),np.inf);bhi=-blo.copy()
    dirs=read(geometry/f"private_{kind}_{bins}.json")["directories"]
    rows=invalid=0;bindings=[]
    for name in dirs:
        directory=Path(name);proof=read(directory/"receipt.json");path=directory/"samples.npz"
        if proof["files"][path.name]!=digest(path):
            raise ValueError("Changed geometry raster")
        rows+=proof["rows"];invalid+=proof["invalid"];bindings.append(digest(directory/"receipt.json"))
        with np.load(path) as d:
            ix=d["ids"];lo[ix]=np.minimum(lo[ix],d["first"]);hi[ix]=np.maximum(hi[ix],d["last"])
            np.minimum(blo,d["low"],out=blo);np.maximum(bhi,d["high"],out=bhi)
    return lo.reshape(ny,nx),hi.reshape(ny,nx),blo,bhi,dict(rows=rows,invalid=invalid,bindings=bindings)


def minus_old(a,d,old):
    # Break only at an actual frozen ownership edge; interiors are disjoint.
    rect=[(a,a+1,d,d+1)]
    for w,e,s,n in old:
        nxt=[]
        for x,y,u,v in rect:
            i,j=max(x,w),min(y,e);k,l=max(u,s),min(v,n)
            if i>=j or k>=l:
                nxt.append((x,y,u,v));continue
            for box in ((x,i,u,v),(j,y,u,v),(i,j,u,k),(i,j,l,v)):
                if box[0]<box[1] and box[2]<box[3]:nxt.append(box)
        rect=nxt
    return rect


def run(base,output):
    geometry=base/"v2.4/geometry"
    oldpath=base/"v2.2/registration/private_full_coordinate_plan.json"
    oldsha=digest(oldpath)
    if oldsha!=read(oldpath.parent/"v22_registration_summary.json")["plan_sha256"]:
        raise ValueError("Changed old footprint")
    boxes={}
    for item in read(oldpath)["cores"]:
        spec=item["spec"];w,e=spec["ra"];s,n=spec["dec"]
        if not 0<e-w<=3:raise ValueError("Unexpected legacy RA interval")
        for j in range(int(np.floor(w)),int(np.ceil(e))):
            for i in range(int(np.floor(s)),int(np.ceil(n))):
                shift=j-j%360
                boxes.setdefault((j%360,i),[]).append((w-shift,e-shift,s,n))
    grids={}
    for bins in (20,40):
        e0,e1,el,eh,edata=layers(geometry,"early",bins)
        n0,n1,nl,nh,ndata=layers(geometry,"late",bins)
        physical=(eh>=el)&(nh-nl>=.6*365.25)&(np.maximum(eh,nh)-np.minimum(el,nl)>=5*365.25)
        cells={};totals={k:0. for k in ("sdss","nsc","intersection","time","old_time","new_time")}
        for i in range(180):
            sl=slice(i*bins,(i+1)*bins)
            lat=np.linspace(i-90,i-89,bins+1)
            weight=np.diff(np.sin(np.deg2rad(lat)))*180/np.pi/bins
            early=np.isfinite(e0[sl]);late=np.isfinite(n0[sl]);joint=early&late
            use=joint&((n1[sl]-n0[sl])>=.6*365.25)&((np.maximum(e1[sl],n1[sl])-np.minimum(e0[sl],n0[sl]))>=5*365.25)
            for k,m in (("sdss",early),("nsc",late),("intersection",joint),("time",use)):
                totals[k]+=float(np.sum(m.sum(axis=1)*weight))
            for j in np.flatnonzero(physical[i]):
                j=int(j);a=j;d=i-90
                rects=minus_old(a,d,boxes.get((a,d),[]))
                section=use[:,j*bins:(j+1)*bins]
                all_area=float(np.sum(section.sum(axis=1)*weight))
                ra=a+(np.arange(bins)+.5)/bins
                dec=d+(np.arange(bins)+.5)/bins
                for number,(w,e,s,n) in enumerate(rects):
                    sub=section&(ra>=w)&(ra<e)&((dec>=s)&(dec<n))[:,None]
                    sampled=float(np.sum(sub.sum(axis=1)*weight))
                    label=f"v24_{a:03d}_{i:03d}_{number:02d}"
                    cells[label]=dict(ra=[w,e],dec=[s,n],time_area=sampled)
                    totals["new_time"]+=sampled
        totals["old_time"]=totals["time"]-totals["new_time"]
        grids[str(bins)]=dict(areas_sqdeg=totals,sdss=edata,nsc=ndata)
        if bins==20:coarse_keys=set(cells)
        elif coarse_keys!=set(cells):raise ValueError("Conservative processing envelope changed with quadrature")
    output.mkdir(exist_ok=True,parents=True)
    prospective=output/"private_full_footprint_plan.json"
    plan=dict(baseline_sha256=oldsha,prior_cores=len(read(oldpath)["cores"]),new=cells,
              definition="All overlapping early/late spherical bounding caps in 1-degree cells with potential time support. Includes subgrid and false-positive boundaries, evaluated at intake; disjoint from frozen old rectangles.",
              geometry=grids)
    if prospective.exists() and read(prospective)!=plan:raise ValueError("Changed frozen production manifest")
    save_json(prospective,plan)
    result=dict(signed_manifest_sha256=digest(prospective),new_rectangles=len(cells),
                new_rectangles_with_sampled_time=sum(x["time_area"]>0 for x in cells.values()),
                new_bounding_area_sqdeg=sum(area(c) for c in cells.values()),grids=grids,
                scope="Pre-reference native scan and CCD opportunity, two independent area quadratures. Includes all actual filter labels. Bounding area includes slivers/empty cells. Final accepted reference area and produced footprint require separate accounting.")
    save_json(output/"v24_footprint_inventory_summary.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("base",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm compute node")
    run(a.base,a.output)
