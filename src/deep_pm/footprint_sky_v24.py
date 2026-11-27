"""Integrate the registered CCD/field overlap into equal identity-free sky cells."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .footprint_manifest_v24 import layers
from .io import digest,save_json


def aggregate(base):
    root=base/"v2.4"
    inventory=root/"registration/v24_footprint_inventory_summary.json"
    proof=json.loads(inventory.read_text())
    bins=40
    early0,early1,_,_,_=layers(root/"geometry","early",bins)
    late0,late1,_,_,_=layers(root/"geometry","late",bins)
    area=np.zeros((180,360,3),float)
    for i in range(180):
        s=slice(i*bins,(i+1)*bins)
        lat=np.linspace(i-90,i-89,bins+1)
        w=np.diff(np.sin(np.deg2rad(lat)))*180/np.pi/bins
        joint=np.isfinite(early0[s])&np.isfinite(late0[s])
        use=joint & ((late1[s]-late0[s])>=.6*365.25) & ((np.maximum(late1[s],early1[s])-np.minimum(late0[s],early0[s]))>=5*365.25)
        for k,m in enumerate((joint,use,use&(np.maximum(late1[s],early1[s])-np.minimum(late0[s],early0[s])>=15*365.25))):
            area[i,:,k]=np.einsum("i,ijk->j",w,m.reshape(bins,360,bins))
    sums=area.sum(axis=(0,1))
    for k,name in enumerate(("intersection","time")):
        np.testing.assert_allclose(sums[k],proof["grids"]["40"]["areas_sqdeg"][name],rtol=1e-10)
    cells=np.argwhere(area[:,:,0]>0.)
    save_json(root/"registration/v24_geometric_sky_summary.json",dict(signature=digest(inventory),
              columns=["ra_low","dec_low","intersection_sqdeg","time_sqdeg","time_and_15_year_sqdeg"],
              cells=[[int(j),int(i)-90,*area[i,j]] for i,j in cells],
              total_sqdeg=sums,definition="Fully aggregated spherical area in 1-degree display cells; 0.025-degree integration of actual observation boundaries."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm CPU node")
    aggregate(a.base)
