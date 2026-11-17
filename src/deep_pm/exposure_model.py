"""Fit detector offsets with object-level bootstrap and parent constraints."""
import os

import numpy as np
import pandas as pd

from .precision_v03 import calibration_groups
from .zero_point import fit_offsets


def observation_keys(data):
    parent=calibration_groups(data)
    sdss=data.survey.eq("SDSS").to_numpy()
    field=pd.to_numeric(data.get("field_id"),errors="coerce") if "field_id" in data else None
    if field is None:
        field=pd.Series(np.nan,index=data.index)
    fallback=pd.to_numeric(data.detection_id.str.extract(r"^SDSS:\d+:\d+:(\d+):\d+$")[0],errors="coerce")
    field=field.fillna(fallback)
    detector=pd.to_numeric(data.detector_id,errors="coerce")
    local=field.where(sdss,detector)
    fine=np.array([f"{p}|{int(k)}" if np.isfinite(k) else p for p,k in zip(parent,local)])
    return parent,fine


def hierarchy(data,refs,xy,sigma,config,settings,c0,overlaps=None):
    parent,fine=observation_keys(data)
    train=refs.loc[~refs.is_holdout]
    i=train.index.to_numpy()
    counts=pd.DataFrame(dict(k=fine[i],reference=train.reference.to_numpy())).groupby("k").reference.nunique()
    minimum=settings["min_child_references"]
    retained={k:p for k,p in zip(fine[i],parent[i]) if k!=p and counts[k]>=minimum and p in c0}
    keys=np.array([k if k in retained else p for k,p in zip(fine,parent)])
    opts=config["calibration"]
    model=fit_offsets(train.reference,xy[i],sigma[i],keys[i],opts["min_references"],
                      opts["bootstrap_draws"],config["seed"],
                      workers=int(os.environ.get("DEEP_PM_M3_WORKERS","1")),
                      parents=retained,shrinkage_mas=settings["child_prior_mas"],overlaps=overlaps)
    # Use a common NSC parent gauge in each bootstrap draw. Positions from
    # different blocks may be merged only if their joint calibration is shared.
    anchor=[k for k in model if k.startswith("NSC:") and "|" not in k and k in c0]
    if not anchor:
        raise ValueError("C1 has no C0-supported NSC reference groups")
    datum=np.median(np.stack([model[k][1] for k in anchor]),axis=0)
    position_datum=np.median(np.stack([model[k][0]-c0[k][0] for k in anchor]),axis=0)
    offsets={}
    for k,(offset,draws,n) in model.items():
        error=draws-datum
        error-=error.mean(axis=0)
        if np.std(error,axis=0,ddof=1).max()<=opts["max_offset_se_mas"]:
            offsets[k]=(offset-position_datum,error,n)
    keys=np.array([k if k in offsets else p for k,p in zip(keys,parent)])
    # Parent fallback within C1 uses draws from the same connected fit. A
    # disconnected C0 estimate cannot be spliced into the C1 covariance.
    return keys,offsets,dict(c0_groups=len(c0),candidate_children=len(counts),
        supported_children=sum(k in offsets for k in retained),fitted_groups=len(model),
        supported_groups=len(offsets),fallback_rows=int(np.count_nonzero(keys!=fine)),
        usable_rows=int(np.count_nonzero(np.isin(keys,list(offsets)))))
