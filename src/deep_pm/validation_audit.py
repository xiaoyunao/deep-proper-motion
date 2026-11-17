"""Independent bright-star, dark selection, and reference-graph diagnostics."""
import healpy as hp
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import access
from .astrometry import unproject
from .exposure_preflight import where_box


def gaia_bright(cat, r):
    required = cat.phot_g_mean_mag.lt(20) & cat.visibility_periods_used.ge(9)
    required &= cat.pmra_error.lt(.5) & cat.pmdec_error.lt(.5)
    return required.to_numpy() & np.isfinite(r) & (r < 21)


def gaia_observation_quality(cache, box):
    halo = .06
    low, high = box["dec"]
    return access.checked_spatial(cache, "nsc", "source_id,visibility_periods_used",
        "gaia_dr3.gaia_source", where_box(box, halo), low-halo-1e-6, high+halo+1e-6)


def quantiles(v):
    v = np.asarray(v)
    return np.quantile(v, [.16, .5, .84], axis=0) if len(v) else None


def dark_quality(records):
    values = np.asarray(records, dtype=float).reshape(-1, 9)
    a, b, finite = [values[:,i].astype(bool) for i in range(3)]
    outcome = {"passed":~a & ~b, "formal_only":a & ~b, "residual_only":~a & b, "both":a & b}
    return dict(calibrated=len(values), nonfinite_formal=int((~finite).sum()),
        depth_limit="Median r among contributing NSC detections; this is not the exposure completeness limit.",
        quantile_order=["maximum_axis_formal_masyr","radial_normalized_residual_p68","baseline_years",
                        "earliest_sdss_run_leverage_fraction","nsc_visits","nsc_median_r"],
        by_reason={label:dict(n=int(mask.sum()), p16_median_p84=quantiles(values[mask & finite,3:]))
                   for label,mask in outcome.items()},
        formal_limit_sensitivity={str(limit):int(np.count_nonzero(finite & ~b & (values[:,3]<=limit)))
                                  for limit in [5,10,15,20]})


def graph_support(refs, keys, solution):
    """Describe measured links and applied support for calibrated visits."""
    subset = refs.loc[~refs.is_holdout]
    oi, objects = pd.factorize(subset.reference, sort=False)
    gi, groups = pd.factorize(keys[subset.index], sort=False)
    matrix = coo_matrix((np.ones(len(gi),dtype=np.int8),
                         (oi, len(objects)+gi)),shape=(len(objects)+len(groups),)*2)
    components, labels = connected_components(matrix.tocsr(), directed=False)
    group_count = np.bincount(labels[len(objects):], minlength=components)
    ref_count = np.bincount(labels[:len(objects)], minlength=components)
    fitted = np.array([k in solution for k in groups])
    fitted_by_component = np.bincount(labels[len(objects):][fitted], minlength=components)
    largest = sorted(zip(group_count,ref_count,fitted_by_component), reverse=True)[:10]
    summaries = {}
    for family in ("SDSS", "NSC"):
        children = [v for k,v in solution.items() if k.startswith(family+":") and "|" in k]
        summaries[family] = dict(count=len(children), trained_objects=quantiles([v[2] for v in children]),
            bootstrap_se_mas=quantiles([np.std(v[1],axis=0,ddof=1) for v in children]))
    return dict(components=components, largest_components_groups_references_and_supported=largest,
                fine_observation_support=summaries)


def boundary_pairs(refs, data, xy, original_ids, obs, solutions, center, spec):
    """Evaluate genuine repeated detector entries and cross-boundary visit links."""
    groups=[]
    sid=pd.Series(original_ids.to_numpy(), index=data.index)
    # Each held reference may have archive rows discarded by the exposure
    # de-duplication used in calibration. Rejoin those rows only after fitting.
    held=refs.loc[refs.is_holdout & data.survey.loc[refs.index].eq("NSC")]
    candidates=sid.isin(held.reference.unique()) & data.survey.eq("NSC")
    common=np.isin(obs[0],list(solutions[0]))&np.isin(obs[1],list(solutions[1]))
    subset=data.loc[candidates & common,["exposure_id","detector_id"]].assign(reference=sid)
    subsets={}
    for (_, _), table in subset.groupby(["reference","exposure_id"],sort=False):
        if table.detector_id.nunique()<2:
            continue
        ix=table.index.to_numpy()
        for i in range(len(ix)):
            for j in range(i+1,len(ix)):
                a,b=ix[i],ix[j]
                if (data.detector_id.iat[a]!=data.detector_id.iat[b]
                        and np.linalg.norm(xy[a]-xy[b])<=1000):
                    groups.append((a,b))
    if groups:
        subsets["nsc_same_exposure_ccd"]=np.asarray(groups)
    original=refs.loc[refs.is_holdout & common[refs.index]]
    pairs=near_boundaries(original,data,xy,center,spec)
    subsets.update(pairs)
    output={}
    for tag,pair in subsets.items():
        a,b=pair.T
        before=xy[a]-xy[b]
        values=[before]
        for keys,fit in zip(obs,solutions):
            offset=np.stack([fit[k][0] for k in keys[a]])-np.stack([fit[k][0] for k in keys[b]])
            values.append(before-offset)
        output[tag]=dict(pair_count=len(pair), median_mas=np.median(values,axis=1),
                         radial_p68_mas=np.quantile(np.linalg.norm(values,axis=2),.68,axis=1))
    return output


def near_boundaries(refs, data, xy, center, spec):
    pairs={}
    sky=unproject(xy[refs.index],center)
    pixel=hp.ang2pix(128,*sky.T,lonlat=True)
    # Probe a 2 arcmin strip on each side of an actual RING edge.
    width=2/60
    displaced=np.column_stack([hp.ang2pix(128,sky[:,0]+a,sky[:,1]+b,lonlat=True)
        for a,b in ((width,0),(-width,0),(0,width),(0,-width))])
    close=(displaced!=pixel[:,None]).any(axis=1)
    # The longitudinal seam bisects each frozen pilot and agrees with
    # acquisition seams in their overlap. It is evaluated in one shared fit.
    seam=np.mean(spec["ra"])
    seam_close=abs(sky[:,0]-seam)<=width
    ns=data.survey.eq("NSC").to_numpy()
    for name, mask in (("near_healpix_edge",close),("near_partition_seam",seam_close)):
        rows=refs.loc[mask]
        selected=[]
        for _, part in rows.groupby("reference",sort=False):
            ix=part.index.to_numpy()
            early,late=ix[~ns[ix]],ix[ns[ix]]
            if len(early) and len(late):
                selected.append((early[np.argmin(np.linalg.norm(xy[early]-np.median(xy[early],axis=0),axis=1))],
                                 late[np.argmin(np.linalg.norm(xy[late]-np.median(xy[late],axis=0),axis=1))]))
        if selected:
            pairs[name]=np.asarray(selected)
    return pairs
