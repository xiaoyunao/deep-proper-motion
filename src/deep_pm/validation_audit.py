"""Independent bright-star, dark selection, and reference-graph diagnostics."""
import healpy as hp
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import access
from .astrometry import project, unproject
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


def anchored_bright_gaia(cat, anchors, xy, magnitude, center):
    """Decide brightness using NSC positions alone before reading any Gaia motion."""
    from scipy.spatial import cKDTree

    from .dark_precision import query_unique

    screened=cat.dropna(subset=["ra","dec"])
    bright_precheck=gaia_bright(screened,np.zeros(len(screened)))
    screened=screened.loc[bright_precheck].copy()
    if not len(anchors) or screened.empty:
        return screened.iloc[:0], dict(bright_position_anchors=0, excluded_faint_anchors=0)
    positions=np.asarray([np.median(xy[a],axis=0) for a in anchors])
    magnitudes=np.asarray([np.median(magnitude[a]) for a in anchors])
    match,index=query_unique(cKDTree(positions),project(screened.ra,screened.dec,center),1000,300)
    r=np.where(match,magnitudes[np.minimum(index,len(anchors)-1)],np.nan)
    bright=gaia_bright(screened,r)
    return screened.loc[bright], dict(bright_position_anchors=int(bright.sum()),
                                      excluded_faint_anchors=int(np.count_nonzero(r>=21)))


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


def boundary_pairs(refs, data, xy, original_ids, obs, solutions, center, spec, *, include_geometric=True):
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
    if include_geometric:
        original=refs.loc[refs.is_holdout & common[refs.index]]
        subsets.update(near_boundaries(original,data,xy,center,spec))
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
        if tag == "nsc_same_exposure_ccd":
            output[tag].update(held_references=sid.iloc[a].nunique(),
                               held_exposures=data.exposure_id.iloc[a].nunique())
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


def sdss_overlap_rows(sas, refs, data, xy, observation, solutions, center, *,
                      holdout, max_fields=160, match_mas=500, margin_mas=300,
                      search_radius_deg=1.7, require_fine_support=True):
    """Re-read adjacent photoObj fields before RUN_PRIMARY de-duplication."""
    from concurrent.futures import ThreadPoolExecutor

    from scipy.spatial import cKDTree

    from .astrometry import project
    from .dark_precision import query_unique
    from .sdss_sas import select_sdss_field

    opts = refs.loc[refs.is_holdout.eq(holdout) & data.survey.loc[refs.index].eq("SDSS")]
    keys = observation[0]
    fine = data.field_id.to_numpy()
    eligible = np.isin(keys[opts.index],list(solutions[0])) & np.isfinite(fine[opts.index])
    requests = {}
    for i in opts.index[eligible]:
        parent = keys[i]
        _, run, camcol = parent.split(":")
        for neighbor in (int(fine[i])-1,int(fine[i])+1):
            if neighbor >= 0:
                requests.setdefault((int(run),301,int(camcol),neighbor), []).append(i)
    # A deterministic cap keeps this a small targeted audit even in Stripe 82.
    ordered = sorted(requests, key=lambda k:(-len(requests[k]),k))[:max_fields]
    radius = np.deg2rad(search_radius_deg)
    def read(key):
        try:
            return select_sdss_field(sas,center,np.tan(radius)*206264806.247,key,run_primary=False)
        except FileNotFoundError:
            return None
    pairs=[]
    scanned=0
    runs=set()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for key, table in zip(ordered,pool.map(read,ordered)):
            if table is None:
                continue
            scanned+=1
            good=table["type"].eq(3)&table.nChild.eq(0)&table.offsetRa_r.eq(0)&table.offsetDec_r.eq(0)
            if not good.any():
                continue
            table = table.loc[good].reset_index(drop=True)
            counterpart=project(table.ra,table.dec,center)
            ix=np.asarray(requests[key],dtype=int)
            hit, j=query_unique(cKDTree(counterpart),xy[ix],match_mas,margin_mas)
            for a,b in zip(ix[hit],j[hit]):
                parent=keys[a]
                bkey=f"{parent}|{key[-1]}"
                if require_fine_support and (bkey not in solutions[1] or observation[1][a] not in solutions[1]):
                    continue
                raw=xy[a]-counterpart[b]
                pairs.append(dict(index=a, ref=refs.loc[a, "reference"], first=observation[1][a],
                                  second=bkey, run=key[0], counterpart=table.objID.iloc[b],
                                  east=raw[0], north=raw[1],
                                  err_e=table.raErr.iloc[b]*1000, err_n=table.decErr.iloc[b]*1000))
                runs.add(key[0])
    summary=dict(eligible_neighbor_fields=len(requests), selected_fields=len(ordered),
                 fields_read=scanned, pair_count=len(pairs), independent_run_count=len(runs))
    columns = ("index", "ref", "first", "second", "run", "counterpart", "east", "north", "err_e", "err_n")
    return pd.DataFrame(pairs, columns=columns), summary


def sdss_field_overlap(sas, refs, data, xy, observation, solutions, center, spec):
    pairs, result = sdss_overlap_rows(sas,refs,data,xy,observation,solutions,center,holdout=True)
    if len(pairs):
        c1 = solutions[1]
        delta = np.stack([c1[row.first][0]-c1[row.second][0] for row in pairs.itertuples()])
        raw = pairs[["east", "north"]].to_numpy()
        values = np.stack((raw, raw, raw-delta), axis=1)
        result.update(median_mas=np.median(values,axis=0),
                      radial_p68_mas=np.quantile(np.linalg.norm(values,axis=2),.68,axis=0))
    return result


def held_conditions(refs, data, xy, keys, offsets):
    """Summarize held-galaxy residuals within an NSC passband."""
    subset=refs.loc[refs.is_holdout & np.isin(keys[refs.index],list(offsets))]
    early=subset.loc[data.survey.loc[subset.index].eq("SDSS")]
    observed_color=data.color.where(data.color.between(-.3,3.5))
    colors=early.assign(gi=observed_color).groupby("reference").gi.median()
    subset=subset.loc[data.survey.loc[subset.index].eq("NSC")]
    ix=subset.index.to_numpy()
    corrected=xy[ix]-np.stack([offsets[g][0] for g in keys[ix]])
    clipped=pd.DataFrame(dict(reference=subset.reference.to_numpy(), band=data.band.iloc[ix].to_numpy(),
        east=corrected[:,0],north=corrected[:,1]))
    group=clipped.groupby(["reference","band"]).ngroup().to_numpy()
    n=np.bincount(group)[group]
    residual=corrected-leave_one_median(corrected,group)
    conditions=dict(airmass=data.airmass.iloc[ix], fwhm=data.fwhm.iloc[ix], x=data.x.iloc[ix],y=data.y.iloc[ix],
                    g_minus_i=clipped.reference.map(colors),r_magnitude=data.magnitude.iloc[ix])
    output={}
    for name,v in conditions.items():
        value=np.asarray(v)
        m=(n>=3)&np.isfinite(value)&clipped.band.eq("r").to_numpy()
        if name=="fwhm":
            m &= value>0
        if name=="airmass":
            m &= value>=1
        if np.count_nonzero(m)<100:
            continue
        limits=np.quantile(value[m],[0,.33,.67,1])
        bins=np.searchsorted(limits[1:-1],value,side="right")
        output[name]=dict(valid_detection_count=int(m.sum()), edges=limits, bins=[
            dict(n=int(selected.sum()), med_mas=np.median(residual[selected],axis=0),
                 radial_p68_mas=np.quantile(np.linalg.norm(residual[selected],axis=1),.68))
            if selected.any() else dict(n=0)
            for selected in (m & (bins==i) for i in range(3))])
    return output


def leave_one_median(values,group):
    """Return the per-column group median with the measurement itself removed."""
    counts=np.bincount(group)
    start=np.cumsum(counts)-counts
    size=counts[group]
    base=start[group]
    valid=size>=2
    result=np.full(values.shape,np.nan)
    for axis,value in enumerate(values.T):
        order=np.lexsort((value,group))
        rank=np.empty(len(order),dtype=int)
        rank[order]=np.arange(len(order))
        within=rank[valid]-base[valid]
        left=(size[valid]-2)//2
        right=(size[valid]-1)//2
        first=base[valid]+left+(within<=left)
        second=base[valid]+right+(within<=right)
        result[valid,axis]=(value[order[first]]+value[order[second]])/2
    return result


def color_control(refs, data, xy, time, sigma, keep, keys, offsets, points, config, settings):
    """Train one relative g−i term on galaxies, then test galaxies and QSOs."""
    from scipy.optimize import least_squares

    from .nsc_comparison import comparison, qso_matches
    from .pilot_production import valid_epochs
    from .precision_v03 import fit_with_calibration

    r=data.band.eq("r").to_numpy()
    nsc=data.survey.eq("NSC").to_numpy()
    color=data.color.to_numpy()
    support=np.isin(keys,list(offsets))
    chosen=refs.loc[(r & support)[refs.index]]
    control={False:[],True:[]}
    for (hold,_),group in chosen.groupby(["is_holdout","reference"],sort=False):
        i=group.index.to_numpy()
        early,late=i[~nsc[i]],i[nsc[i]]
        usable=color[early]
        usable=usable[np.isfinite(usable)&(usable>=-.3)&(usable<=3.5)]
        if not (len(usable) and len(late)):
            continue
        delta=xy[i]-np.stack([offsets[k][0] for k in keys[i]])
        control[hold].append((np.median(usable),np.median(delta[~nsc[i]],axis=0)-np.median(delta[nsc[i]],axis=0)))
    if min(map(len,control.values()))<30:
        return dict(status="insufficient_galaxy_color_support",train=len(control[False]),held=len(control[True]))
    colors=np.array([v[0] for v in control[False]])
    med=np.median(colors)
    lower,upper=np.quantile(colors,[.02,.98])
    target=np.stack([v[1] for v in control[False]])
    slope=np.array([least_squares(lambda p:target[:,i]-p[0]-p[1]*(colors-med),
                                 [0.,0.], loss="soft_l1",f_scale=75).x[1] for i in range(2)])
    matches=[]
    for col,value in control[True]:
        if lower<=col<=upper:
            matches.append(dict(C1=(value,np.ones(2)),C3=(value-(col-med)*slope,np.ones(2))))
    galaxy=comparison(matches,dict(settings,tail_masyr=150),(("C3","C1"),))
    _,tracks=qso_matches(data,xy,keep,points)
    qa=[]
    for i in tracks.values():
        i=i[support[i]]
        if not valid_epochs(i,nsc,time,settings["production"]):
            continue
        early=~nsc[i]
        cols=color[i[early]]
        cols=cols[np.isfinite(cols)&(cols>=-.3)&(cols<=3.5)]
        if not len(cols):
            continue
        col=np.median(cols)
        if not lower<=col<=upper:
            continue
        row=xy[i]
        models=[]
        for scale in (0.,1.):
            fixed=row-early[:,None]*(scale*(col-med)*slope)
            models.append(fit_with_calibration(fixed,time[i],sigma[i],config["reference_epoch"],
                                                [offsets[k] for k in keys[i]]))
        qa.append({name:(model.coeff[1],err) for name,(model,err) in zip(("C1","C3"),models)})
    return dict(status="evaluated",train=len(colors),training_color_range=[lower,upper],
                slope_mas_per_mag=slope, galaxy=galaxy,
                qso=comparison(qa,settings,(("C3","C1"),)))
