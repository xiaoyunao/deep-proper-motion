"""Trace strict QSO exits and verify separate blind catalogues on common objects."""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import access
from .astrometry import project
from .dark_precision import bin_name
from .nsc_comparison import comparison, qso_matches
from .paired_control import catalogue_motion, common_catalogue_ids, published_rows
from .pilot_validation import track_audit
from .pilot_production import ANCHOR_EXITS
from .validation import gaia_labels, groups_by_label, ratio

CONTRASTS = (("W0", "published_nsc"),("W1", "published_nsc"),("multi", "published_nsc"),
             ("W1", "W0"),("multi", "W1"),("multi", "W0"))


def official_table(work, linked):
    table = work/"official_nsc.csv"
    if table.is_file():
        return pd.read_csv(table,dtype={"id":str}).set_index("id",verify_integrity=True)
    return published_rows(access.Cache(work/"validation_cache"),linked.values())


def valid_published(row, center):
    values = row[["pmra","pmdec","pmraerr","pmdecerr","deltamjd"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or abs(values[:2]).max()>1e5 or (values[2:]<=0).any():
        return None
    return catalogue_motion(row,center), values[2:4]


def tag_magnitude(mag,settings):
    names=["all",bin_name(mag,settings["reference_r_edges"])]
    if 21<=mag<23:
        names.append("21_to_23")
    return names


def first_exit(label, slots, truth, labels, trace, tracks, nsc, support, recover):
    accepted=[]
    mixed=False
    early_late=False
    for slot in slots:
        ix=tracks[slot]
        aims=ix[labels[ix]==label]
        if not len(aims):
            continue
        if nsc[aims].any() and not nsc[aims].all():
            early_late=True
        else:
            continue
        if np.any((labels[ix]>=0)&(labels[ix]!=label)):
            mixed=True
            continue
        if np.count_nonzero(np.isin(aims,truth[label])) / len(truth[label]) < recover:
            continue
        accepted.append(trace.get(slot,"source_selection_or_competition"))
    if accepted:
        order=("source_selection_or_competition",)+ANCHOR_EXITS
        furthest=max(accepted,key=order.index)
        return "validation_ambiguity" if furthest=="final" else furthest
    if mixed:
        return "mixed_known_objects"
    return "insufficient_member_recovery" if early_late else "no_cross_survey_track"


def audit_qso(work,center,data,xy,t,nsc,keep,expanded,support,products,qso,settings):
    qso = qso.loc[qso.in_core].reset_index(drop=True)
    positions=project(qso.ra,qso.dec,center)
    labels, matches=qso_matches(data,xy,keep,positions)
    broad_labels,_=qso_matches(data,xy,expanded,positions)
    truth={key:idx for key,idx in matches.items() if nsc[idx].any() and not nsc[idx].all()}
    prod=settings["production"]
    method_labels={"W0":labels,"W1":labels,"multi":broad_labels}
    originals=pd.read_csv(work/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    linked, _=common_catalogue_ids(truth,labels,data,originals)
    published=official_table(work,linked)
    finals, recoveries, slots_by_label = {}, {}, {}
    for name,(tracks,chosen,trace) in products.items():
        method=method_labels[name]
        finals[name],recoveries[name]=track_audit(tracks,chosen,truth,method,nsc,prod["validation_min_recovery"])
        slots=defaultdict(list)
        for j,index in enumerate(tracks):
            for label in np.unique(method[index]):
                if label>=0:
                    slots[label].append(j)
        slots_by_label[name]=slots
    cover={}
    search_radius=500.
    for name, use in (("NSC",nsc),("SDSS",~nsc)):
        tree=cKDTree(xy[use & data.band.eq("r").to_numpy()])
        cover[name]=tree.query(positions)[0]<=search_radius
    flows=defaultdict(Counter)
    strict_exits=defaultdict(Counter)
    measures=defaultdict(lambda:defaultdict(list))
    pairs=[]
    for label,mag in enumerate(qso.reference_r):
        tags=tag_magnitude(mag,settings)
        if label not in truth:
            if not cover["NSC"][label] or not cover["SDSS"][label]:
                reason="no_nsc_or_sdss_position"
            else:
                reason="snr_or_ambiguous_detection"
            for tag in tags:
                flows[tag][reason]+=1
            continue
        index=truth[label]
        late=index[support[index]&nsc[index]]
        if len(late)<2 or np.ptp(t[late])<prod["min_nsc_span_years"] or not np.any(support[index]&~nsc[index]):
            for tag in tags:
                flows[tag]["no_anchored_m3_span"]+=1
            continue
        key=linked.get(label)
        official=valid_published(published.loc[key],center) if key in published.index else None
        if official is None:
            for tag in tags:
                flows[tag]["no_usable_unique_nsc_motion"]+=1
            continue
        result={"published_nsc":official}
        for tag in tags:
            flows[tag]["strict_common"]+=1
        for name,(tracks,chosen,trace) in products.items():
            slot=finals[name].get(label)
            if slot is not None:
                _,pm,error,*_=chosen[slot]
                result[name]=(pm,error)
                reason="final"
            else:
                reason=first_exit(label,slots_by_label[name][label],truth,method_labels[name],
                                  trace,tracks,nsc,support,prod["validation_min_recovery"])
            for tag in tags:
                strict_exits[f"{name}_{tag}"][reason]+=1
        pairs.append((float(mag),result))
        for tag in tags:
            for name in ("W0","W1","multi"):
                if name in result:
                    measures[tag][name].append({"published_nsc":official,name:result[name]})
            if len(result)==4:
                measures[tag]["same_three"].append(result)
            if "multi" in result and "W1" not in result:
                measures[tag]["added_vs_w1"].append({"published_nsc":official,"multi":result["multi"]})
    summary={tag:{k:comparison(v,settings,CONTRASTS) for k,v in d.items()} for tag,d in measures.items()}
    converted={}
    for tag,c in strict_exits.items():
        n=sum(c.values())
        converted[tag]={k:ratio(v,n) for k,v in c.items()}
    return dict(spectra_in_core=len(qso),photometric_funnel=flows,strict_exits=converted,
                association=recoveries,precision=summary),pairs


def audit_gaia(work,center,spec,data,xy,t,sigma,keep,expanded,nsc,products,settings,config):
    gaia=pd.read_csv(work/"validation"/"gaia.csv")
    idx=np.flatnonzero(expanded)
    subset,cat=gaia_labels(data.iloc[idx].reset_index(drop=True),xy[idx],t[idx],sigma[idx],
                           gaia,center,dict(config,gaia_max_pm_error=2.))
    all_labels=np.full(len(data),-1)
    all_labels[idx]=subset
    labels=np.where(keep,all_labels,-1)
    truth=groups_by_label(labels,data,t,cat,center,spec["radius"],config)
    motion=cat[["pm_tangent_x","pm_tangent_y"]].to_numpy()
    error=cat[["pmra_error","pmdec_error"]].to_numpy()
    magnitude=data.magnitude.to_numpy()
    mag={key:float(np.median(magnitude[ix[nsc[ix]]])) for key,ix in truth.items()}
    labels_by_method={name:all_labels if name=="multi" else labels for name in products}
    prod=settings["production"]
    results,recovery={},{}
    for name,(tracks,selected,_) in products.items():
        results[name],recovery[name]=track_audit(tracks,selected,truth,labels_by_method[name],nsc,prod["validation_min_recovery"])
    sample=defaultdict(lambda:defaultdict(list))
    for key in truth:
        current={}
        for name,found in results.items():
            if key not in found:
                continue
            _,pm,unc,*_=products[name][1][found[key]]
            current[name]=(pm-motion[key],np.hypot(unc,error[key]))
        groups={name:{name:value} for name,value in current.items()}
        if len(current)==len(products):
            groups["same_three"]=current
        if "multi" in current and "W1" not in current:
            groups["added_vs_w1"]={"multi":current["multi"]}
        for tag in tag_magnitude(mag[key],settings):
            for name,value in groups.items():
                sample[tag][name].append(value)
    # Gaia comparisons retain the common published-NSC baseline where it can be
    # independently identified from all true r measurements.
    originals=pd.read_csv(work/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    linked,_=common_catalogue_ids(truth,labels,data,originals)
    release=official_table(work,linked) if (work/"official_nsc.csv").is_file() else pd.DataFrame()
    if len(release):
        for key in sorted(set.intersection(*(set(found) for found in results.values()))):
            k=linked.get(key)
            official=valid_published(release.loc[k],center) if k in release.index else None
            if official is None:
                continue
            values={"published_nsc":(official[0]-motion[key],np.hypot(official[1],error[key]))}
            for method,found in results.items():
                _,pm,unc,*_=products[method][1][found[key]]
                values[method]=(pm-motion[key],np.hypot(unc,error[key]))
            for tag in tag_magnitude(mag[key],settings):
                sample[tag]["same_three_and_nsc"].append(values)
    precision={tag:{k:comparison(v,settings,CONTRASTS) for k,v in values.items()} for tag,values in sample.items()}
    return dict(association=recovery,precision=precision,
                 truth_with_both=len(truth))
