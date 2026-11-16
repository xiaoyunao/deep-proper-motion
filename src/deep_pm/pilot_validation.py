"""Independent labels are joined only after blind tracks are frozen."""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import access
from .astrometry import project
from .dark_precision import bin_name
from .io import save_json
from .nsc_comparison import comparison, fit_methods, qso_matches
from .paired_control import catalogue_motion, common_catalogue_ids, published_rows
from .precision_v03 import calibration_groups
from .validation import gaia_labels, groups_by_label, ratio


def track_audit(tracks, selected, truth, labels, nsc, recovery):
    recovered, final = {}, {}
    considered = mixed = unlabeled = known_members = wrong = 0
    attempted = defaultdict(list)
    for slot,index in enumerate(tracks):
        if len(index)<2 or nsc[index].all() or not nsc[index].any():
            continue
        known = labels[index]
        observed = known[known>=0]
        if slot in selected:
            unlabeled += int((known<0).sum())
        if not len(observed):
            continue
        labels_seen, counts = np.unique(observed,return_counts=True)
        if slot in selected:
            considered += 1
            mixed += len(labels_seen)>1
            known_members += counts.sum()
            wrong += counts.sum()-counts.max()
        if len(labels_seen)!=1:
            continue
        label = labels_seen[0]
        if label not in truth:
            continue
        member = index[known==label]
        if nsc[member].all() or not nsc[member].any() or len(member)/len(truth[label])<recovery:
            continue
        recovered[label] = True
        if slot not in selected:
            continue
        used = selected[slot][4]
        # Reject any audit association with a known competing validation member
        # in the final M3 observations or a missing validated survey endpoint.
        surviving = labels[used]
        if np.any((surviving>=0) & (surviving!=label)):
            continue
        matched = used[surviving==label]
        if len(matched) and nsc[matched].any() and not nsc[matched].all():
            attempted[label].append(slot)
    final.update((k,v[0]) for k,v in attempted.items() if len(v)==1)
    return final, dict(cross_survey_recovered=ratio(len(recovered),len(truth)),
                       final_recovered=ratio(len(final),len(truth)),
                       final_mixed_known_tracks=ratio(mixed,considered),
                       final_wrong_known_members=ratio(wrong,known_members),
                       final_unlabelled_members=unlabeled,
                       repeated_final_truth_matches=sum(len(v)>1 for v in attempted.values()))


def compare_truth(work, center, data, xy, time, sigma, truth, labels, ref, aim,
                  r_mag, offsets, blind, selected, config, settings):
    raw = pd.read_csv(work/"external_labels.csv",dtype=str).set_index("detection_id").original_object_id
    linked, exits = common_catalogue_ids(truth,labels,data,raw)
    published = published_rows(access.Cache(work/"cache"),linked.values())
    nsc = data.survey.eq("NSC").to_numpy()
    group = calibration_groups(data)
    supported = np.isin(group,list(offsets))
    span = config["min_seed_years"]
    epoch = config["reference_epoch"]
    anchored, common, individual = defaultdict(list),defaultdict(list),[]
    for label,index in truth.items():
        obj = linked.get(label)
        if obj not in published.index:
            exits["no_unique_published"] += 1
            continue
        row = published.loc[obj]
        error = row[["pmraerr","pmdecerr"]].to_numpy(dtype=float)
        pm = catalogue_motion(row,center)
        if (not np.isfinite(error).all() or (error<=0).any() or not np.isfinite(pm).all()
                or abs(pm).max()>1e5 or not row.deltamjd>0):
            exits["invalid_published_motion"] += 1
            continue
        use = index[supported[index]]
        late = use[nsc[use]]
        if len(late)<2 or len(late)==len(use) or np.ptp(time[late])<span:
            exits["insufficient_frozen_m3"] += 1
            continue
        estimates = fit_methods(use,xy,time,sigma,nsc,epoch,[offsets[g] for g in group[use]])
        estimates["published_nsc"] = (pm,error)
        result = {k:(v[0]-ref[label],np.hypot(v[1],aim[label])) for k,v in estimates.items()}
        r = r_mag[label]
        tags = ["all",bin_name(r,settings["reference_r_edges"])]
        if 21<=r<23:
            tags.append("faint_21_to_23")
        for k in tags:
            anchored[k].append(result)
        individual.append((r,np.ptp(time[late]),result))
        if label not in blind:
            continue
        _, motion, uncertainty, *_ = selected[blind[label]]
        paired = dict(result, M3_blind=(motion-ref[label],np.hypot(uncertainty,aim[label])))
        for k in tags:
            common[k].append(paired)
    return dict(anchored={k:comparison(v,settings) for k,v in anchored.items()},
                common_blind={k:comparison(v,settings) for k,v in common.items()},
                exclusions=dict(exits)),individual


def validate_catalogue(work,center,radius,data,xy,t,sigma,keep,tracks,selected,offsets,
                       qso,config,settings):
    prod = settings["production"]
    nsc = data.survey.eq("NSC").to_numpy()
    qso = qso.loc[qso.in_core].reset_index(drop=True)
    if len(qso):
        labels, all_qso = qso_matches(data,xy,keep,project(qso.ra,qso.dec,center))
    else:
        labels, all_qso = np.full(len(data),-1), {}
    truth = {k:v for k,v in all_qso.items() if nsc[v].any() and not nsc[v].all()}
    blind, audit = track_audit(tracks,selected,truth,labels,nsc,prod["validation_min_recovery"])
    zero = np.zeros((len(qso),2))
    qso_stats, individual = compare_truth(work,center,data,xy,t,sigma,truth,labels,zero,zero,
            qso.reference_r.to_numpy(),offsets,blind,selected,config,settings)
    qso_stats.update(association=audit, spectra_in_core=len(qso))
    del labels, all_qso, truth
    save_json(work/"private_qso_pairs.json",individual)

    def validation_stars(reference, kind, max_error):
        idx = np.flatnonzero(keep)
        controls = dict(config,gaia_max_pm_error=max_error)
        subset_labels, catalogue = gaia_labels(data.iloc[idx].reset_index(drop=True),
                                               xy[idx],t[idx],sigma[idx],reference,center,controls)
        label = np.full(len(data),-1,dtype=int)
        label[idx] = subset_labels
        all_truth = groups_by_label(label,data,t,catalogue,center,radius,controls)
        magnitudes = {j:float(data.magnitude.iloc[v[nsc[v]]].median()) for j,v in all_truth.items()}
        if kind=="gaia_faint":
            all_truth = {j:v for j,v in all_truth.items() if magnitudes[j]>=20}
        blind, audit = track_audit(tracks,selected,all_truth,label,nsc,prod["validation_min_recovery"])
        measures, _ = compare_truth(work,center,data,xy,t,sigma,all_truth,label,
                catalogue[["pm_tangent_x","pm_tangent_y"]].to_numpy(),
                catalogue[["pmra_error","pmdec_error"]].to_numpy(),magnitudes,
                offsets,blind,selected,controls,settings)
        return dict(measures,association=audit)

    gaia = pd.read_csv(work/"validation"/"gaia.csv")
    bright = validation_stars(gaia,"gaia_precision",config["gaia_max_pm_error"])
    faint = validation_stars(gaia,"gaia_faint",2.)
    external_file = work/"validation"/"external_pm.csv"
    external = dict(status="no_verified_external_catalogue")
    if external_file.is_file():
        reference = pd.read_csv(external_file).assign(ruwe=1.,parallax=0.)
        external = validation_stars(reference,"external",5.)
    return dict(qso=qso_stats,gaia_precision=bright,gaia_faint=faint,external_dark_stars=external)
