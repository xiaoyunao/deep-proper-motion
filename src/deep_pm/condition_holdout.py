"""Compare error models and bands on identical, truly held-year positions."""
from collections import Counter, defaultdict

from astropy.time import Time
import numpy as np
from scipy.spatial import cKDTree

from .astrometry import unproject
from .astrometry_audit import augment_tracks
from .dark_precision import query_unique
from .pilot_production import make_tracks, source_funnel
from .precision_v03 import fit_with_calibration
from .standard_motion import calibrated_covariance
from .validation import ratio


def held_prediction_variance(model, time, sigma, offsets, held_offsets, held_sigma, held_time):
    """Includes reference correlation BETWEEN training groups and held exposure."""
    _,independent,_,influence = calibrated_covariance(model,time,sigma,offsets)
    draw = np.stack([o[1] for o in offsets],axis=1)
    state_draw = np.einsum('pna,dna->dp',influence,draw)
    dt = held_time-model.epoch
    design = np.array([[1.,dt,0.,0.],[0.,0.,1.,dt]])
    common_difference = held_offsets[1]-state_draw@design.T
    return held_sigma**2+np.diag(design@independent@design.T)+np.var(common_difference,axis=0,ddof=1)


def fixed_year_prediction(data,xy,time,sigma,revised,keys,nsc,keep,expanded,offsets,settings,config,spec,center):
    edges = Time([f'{y}-01-01' for y in range(2010,2021)],scale='utc').tt.jyear
    year_index = np.searchsorted(edges,time,side='right')-1
    calibrated = np.isin(keys,list(offsets))
    eligible = nsc & keep & calibrated
    years = np.unique(year_index[eligible])
    if len(years) < 3:
        return dict(status='insufficient_calibrated_nsc_years',n_years=len(years)),[]
    exposures = data.exposure_id.to_numpy()
    held_year = max(years[1:-1],key=lambda y:len(set(exposures[eligible & (year_index==y)])))
    held = nsc & (year_index==held_year)
    train = keep & ~held
    prod = settings['production']
    tracks = make_tracks(xy,exposures,train,prod['match_radius_mas'])
    seeds = make_tracks(xy,exposures,train & nsc,prod['match_radius_mas'])
    extra = augment_tracks(tracks,data,xy,expanded & ~held,train,prod)
    def core(pos):
        return inside_position(pos,center,spec)
    fits = {}
    _,baseline = source_funnel(data,xy,time,sigma,train,tracks,offsets,config,settings,9.,initial=seeds,
                               observation_groups=keys,core_members=core,fits=fits)
    models = {}
    for slot,value in baseline.items():
        if value[3] not in ('20_to_21','21_to_22','22_to_23'):
            continue
        r = value[4]
        multi = extra[slot][calibrated[extra[slot]]]
        if np.any(held[multi]) or not set(r).issubset(multi):
            raise ValueError('Held-year leakage or changed r identity')
        models[slot] = {}
        for name,use,error in [('W0_r',r,sigma),('W2_r',r,revised),('W0_multi',multi,sigma),('W2_multi',multi,revised)]:
            model,_ = fit_with_calibration(xy[use],time[use],error[use],config['reference_epoch'],
                                           [offsets[keys[j]] for j in use])
            models[slot][name] = model,use,error
    slots = sorted(models)
    if not slots:
        return dict(status='no_eligible_training_sources',year=int(held_year+2010)),[]
    coordinate = xy-np.stack([offsets[k][0] if k in offsets else np.zeros(2) for k in keys])
    matched = []
    counters = Counter()
    for _, frame in data.loc[eligible & held].groupby('exposure_id',sort=False):
        ix = frame.index.to_numpy()
        midpoint = np.median(time[ix])
        predictions = np.stack([models[j]['W0_r'][0].predict(midpoint) for j in slots])
        good,closest = query_unique(cKDTree(coordinate[ix]),predictions,1000,prod['competitor_margin_mas'])
        counters['held_exposures'] += 1
        counters['training_source_exposure_opportunities'] += len(slots)
        for local in np.flatnonzero(good):
            slot = slots[local]
            k = ix[closest[local]]
            values = {}
            for name,(model,use,error) in models[slot].items():
                residual = coordinate[k]-model.predict(time[k])
                variance = held_prediction_variance(model,time[use],error[use],
                     [offsets[keys[j]] for j in use],offsets[keys[k]],error[k],time[k])
                values[name] = dict(residual_mas=residual,error_mas=np.sqrt(variance))
            matched.append(dict(track_id=slot,r_bin=baseline[slot][3],held_detection_id=data.detection_id.iloc[k],
                                values=values))
    by_r = {}
    for tag in ('20_to_21','21_to_22','22_to_23'):
        selected = [m for m in matched if m['r_bin']==tag]
        ntrain = sum(baseline[j][3]==tag for j in slots)
        out = dict(train_sources=ntrain,matched_sources=ratio(len(set(r['track_id'] for r in selected)),ntrain),
                   matched_positions=len(selected),methods={})
        for name in ('W0_r','W2_r','W0_multi','W2_multi'):
            if not selected:
                continue
            residual = np.array([r['values'][name]['residual_mas'] for r in selected])
            error = np.array([r['values'][name]['error_mas'] for r in selected])
            out['methods'][name] = dict(radial_p68_mas=np.quantile(np.linalg.norm(residual,axis=1),.68),
                median_mas=np.median(residual,axis=0),
                within_1sigma=[ratio(int(n),len(selected)) for n in (abs(residual)<=error).sum(axis=0)])
        by_r[tag] = out
    return dict(status='completed',year=int(held_year+2010),counts=counters,by_r=by_r,
                selection_scope='Fixed W0-r training sources and one unique held r detection per source/exposure; 1 arcsec W0 prediction gate.'),matched


def inside_position(position,center,spec):
    from .exposure_catalogues import inside
    return inside(*unproject(position,center)[0],spec)
