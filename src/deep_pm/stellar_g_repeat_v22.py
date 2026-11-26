"""Observed same-star g repeats in two metadata-selected cores, without Gaia."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
from itertools import combinations
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets
from .coordinate_metric import AngularTree,native_covariance,projected_covariance,sky_basis
from .dcr_geometry import sdss_colors,combine_color,zenith_vectors,tangent_geometry
from .io import digest,save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .stellar_g_mechanism_v22 import deep_neighbours,finite_json


def run(root,lsroot,sas,output):
    output.mkdir(parents=True,exist_ok=True)
    path=root/'v2.2/stellar-g-metadata-final-single/private_additional_core_plan.json';plan=read(path)
    if len(plan['selected'])>2:raise ValueError('At most two metadata-selected cores')
    all_pairs=[];reports=[];receipts={}
    for entry in plan['selected']:
        core=entry['core'];center=entry['center'];source=Path(entry['source']);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
        data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);old=read_offsets(entry['old_graph'])
        nsc=data.survey.eq('NSC').to_numpy();g=nsc&data.band.eq('g').to_numpy()&data.instrument.eq('c4d').to_numpy()
        valid=pd.Series(keys).isin(old).to_numpy();snr=1.0857362047581296/data.magnitude_error.to_numpy(float)
        initial=g&valid&np.isfinite(snr)&(snr>=20)
        oldrows=read(Path(entry['fit'])/'private_standard_motion.json.gz');lookup=pd.Index(data.detection_id);universe=[]
        # Physical source identities already frozen, not NSC object labels.
        for slot,row in enumerate(oldrows):
            ix=lookup.get_indexer(row['detection_ids'])
            if (ix<0).any():continue
            selected=ix[initial[ix]]
            if len(selected)>=2:universe.append((slot,row,ix,selected))
        sdss=sorted({str(data.detection_id.iat[i]) for _,_,ix,_ in universe for i in ix[~nsc[ix]]})
        phot,color_receipts=sdss_colors(sdss,sas);receipts.update(color_receipts);phot=phot.set_index('detection_id',drop=False)
        sources=[]
        veto=pd.read_csv(source/'spectra_union.csv')
        veto_tree=AngularTree(project(veto.ra,veto.dec,center),center) if len(veto) else None
        for slot,row,ix,selected in universe:
            if veto_tree is not None and veto_tree.query(project(*row['position_deg'],center)[0])[0]<1000:continue
            ids=data.detection_id.iloc[ix[~nsc[ix]]].astype(str).tolist();colors=phot.reindex(ids)
            color,error=combine_color(colors)
            if not np.isfinite(color+error) or error>.2:continue
            sources.append(dict(source_slot=slot,members=row['detection_ids'],position_deg=row['position_deg'],g_indices=selected,color=color,color_error=error))
        neighbors,deep_receipts=deep_neighbours(np.array([r['position_deg'] for r in sources]).reshape(-1,2),lsroot) if sources else ([],{})
        receipts.update(deep_receipts)
        stamps=Time(data.observation_time.to_numpy(),format='mjd',scale='tt');mjd=stamps.utc.mjd
        nights=Time(np.floor(mjd-.75),format='mjd',scale='utc').isot
        eligible=np.flatnonzero(initial);unique,inverse=np.unique(mjd[eligible],return_inverse=True)
        zenith=zenith_vectors(unique)[inverse]
        directions=sky_basis(data.ra_deg.to_numpy()[eligible],data.dec_deg.to_numpy()[eligible])[:,0,:]
        supported=np.sum(directions*zenith,axis=1)>.2
        initial[eligible[~supported]]=False
        eligible=eligible[supported];zenith=zenith[supported]
        h_native,air=tangent_geometry(data.ra_deg.to_numpy()[eligible],data.dec_deg.to_numpy()[eligible],zenith,(center[0],center[1]))
        # tangent_geometry already expresses the zenith displacement in the
        # common chart defined by its center; do not apply another Jacobian.
        geometry=np.zeros((len(data),2));geometry[eligible]=h_native
        cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma))
        selected_sources=0;pair_start=len(all_pairs);fail=Counter()
        for j,(row,neighbor) in enumerate(zip(sources,neighbors)):
            if j%400==0:print('G_REPEAT_CORE_PROGRESS',core,j,len(sources),flush=True)
            if neighbor.get('status')!='unique_deep_self' or neighbor.get('neighbour_distance_mas',0)<3000:fail['deep_neighbor_control']+=1;continue
            selected_sources+=1;physical_key=sha256('\0'.join(sorted(row['members'])).encode()).hexdigest();by_night=defaultdict(list)
            for i in row['g_indices']:
                if initial[i]:by_night[nights[i][:10]].append(int(i))
            for night,indices in by_night.items():
                pairs=[(a,b) if time[a]<time[b] else (b,a) for a,b in combinations(indices,2) if data.exposure_id.iat[a]!=data.exposure_id.iat[b]]
                if not pairs:continue
                a,b=max(pairs,key=lambda p:np.linalg.norm(geometry[p[1]]-geometry[p[0]]))
                dh=geometry[b]-geometry[a];za=old[keys[a]][0];zb=old[keys[b]][0]
                group='|'.join((str(keys[a]),str(data.detector_id.iat[a]),str(keys[b]),str(data.detector_id.iat[b])))
                star_held=int(sha256(('v22-g-repeat-star:'+physical_key).encode()).hexdigest()[:8],16)%3==0
                night_held=int(sha256(('v22-g-repeat-night:'+night).encode()).hexdigest()[:8],16)%4==0
                all_pairs.append(dict(source_key=core+':'+physical_key,core=core,night=night,group=group,held_star=star_held,held_night=night_held,
                    delta_observed_common_mas=xy[b]-xy[a]-(zb-za),delta_geometry=dh,color=row['color'],color_error=row['color_error'],
                    independent_common_covariance=cp[a]+cp[b],snr=[float(snr[a]),float(snr[b])],g_mag=[float(data.magnitude.iat[a]),float(data.magnitude.iat[b])],
                    seeing=[float(data.fwhm.iat[a]),float(data.fwhm.iat[b])],
                    time_separation_hours=float(abs(time[b]-time[a])*365.25*24),
                    motion_sensitivity_mas_at10000=float(abs(time[b]-time[a])*10000),
                    group_reference_difference_draws=np.asarray(old[keys[b]][1])-old[keys[a]][1],
                    deep_neighbor=neighbor,source_role='held' if star_held else 'train'))
        values=all_pairs[pair_start:]
        reports.append(dict(core=core,frozen_source_candidates=len(universe),color_supported_nonQSO_sources=len(sources),deep_control_sources=selected_sources,
            pairs=len(values),nights=len({r['night'] for r in values}),
            training_pairs=sum(not r['held_star'] and not r['held_night'] for r in values),
            independent_identity_and_night_pairs=sum(r['held_star'] and r['held_night'] for r in values),
            color_p16_p50_p84=np.quantile([r['color'] for r in values],[.16,.5,.84]) if values else None,exits=dict(fail)))
    save_json(output/'private_observed_repeat_pairs.json',finite_json(dict(pairs=all_pairs,metadata_selection_sha256=digest(path),file_receipts=receipts)))
    result=dict(cores=reports,metadata_selection_sha256=digest(path),pairs=len(all_pairs),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='At most two preselected existing cores. Frozen physical blind source members; known optical spectra vetoed before training. Observed g positions, actual gSNR>=20, color error<=0.2, deep neighbor>=3arcsec. One maximum-geometry pair per physical star/UTC18 night avoids duplicate exposure pairs. No Gaia PM/parallax/epoch position or residual used. Exposure-CCD-pair intercept nuisance must remain in coefficient fit; absolute zero and stellar purity not certified.')
    check_aggregate(result);save_json(output/'v22_extra_g_repeat_support_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','lsroot','sas','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(**vars(a))
