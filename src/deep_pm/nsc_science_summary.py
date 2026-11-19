"""Pool individual private research fits into deidentified continuous statistics."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_ownership import close_pairs
from .continuous_reference import verify_directory,read_offsets
from .allband_qso import physical_candidates
from .continuous_summary import area,estimate
from .exposure_catalogues import inside
from .independent_summary import standard_block
from .io import digest,save_json
from .nsc_comparison import comparison
from .nsc_information import summarize_information
from .public_summary import check_aggregate
from .validation_blocks import metrics


SETTINGS=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=800,bootstrap_absolute_zero=True,seed=9137)
METHODS=('r_legacy_quality','r_physical_quality','all_nsc')


def legacy_tangent_replay(old,new):
    """Historical scalar-output QA, never a detection identity or precision label."""
    tables=[]
    for rows in (old,new):
        grouped=defaultdict(list)
        for row in rows:
            if np.isfinite(row['magnitude']):grouped[(float(row['magnitude']).hex(),tuple(sorted(row['exposures'])))].append(row)
        tables.append(grouped)
    before,after=tables;delta=[];ambiguous=0;unmatched=0
    for key,rows in before.items():
        candidates=after.get(key,[])
        if len(rows)!=1 or len(candidates)>1:ambiguous+=len(rows);continue
        if not candidates:unmatched+=1;continue
        delta.append(np.asarray(candidates[0]['motion']['C0'])-rows[0]['motion']['C0'])
    return dict(old_archived_controls=len(old),unique_timing_photometry_pairs=len(delta),ambiguous_archived_controls=ambiguous,unmatched_archived_controls=unmatched,
        maximum_tangent_motion_change_masyr=float(np.max(abs(np.asarray(delta)))) if delta else None)


def known_cohorts(rows, instrument_diagnostics=False):
    out={};pairs=(('all_nsc','r_physical_quality'),('r_physical_quality','r_legacy_quality'),('all_nsc','Xiao'),('all_nsc','NSC'))
    for low in (21,22):
        predicates=[('all',lambda r:True),('one_sdss_run',lambda r:r['sdss_runs']==1),('multiple_sdss_runs',lambda r:r['sdss_runs']>1),('nsc_span_lt2',lambda r:r['nsc_span']<2),('nsc_span_ge2',lambda r:r['nsc_span']>=2),
            ('pointlike_qso',lambda r:r.get('shape_class')=='pointlike'),('extended_or_mixed_qso',lambda r:r.get('shape_class')=='extended_or_mixed'),('unknown_morphology',lambda r:r.get('shape_class','unknown')=='unknown'),
            ('one_run_pointlike',lambda r:r['sdss_runs']==1 and r.get('shape_class')=='pointlike'),('one_run_extended_or_mixed',lambda r:r['sdss_runs']==1 and r.get('shape_class')=='extended_or_mixed')]
        if instrument_diagnostics:
            predicates.append(('nsc_nights_le5',lambda r:len(r['nsc_epoch_days'])<=5))
            predicates.append(('nsc_nights_gt5',lambda r:len(r['nsc_epoch_days'])>5))
            for label in sorted({r['instruments'] for r in rows}):
                predicates.append(('instrument_'+label,lambda r,label=label:r['instruments']==label))
            for label in sorted({r['bands'] for r in rows}):
                predicates.append(('filter_'+label,lambda r,label=label:r['bands']==label))
        for tag,select in predicates:
            part=[r for r in rows if low<=r['magnitude']<low+1 and select(r)];common=[r for r in part if {'NSC','Xiao'}<=r['values'].keys()]
            blocks=[dict(exposures=r['exposures'],nsc_epoch_days=r['nsc_epoch_days'],motion={'C0':r['values']['Xiao'][0],'C1':r['values']['all_nsc'][0]}) for r in common]
            sensitivity=[dict(exposures=r['exposures'],nsc_epoch_days=r['nsc_epoch_days'],motion={m:r['values'][m][0] for m in ('NSC','Xiao','all_nsc')}) for r in common]
            out[f'r{low}__{tag}']=dict(same_objects=comparison([{m:r['values'][m] for m in METHODS} for r in part],SETTINGS,pairs),
                strict_published_common=comparison([r['values'] for r in common],SETTINGS,pairs),
                observational_blocks=dict(method_order=['Xiao','all_nsc'],physical=metrics(blocks),conservative_night=metrics(blocks,True)),
                shared_group_sensitivity=estimate(sensitivity,('NSC','Xiao','all_nsc'),9137,candidate='all_nsc'))
    return out


def public_core_summary(summary,spec):
    # These are declared rectangular footprint bounds, never per-source sky rows.
    result=dict(core_bounds_deg=dict(ra_bounds_deg=spec['ra'],dec_bounds_deg=spec['dec']),**summary)
    check_aggregate(result)
    return result


def run(state,geometry,output):
    config=tomllib.loads(geometry.read_text());plan=json.loads((state/'private_frozen_plan.json').read_text());fits=json.loads((state/'fits.json').read_text());original=Path(plan['original_state']) if 'original_state' in plan else None;graph=Path(plan['reference_graph']) if 'reference_graph' in plan else original/'shared/private_c0.npz';graph_hash=digest(graph)
    old_fits=json.loads((original/'fits.json').read_text()) if original else {}
    if graph_hash!=plan['reference_sha256']:raise ValueError('Frozen common graph changed')
    supported_groups=frozenset(read_offsets(graph))
    known=[];blind=[];by_class=defaultdict(list);spatial={};frames=[];receipts={};members={};crossclaims=0;outside=0;replay=[];acquisitions=[];unmatched_old=0;core_input_counts=Counter();qualified_core_bands=Counter();quality_flags=Counter();all_classes=Counter();historical={};legacy_missing_position=0
    for core,folder in fits.items():
        print('Summarizing',core,flush=True)
        folder=Path(folder);receipt=verify_directory(folder,('nsc_science_catalogue_summary.json','private_standard_motion.json.gz','private_known_qso_blocks.json','private_blind_qso.json'))
        ref=(folder/'reference_receipt.json').resolve().parent/'private_c0.npz'
        if digest(ref)!=graph_hash:raise ValueError('Mixed reference frames in the continuous catalogue')
        receipts[core]=digest(receipt);summary=json.loads((folder/'nsc_science_catalogue_summary.json').read_text());spatial[core]=public_core_summary(summary,config['fields'][core])
        source=(folder/'source_receipt.json').resolve().parent;acq=json.loads((source/'acquisition_summary.json').read_text());acquisitions.append(acq)
        columns=['survey','band','exposure_id','ra_deg','dec_deg','observation_time','sigma_east_mas','sigma_north_mas','magnitude','magnitude_error','quality_flags']
        for chunk in pd.read_csv(source/'detections.csv',usecols=columns,chunksize=250000):
            on_core=inside(chunk.ra_deg,chunk.dec_deg,config['fields'][core]);core_input_counts.update(chunk.loc[on_core,'survey'])
            qualified=on_core&chunk.survey.eq('NSC')&physical_candidates(chunk)&chunk.exposure_id.isin(supported_groups)
            qualified_core_bands.update(chunk.loc[qualified,'band'])
        with gzip.open(folder/'private_standard_motion.json.gz','rt') as stream:records=json.load(stream)
        if len(records)!=summary['owned_all_nsc']:raise ValueError('Catalogue count differs from signed summary')
        points=[]
        for record in records:
            quality_flags.update(record['quality_flags']);all_classes.update([record['kind']])
            point=record['position_deg'];pm=record['pm_icrs_masyr'];cov=np.asarray(record['covariance'])
            if not np.isfinite(point+pm).all() or cov.shape!=(4,4) or not np.isfinite(cov).all():raise ValueError('Invalid standard state')
            np.testing.assert_allclose(np.sqrt(np.diag(cov)[[1,3]]),record['pm_error_masyr'],rtol=1e-9,atol=1e-9)
            outside+=int(not inside(*point,config['fields'][core]));points.append([*point,*pm])
            for detection in record['detection_ids']:
                owner=members.setdefault(detection,core)
                crossclaims+=int(owner!=core)
            mag=record['magnitude']
            if mag is not None and 20<=mag<23:by_class[record['kind']+'__r'+str(int(mag))].append(record['information'])
        frames.append(pd.DataFrame(points,columns=['ra','dec','pm_east','pm_north'],dtype=float).assign(core=core))
        # Actual individual QSO vectors are pooled, never medians of per-core P68.
        raw=json.loads((folder/'private_known_qso_blocks.json').read_text());center=plan_center(folder)
        lookup={m:{r['validation_local_index']:standard_block(r,center) for r in raw[m]} for m in METHODS}
        instrument_by_exposure={};band_by_exposure={}
        if original is None:
            for chunk in pd.read_csv(source/'detections.csv',usecols=['survey','exposure_id','instrument','band'],chunksize=250000):
                for visit,band,instrument in chunk.loc[chunk.survey.eq('NSC'),['exposure_id','band','instrument']].drop_duplicates().itertuples(index=False,name=None):
                    if pd.notna(instrument):instrument_by_exposure.setdefault(visit,set()).add(str(instrument))
                    if pd.notna(band):band_by_exposure.setdefault(visit,set()).add(str(band))
        oldpoints={};oldrows=[]
        if core in old_fits:
            oldfolder=Path(old_fits[core]);oldrows=json.loads((oldfolder/'private_qso_blocks.json').read_text())
        if core in old_fits:historical[core]=legacy_tangent_replay(oldrows,raw['r_legacy_quality'])
        legacy_missing_position+=sum('position_deg' not in row or 'motion_icrs' not in row for row in oldrows)
        oldrows=[row for row in oldrows if 'position_deg' in row and 'motion_icrs' in row]
        # Fixed spectral identity comes from source spectra index, not PM closeness.
        from .dark_precision import query_unique
        from .astrometry import project
        from scipy.spatial import cKDTree
        qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,config['fields'][core])].reset_index(drop=True)
        if oldrows and len(qso):
            found,nearest=query_unique(cKDTree(project(qso.ra,qso.dec,plan_center(folder))),project(*np.asarray([r['position_deg'] for r in oldrows]).T,plan_center(folder)),500,300);counts=Counter(nearest[found])
            oldpoints={int(nearest[j]):standard_block(oldrows[j],plan_center(folder)) for j in np.flatnonzero(found) if counts[nearest[j]]==1}
        for k in sorted(set.intersection(*(set(v) for v in lookup.values()))):
            r=lookup['r_legacy_quality'][k];allrow=lookup['all_nsc'][k];values={m:(lookup[m][k]['motion']['C0'],lookup[m][k]['error_icrs']['C0']) for m in METHODS}
            values.update({m:(r['motion'][m],r['error_icrs'][m]) for m in ('NSC','Xiao') if m in r['motion']})
            shape=r.get('median_nsc_r_class_star');shape_class='unknown' if shape is None or not np.isfinite(shape) or not 0<=shape<=1 else 'pointlike' if shape>=.8 else 'extended_or_mixed'
            instruments='+'.join(sorted({v for e in allrow['exposures'] for v in instrument_by_exposure.get(e,set())})) or 'unknown'
            bands='+'.join(sorted({v for e in allrow['exposures'] for v in band_by_exposure.get(e,set())})) or 'unknown'
            known.append(dict(instruments=instruments,bands=bands,magnitude=r['magnitude'],shape_class=shape_class,sdss_runs=len([e for e in r['exposures'] if e.startswith('SDSS:')]),nsc_span=allrow['nsc_span_years'],values=values,exposures=allrow['exposures'],nsc_epoch_days=allrow['nsc_nights_utc18']))
            if k in oldpoints and oldpoints[k]['exposures']==r['exposures']:
                old=oldpoints[k];replay.append(dict(pm_delta=np.asarray(r['motion']['C0'])-old['motion']['C0'],error_delta=np.asarray(r['error_icrs']['C0'])-old['error_icrs']['C0']))
            elif core in old_fits:unmatched_old+=1
        blind.extend(json.loads((folder/'private_blind_qso.json').read_text()))
    ownership=close_pairs(pd.concat(frames,ignore_index=True));ownership.update(outside_common_epoch_owner=outside,cross_core_shared_detection_claims=crossclaims)
    catalogue_classes={k:dict(research_fits=len(v),information=summarize_information(v)) for k,v in by_class.items()}
    blind_cohorts={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (21,22):
            part=[r for r in blind if r['kind']==kind and r['magnitude'] is not None and low<=r['magnitude']<low+1];common=[r['values'] for r in part if {'blind_all','C0','NSC','Xiao'}<=r['values'].keys()]
            blind_cohorts[kind+'__r'+str(low)]=dict(position_identified_qso=len(part),all_spectral=comparison([{'blind_all':r['values']['blind_all']} for r in part],SETTINGS),strict_published_common=comparison(common,SETTINGS,(('blind_all','Xiao'),('blind_all','NSC'),('blind_all','C0'))))
    result=dict(completed_cores=len(fits),planned_cores=len(plan['sources']),completed_nominal_core_area_sqdeg=sum(area(config['fields'][k]) for k in fits),frozen_nominal_area_sqdeg=plan['nominal_area_sqdeg'],
        research_r_control_solutions=sum(v['owned_r_control'] for v in spatial.values()),research_physical_r_solutions=sum(v['funnel']['strict_r_solutions'] for v in spatial.values()),research_all_nsc_solutions=sum(v['owned_all_nsc'] for v in spatial.values()),all_magnitude_source_classes=dict(all_classes),source_quality_flag_counts=dict(quality_flags),spectral_matches_without_reference_r=sum(r['magnitude'] is None for r in blind),by_class=catalogue_classes,known_stationary=known_cohorts(known,instrument_diagnostics=original is None),blind_spectral=blind_cohorts,spatial=spatial,
        acquisition=dict(downloaded_nsc_measures_with_halo_overlap=sum(a['downloaded_nsc_measures'] for a in acquisitions),added_nsc_measures_with_halo_overlap=sum(a['new_nsc_measures'] for a in acquisitions),sdss_r_preserved_with_halo_overlap=sum(a['sdss_r_measurements_preserved'] for a in acquisitions),
            detections_by_measured_position_core_ownership=dict(core_input_counts),physically_qualified_supported_nsc_core_detections_by_band=dict(qualified_core_bands)),
        frozen_r_qso_same_exposures_replay=dict(n=len(replay),without_exact_old_exposure_identity=unmatched_old,maximum_abs_pm_change_masyr=float(np.max([abs(r['pm_delta']).max() for r in replay])) if replay else None,maximum_abs_error_change_masyr=float(np.max([abs(r['error_delta']).max() for r in replay])) if replay else None),
        historical_r_output_replay=dict(by_core=historical,archived_controls_without_position_or_icrs_metadata=legacy_missing_position,
            scope='v1.2 archived QSO rows lack coordinates, detection identities and full uncertainties. Only unique exact frozen reference-r plus physical-exposure-list keys compare archived/native-tangent C0 scalar outputs; motion never selects a pair, ambiguous keys remain unresolved. This is software-output QA, not detection-level truth, a new independent source identity, ICRS error replay or a sample used for precision scoring. New scientific comparisons use explicit current spectral indices and full ICRS states.'),
        ownership=ownership,reference_graph_sha256=graph_hash,fit_receipt_sha256=receipts,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Pooled individual vectors on strict common spectral objects, original half-open C cores and one frozen physical reference graph. Source yields are provisional research fits; unknown identity/night correlation/spatial zero remain flagged. Nominal core area is not qualified faint-target exposure area or certified area. Same-exposure frozen r replay is conditional on position identity, not PM similarity. Object-bootstrap intervals and physical-observation sensitivity do not replace calibration of spatial systematics. Counts with halo overlap are explicitly not unique detections.')
    check_aggregate(result);save_json(output,result)
    if outside or crossclaims:raise ValueError('Continuous ownership or shared-detection conflict needs audit before adoption')


def plan_center(folder):
    from .continuous_reference import reference_frame
    return reference_frame((folder/'reference_receipt.json').resolve().parent/'private_c0.npz')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('state','geometry','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    run(**vars(a))


if __name__=='__main__':main()
