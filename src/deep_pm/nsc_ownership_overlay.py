"""Rehome immutable research rows by fitted common-epoch position, without refits."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import bind,verify_directory
from .continuous_ownership import close_pairs
from .exposure_catalogues import inside
from .io import digest,save_json
from .nsc_information import summarize_information
from .nsc_science_summary import SETTINGS
from .nsc_comparison import comparison
from .provenance import run_stage
from .public_summary import check_aggregate

STANDARD='private_standard_motion.json.gz'
CONTROL='private_r_control.json.gz'


def record_hash(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def active_specs(config,fits):
    # Configuration also contains alternative pilots/child boxes not in the run.
    return {core:config['fields'][core] for core in fits}


def transfer(record,origin,slot,specs,filename=STANDARD,archive_exterior=False):
    if inside(*record['position_deg'],specs[origin]):return None
    owners=[core for core,spec in specs.items() if inside(*record['position_deg'],spec)]
    if len(owners)>1 or (not owners and not archive_exterior):
        raise ValueError('Common-epoch position has no unique owner inside the frozen science region')
    return dict(origin_core=origin,source_slot=slot,filename=filename,owner_core=owners[0] if owners else None,
                unchanged_record_sha256=record_hash(record))


def owned_records(state,owner,filename=STANDARD):
    """Read one resolved core; return origin, immutable row index, original row."""
    state=Path(state);verify_directory(state,('fits.json','private_owner_overlay.json'));fits=json.loads((state/'fits.json').read_text());overlay=json.loads((state/'private_owner_overlay.json').read_text())
    if digest(state/'fits.json')!=overlay['fits_manifest_sha256']:raise ValueError('Changed source manifest')
    moves=[r for r in overlay['transfers'] if r['filename']==filename]
    outgoing={(r['origin_core'],r['source_slot']):r for r in moves}
    origins={owner}|{r['origin_core'] for r in moves if r['owner_core']==owner}
    for origin in sorted(origins):
        folder=Path(fits[origin]);verify_directory(folder,(filename,))
        with gzip.open(folder/filename,'rt') as stream:rows=json.load(stream)
        for slot,row in enumerate(rows):
            move=outgoing.get((origin,slot));target=move['owner_core'] if move else origin
            if target!=owner:continue
            if move and record_hash(row)!=move['unchanged_record_sha256']:raise ValueError('Rehomed source state or covariance changed')
            yield origin,slot,row


def run(work,state,base_summary,audit,geometry,archive_exterior=False):
    fits=json.loads((state/'fits.json').read_text());specs=active_specs(tomllib.loads(geometry.read_text()),fits);result=json.loads(base_summary.read_text());bad=json.loads((audit/'private_ownership_failures.json').read_text());proof=json.loads((audit/'ownership_failure_summary.json').read_text())
    if proof['source_manifest_sha256']!=digest(state/'fits.json'):raise ValueError('Ownership audit binds a different source index')
    if result['ownership']['cross_core_shared_detection_claims'] or result['ownership']['outside_common_epoch_owner']!=proof['outside_by_type'][STANDARD]:raise ValueError('Base ownership audit does not match')
    moves=[]
    for row in bad:
        move=transfer(row['record'],row['core'],row['slot'],specs,row['file'],archive_exterior)
        if move is None:raise ValueError('Recorded ownership failure was not outside its anchor core')
        if [core for core in row['common_epoch_owners'] if core in fits]!=([move['owner_core']] if move['owner_core'] is not None else []):raise ValueError('Frozen owner geometry changed')
        moves.append(move)
    affected_set={r[k] for r in moves for k in ('origin_core','owner_core') if r[k] is not None}
    for row in bad:
        if row['file']==STANDARD:
            affected_set.update(core for core,spec in specs.items() if inside(*row['record']['position_deg'],spec,halo=.2/3600))
    affected=sorted(affected_set);grouped=defaultdict(list);controls=Counter();blind=defaultdict(list);transfers={(r['origin_core'],r['filename'],r['source_slot']):r for r in moves};matched=set()
    scanned=sorted(fits) if archive_exterior and moves else affected
    exterior=Counter()
    for core in scanned:
        folder=Path(fits[core]);receipt=verify_directory(folder,(STANDARD,CONTROL,'private_blind_qso.json'))
        if digest(receipt)!=result['fit_receipt_sha256'][core]:raise ValueError('Original fitted receipt changed')
        targets={}
        for filename in (STANDARD,CONTROL):
            with gzip.open(folder/filename,'rt') as stream:rows=json.load(stream)
            for slot,row in enumerate(rows):
                move=transfers.get((core,filename,slot));owner=move['owner_core'] if move else core
                if move:
                    if record_hash(row)!=move['unchanged_record_sha256']:raise ValueError('Science row changed after audit')
                    matched.add((core,filename,slot))
                if owner is None:
                    exterior[filename]+=1
                    if filename==STANDARD:targets[slot]=None
                    continue
                if not inside(*row['position_deg'],specs[owner]):raise ValueError('Unresolved common-epoch ownership')
                if filename==STANDARD:grouped[owner].append(row);targets[slot]=owner
                else:controls[owner]+=1
        for row in json.loads((folder/'private_blind_qso.json').read_text()):
            owner=targets[row['source_slot']]
            if owner is not None:blind[owner].append(row)
    if matched!=set(transfers):raise ValueError('Incomplete ownership transfer')
    before=sum(result['spatial'][core]['owned_all_nsc'] for core in scanned);before_r=sum(result['spatial'][core]['owned_r_control'] for core in scanned)
    if sum(map(len,grouped.values()))+exterior[STANDARD]!=before or sum(controls.values())+exterior[CONTROL]!=before_r:
        raise ValueError('Rehoming lost an unaccounted source')
    for core in affected:
        rows=grouped[core];sample=result['spatial'][core];sample['owned_all_nsc']=len(rows);sample['owned_r_control']=controls[core]
        sample['source_level_band_support']=dict(Counter(b for r in rows for b in r['bands']));sample['nside128_output_counts']=dict(Counter(str(hp.ang2pix(128,*r['position_deg'],lonlat=True)) for r in rows));sample['ownership_resolved_at_reference_epoch']=True
        for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
            for low in (20,21,22):
                part=[r for r in rows if r['kind']==kind and r['magnitude'] is not None and low<=r['magnitude']<low+1]
                sample['by_class'][f'{kind}__r{low}']=dict(research_fits=len(part),information=summarize_information([r['information'] for r in part]),source_level_band_support=dict(Counter(b for r in part for b in r['bands'])),formal_error_p16_p50_p84=np.quantile([r['pm_error_masyr'] for r in part],[.16,.5,.84],axis=0) if part else None)
                part=[r for r in blind[core] if r['kind']==kind and r['magnitude'] is not None and low<=r['magnitude']<low+1];common=[r['values'] for r in part if {'blind_all','C0','NSC','Xiao'}<=r['values'].keys()]
                sample['blind_spectral'][f'{kind}__r{low}']=dict(position_identified_qso=len(part),all_spectral=comparison([{'blind_all':r['values']['blind_all']} for r in part],SETTINGS),strict_published_common=comparison(common,SETTINGS,(('blind_all','Xiao'),('blind_all','NSC'),('blind_all','C0'))))
        sample['scope']+=' Source counts and blind spectra now use common-epoch ownership overlay; discovery funnels and input/known-position-QSO diagnostics retain their original anchor/validation footprint.'
    for core,spec in specs.items():
        if result['spatial'][core]['core_bounds_deg']!=dict(ra_bounds_deg=spec['ra'],dec_bounds_deg=spec['dec']):raise ValueError('Base summary has different geometry')
    if moves:
        if not archive_exterior and any(v['cross_core_pairs'] for v in result['ownership']['cross_core_close_pairs_by_arcsec'].values()):raise ValueError('Pre-existing close pairs require a full new audit')
        nearby=pd.DataFrame([dict(ra=r['position_deg'][0],dec=r['position_deg'][1],pm_east=r['pm_icrs_masyr'][0],pm_north=r['pm_icrs_masyr'][1],core=core) for core,rows in grouped.items() for r in rows])
        result['ownership']['cross_core_close_pairs_by_arcsec']=close_pairs(nearby)['cross_core_close_pairs_by_arcsec']
    if exterior:
        population=[r for rows in grouped.values() for r in rows]
        spectra=[r for rows in blind.values() for r in rows]
        result['research_all_nsc_solutions']=len(population)
        result['research_r_control_solutions']=sum(controls.values())
        result['ownership']['catalogue_count']=len(population)
        result['all_magnitude_source_classes']=dict(Counter(r['kind'] for r in population))
        result['source_quality_flag_counts']=dict(Counter(flag for r in population for flag in r['quality_flags']))
        result['spectral_matches_without_reference_r']=sum(r['magnitude'] is None for r in spectra)
        for key in result['by_class']:
            kind,low=key.split('__r')
            part=[r['information'] for r in population if r['kind']==kind and r['magnitude'] is not None
                  and int(low)<=r['magnitude']<int(low)+1]
            result['by_class'][key]=dict(research_fits=len(part),information=summarize_information(part))
        for key in result['blind_spectral']:
            kind,low=key.split('__r')
            part=[r for r in spectra if r['kind']==kind and r['magnitude'] is not None and int(low)<=r['magnitude']<int(low)+1]
            common=[r['values'] for r in part if {'blind_all','C0','NSC','Xiao'}<=r['values'].keys()]
            result['blind_spectral'][key]=dict(position_identified_qso=len(part),
                all_spectral=comparison([{'blind_all':r['values']['blind_all']} for r in part],SETTINGS),
                strict_published_common=comparison(common,SETTINGS,(('blind_all','Xiao'),('blind_all','NSC'),('blind_all','C0'))))
    overlay=dict(fits_manifest_sha256=digest(state/'fits.json'),geometry_sha256=digest(geometry),transfers=moves,scope='Only row ownership changes. Entire state, covariance, physical members, source class and quality flags stay byte-equivalent as canonical JSON. No internal source dropped, duplicated or refit. Null owner archives a valid fitted row outside the field union. Discovery funnel counts remain anchor-based.')
    save_json(work/'private_owner_overlay.json',overlay)
    for filename in ('fits.json','sources.json','private_frozen_plan.json'):bind(work,filename,state/filename)
    audit_summary=dict(reassigned_all_nsc_rows=sum(r['filename']==STANDARD and r['owner_core'] is not None for r in moves),reassigned_r_control_rows=sum(r['filename']==CONTROL and r['owner_core'] is not None for r in moves),affected_cores=affected,all_source_states_covariances_and_members_unchanged=True,retained_all_nsc_solutions=result['research_all_nsc_solutions'],outside_full_region=sum(exterior.values()),unresolved_owners=0,base_summary_sha256=digest(base_summary),audit_summary_sha256=digest(audit/'ownership_failure_summary.json'),private_overlay_sha256=digest(work/'private_owner_overlay.json'),source_code_commit=os.environ.get('EVALUATION_COMMIT') or os.environ.get('DPM_COMMIT'),slurm_job=os.environ['SLURM_JOB_ID'],scope='A source folder was selected by NSC anchor median, whereas output owner is its fitted J2015.5 ICRS position. Each transferred row has one other owner among the original cores. Rehome via a signed private overlay, keep original receipts, recompute affected source/quality spatial summaries. Global source counts and every pooled QSO metric remain exactly unchanged. Physical-r funnel counts by core retain discovery anchor ownership; global physical-r count is unchanged.')
    if exterior:
        audit_summary.update(exterior_rows_by_product=dict(exterior),
            scope='Every final in-footprint row has a unique J2015.5 owner. Exterior rows remain immutable in the original fit and are explicitly archived by a null owner. Source and blind-spectral statistics use retained rows; known-position QSO comparisons and discovery funnels retain their original footprint.')
    result['ownership'].update(outside_common_epoch_owner=0,original_anchor_folder_mismatches=proof['outside_by_type'][STANDARD],common_epoch_reassigned_source_rows=audit_summary['reassigned_all_nsc_rows']);result['ownership_resolution']=audit_summary
    check_aggregate(audit_summary);check_aggregate(result);save_json(work/'ownership_resolution_summary.json',audit_summary);save_json(work/'continuous_science_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','state','base_summary','audit','geometry'):p.add_argument(name,type=Path)
    p.add_argument('--archive-exterior',action='store_true')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    for filename,path in [('state_fits.json',a.state/'fits.json'),('base_summary.json',a.base_summary),('audit_failures.json',a.audit/'private_ownership_failures.json'),('audit_summary.json',a.audit/'ownership_failure_summary.json'),('geometry.toml',a.geometry)]:bind(a.work,filename,path)
    with run_stage(a.work,a.geometry,'all','nsc-ownership-overlay-v17' if a.archive_exterior else 'nsc-ownership-overlay-v15',inputs=('state_fits.json','base_summary.json','audit_failures.json','audit_summary.json','geometry.toml'),outputs=('private_owner_overlay.json','ownership_resolution_summary.json','continuous_science_summary.json','fits.json','sources.json','private_frozen_plan.json')) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
