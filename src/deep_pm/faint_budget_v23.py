"""Eight metadata-preselected cores: frozen covariance and bright residuals."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import os
import numpy as np
import pandas as pd
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from .astrometry import project,apparent_positions
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree,forward_jacobian
from .dark_precision import query_unique
from .full_coordinate_sources import reference_context,write_rows
from .faint_covariance_v23 import decompose
from .stream_json import array_items
from .continuous_reference import PRECISION,SPATIAL
from .__main__ import tomllib
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .io import digest,save_json
from .public_summary import check_aggregate


def registration(base,root):
    plan=read(base/'v2.2/registration/private_full_coordinate_plan.json');entries={r['core']:r for r in plan['cores']}
    selected=[r['cores'][0] for r in read(root/'preflight/private_HST_geometric_registration.json')['selected'] if r['cores']]
    for core in ['w230_110_64_00','w230_090_60_11','w020_110_20_11']:
        if core in entries and core not in selected:selected.append(core)
    for window in ['north_two','w140_090']:
        options=sorted(e['core'] for e in plan['cores'] if e['window']==window and e['core'] not in selected)
        if options:selected.append(options[len(options)//2])
    selected=list(dict.fromkeys(selected))[:8]
    proof=dict(cores=selected,plan_sha256=digest(base/'v2.2/registration/private_full_coordinate_plan.json'),
        rule='HST centre footprint cores plus the three pre-existing v22 epoch/geometry controls (one per reference chart) and lexicographic midpoint cores of north_two/w140_090 if available. No PM precision or residual used. At most100 per actual r magnitude bin20/21/22 selected by frozen physical member hash, plus independently bright Gaia controls.')
    dest=root/'budget';dest.mkdir(exist_ok=True);file=dest/'private_metadata_core_selection.json'
    if file.exists() and read(file)!=proof:raise ValueError('Metadata core plan changed')
    save_json(file,proof);return [entries[c] for c in selected]


def run(base,root):
    config=tomllib.loads(PRECISION.read_text());prod=tomllib.loads(SPATIAL.read_text())['production'];entries=registration(base,root);folder=root/'budget';results=[];position=[]
    for e in entries:
        core=e['core'];dest=folder/core;dest.mkdir(exist_ok=True);source=base/'v2.2/sources'/core;receipt=dest/'private_completed.json';ticket=dict(source_summary_sha256=digest(source/'v22_source_core_summary.json'),selection_sha256=digest(folder/'private_metadata_core_selection.json'),implementation_sha256=digest(Path(__file__)))
        if receipt.exists():
            prior=read(receipt)
            if prior['ticket']!=ticket:raise ValueError('Changed budget input')
            results.append(read(dest/'v23_core_budget_summary.json'));continue
        index=pd.read_parquet(base/'v2.2/catalogue/index'/core/'science.parquet',columns=['slot','owner','member_sha256','magnitude','sdss_runs','nsc_span'])
        index=index.loc[index.owner.notna()].copy();selected=[]
        for lower in [20,21,22]:selected.extend(index.loc[index.magnitude.between(lower,lower+1,inclusive='left')].sort_values('member_sha256').head(100).slot.tolist())
        bright=index.loc[index.magnitude<20].sort_values('member_sha256').head(500).slot.tolist();wanted=set(selected+bright)
        rows={j:r for j,r in enumerate(array_items(source/'private_standard_motion.json.gz')) if j in wanted}
        data,xy,t,sigma,_=quality_detections(Path(e['source']),e['center'],config,prod);groups=calibration_groups(data);contexts={};records=[]
        for slot in selected:
            row=rows[slot];component=int(row['reference_component'])
            if component not in contexts:contexts[component]=reference_context(str(base/'v2.2/reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz'))
            part=decompose(row,data,t,sigma,contexts[component][1]);records.append(dict(slot=slot,r=row['magnitude'],sdss_runs=row['information']['sdss_physical_runs'],nsc_span=row['information']['nsc_span_years'],independent_terms=part['survey_terms'],reference=part['reference'],total=part['total'],reconstruction_error=part['maximum_covariance_reconstruction_error']))
        gaiafile=Path(e['source'])/'validation/gaia.csv';g=pd.read_csv(gaiafile).rename(columns=str.lower);required=['ra','dec','pmra','pmdec','parallax','ruwe','pmra_error','pmdec_error','ref_epoch','phot_g_mean_mag']
        if not set(required)<=set(g):raise ValueError('Bright Gaia control schema incomplete')
        g=g.loc[np.isfinite(g[required]).all(axis=1)&(g.ruwe<1.4)&(g.pmra_error<.2)&(g.pmdec_error<.2)&(g.phot_g_mean_mag<20)&(abs(g.parallax)<10)].reset_index(drop=True)
        center=tuple(e['center']);tree=None
        if len(g):
            vel=np.column_stack([g.pmra,g.pmdec]);points=project(g.ra,g.dec,center)+np.einsum('nij,nj->ni',forward_jacobian(g.ra,g.dec,center),vel)*(2015.5-g.ref_epoch.to_numpy())[:,None];tree=AngularTree(points,center)
        bright_records=[]
        if tree is not None and bright:
            points=project(*np.array([rows[i]['position_deg'] for i in bright]).T,center);good,near=query_unique(tree,points,700,300)
            for k in np.flatnonzero(good)[:100]:
                slot=bright[k];row=rows[slot];truth=g.iloc[near[k]];component=int(row['reference_component'])
                if component not in contexts:contexts[component]=reference_context(str(base/'v2.2/reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz'))
                graph_center,solution,_,_,graphsha=contexts[component];ix=pd.Index(data.detection_id).get_indexer(row['detection_ids'])
                if (ix<0).any():raise ValueError('Bright frozen member missing')
                sky=project(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],graph_center);zero=np.array([solution[k][0] for k in groups[ix]])
                observer=get_body_barycentric('earth',Time(t[ix],format='jyear',scale='tt')).xyz.to_value('AU').T
                rv=float(truth.get('radial_velocity',0));rv=rv if np.isfinite(rv) else 0.
                predicted=apparent_positions(float(truth.ra),float(truth.dec),float(truth.pmra),float(truth.pmdec),float(truth.parallax),rv,t[ix]-float(truth.ref_epoch),observer,graph_center)
                inv=np.linalg.inv(forward_jacobian(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],graph_center));residual=np.einsum('nij,nj->ni',inv,sky-zero-predicted)
                for j,i in enumerate(ix):bright_records.append(dict(source_slot=slot,survey=str(data.survey.iloc[i]),band=str(data.band.iloc[i]),instrument=str(data.instrument.iloc[i]) if 'instrument' in data else 'unknown',physical_group=str(groups[i]),CCD=str(data.detector_id.iloc[i]),field=str(data.field_id.iloc[i]) if 'field_id' in data else 'unknown',time_year=float(t[i]),residual_native_mas=residual[j],r=row['magnitude']))
        write_rows(dest/'private_covariance_budget.json.gz',records);write_rows(dest/'private_bright_position_residuals.json.gz',bright_records)
        summaries={}
        for lower in [20,21,22]:
            part=[r for r in records if lower<=r['r']<lower+1];summary=dict(n=len(part))
            if part:
                for term in ['SDSS','NSC','reference']:
                    matrices=np.array([r['reference'] if term=='reference' else r['independent_terms'][term] for r in part]);total=np.array([r['total'] for r in part]);summary[term]=dict(matrix_mean=np.mean(matrices,axis=0),axis_variance_fraction_p16_p50_p84=np.quantile(np.diagonal(matrices,axis1=1,axis2=2)/np.diagonal(total,axis1=1,axis2=2),[.16,.5,.84],axis=0))
            summaries[str(lower)]=summary
        blocks=[]
        for mode in ['survey_band','physical_group','CCD','field']:
            parts=defaultdict(list)
            for r in bright_records:
                key=(r['survey'],r['band']) if mode=='survey_band' else (r['survey'],r['band'],r['physical_group'],r[mode] if mode in ['CCD','field'] else '')
                parts[str(key)].append(r)
            for key,part in parts.items():
                perstar=defaultdict(list)
                for r in part:perstar[r['source_slot']].append(r['residual_native_mas'])
                values=np.array([np.median(v,axis=0) for v in perstar.values()]);block=dict(mode=mode,observational_block=key,independent_bright_stars=len(values),measurements=len(part))
                if len(values)>=5:
                    med=np.median(values,axis=0);block.update(median_native_position_residual_mas=med,nmad_native_position_residual_mas=1.4826*np.median(abs(values-med),axis=0))
                blocks.append(block)
        result=dict(core=core,covariance_by_actual_source_r=summaries,maximum_covariance_reconstruction_error=max([r['reconstruction_error'] for r in records],default=0.),bright_control_sources=len({r['source_slot'] for r in bright_records}),bright_control_measurements=len(bright_records),bright_position_blocks=blocks,
            source_summary_sha256=ticket['source_summary_sha256'],scope='Metadata preselected core, fixed physical member hashes and saved weights. Survey+reference full2D terms close saved covariance. Bright Gaia conditional apparent positions include annual parallax; exposure/CCD/field blocks first aggregate per star, require>=5 stars for residual statistics. Shared-group/stellar colour effects are diagnostics, not new calibration or independent spatial certification.')
        check_aggregate(result);save_json(dest/'v23_core_budget_summary.json',result);save_json(receipt,dict(ticket=ticket,summary_sha256=digest(dest/'v23_core_budget_summary.json')));results.append(result)
        print('V23_FROZEN_BUDGET_CORE',core,len(records),result['bright_control_sources'],flush=True)
    out=dict(cores=results,metadata_selection_sha256=digest(folder/'private_metadata_core_selection.json'),slurm_job=os.environ['SLURM_JOB_ID'],scope='At most eight actual preselected cores. Native50mas already included once; conditional survey covariance split preserves off-diagonal terms and the complete saved estimator. No new source fitting, g coefficients, morphology thresholds or accepted-source changes.')
    check_aggregate(out);save_json(folder/'v23_frozen_error_budget_summary.json',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.base,a.root)
