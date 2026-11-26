"""Register existing cores by geometry before reading new motion results."""
import argparse
from collections import Counter
import os
from pathlib import Path

import healpy as hp
import numpy as np

from .__main__ import tomllib
from .astrometry import basis
from .blind_catalogue_quality import read
from .continuous_reference import reference_frame
from .io import digest,save_json
from .ngc5466_demo import footprints
from .public_summary import check_aggregate


def angle(ra,dec,center):
    a,d=np.deg2rad([ra,dec]);v=np.stack([np.cos(d)*np.cos(a),np.cos(d)*np.sin(a),np.sin(d)],axis=-1)
    return np.rad2deg(np.arccos(np.clip(v@basis(*center)[0],-1,1)))


def run(current,output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=current/'private_continuation_manifest.json';scope=read(manifest)
    wide=read(current/'continuation_production_science_summary.json')
    parts=[];cores=[]
    ngc={c for _,_,names in footprints(current) for c in names}
    for part in scope['partitions']:
        paths=read(Path(part['owned'])/'fits.json')
        first=Path(next(iter(paths.values())))
        graph=(first/'reference_receipt.json').resolve().parent/'private_c0.npz'
        center=reference_frame(graph);graph_sha=digest(graph)
        with np.load(graph,allow_pickle=False) as arrays:
            names=len(arrays['names']);components=len(set(arrays['gauge']));draw_shape=arrays['draws'].shape
        spec=tomllib.loads(Path(part['geometry']).read_text())['fields']
        hist=np.zeros(18,dtype=np.int64);counts=Counter()
        for core,window in part['region_by_core'].items():
            entry=wide['cores'][core];bounds=spec[core]
            ra=sum(bounds['ra'])/2;dec=sum(bounds['dec'])/2;theta=float(angle(ra,dec,center))
            folder=Path(paths[core]);source=(folder/'source_receipt.json').resolve().parent
            summary=read(folder/'nsc_science_catalogue_summary.json')
            item=dict(core=core,window=window,graph_label=part['label'],graph=str(graph),graph_center=list(center),
                graph_sha256=graph_sha,fit=str(folder),source=str(source),geometry=part['geometry'],
                reference_gauge=summary['reference_gauge'],spec=bounds,center_theta_deg=theta,
                input_rows=summary['input_detections'],research_sources=entry['research_fits'],ngc=core in ngc)
            cores.append(item)
            for pix,n in entry['nside128_counts'].items():
                a,d=hp.pix2ang(128,int(pix),lonlat=True);value=float(angle(a,d,center))
                hist[min(17,int(value//5))]+=n
            counts.update(entry['observing_support_counts'])
        parts.append(dict(label=part['label'],reference_sha256=graph_sha,center_deg=list(center),
            core_count=len(part['region_by_core']),groups=names,components=components,draw_shape=list(draw_shape),
            source_theta_5deg_histogram=hist.tolist(),source_theta_scope='Actual J2015.5 source counts at NSIDE128 pixel centers; angular cell uncertainty <0.5deg, exact selected-source angles supplied by replay.',
            observing_support_counts=dict(counts)))
    # Choose geometry extrema and middle first, with source support and modest
    # input size inside a narrow angle bin. No new motion or residual is read.
    pool=[r for r in cores if r['research_sources']>=500 and not r['ngc']]
    values=np.asarray([r['center_theta_deg'] for r in pool])
    pilot=[]
    for theta in np.quantile(values,[0,.5,1]):
        choices=[r for r in pool if abs(r['center_theta_deg']-theta)<=3 and r not in pilot]
        pilot.append(min(choices,key=lambda r:(r['input_rows'],abs(r['center_theta_deg']-theta),r['core'])))
    selected=list(pilot)
    for part in parts:
        choices=[r for r in pool if r['graph_label']==part['label'] and r not in selected]
        if not any(r['graph_label']==part['label'] for r in selected):selected.append(min(choices,key=lambda r:r['input_rows']))
    for r in cores:
        if r['ngc'] and r not in selected:selected.append(r)
    while len(selected)<16:
        choices=[r for r in pool if r not in selected]
        # Add coverage in angle, without selecting on fitted precision.
        chosen=max(choices,key=lambda r:(min(abs(r['center_theta_deg']-s['center_theta_deg']) for s in selected),-r['input_rows']))
        selected.append(chosen)
    if len(selected)>16:raise ValueError('NGC and graph support exceed registered core budget')
    private=dict(base_commit='6faabde4fe6205675d9ed2e02c631e7f2e350965',manifest=str(manifest),
        manifest_sha256=digest(manifest),parts=parts,pilot=[r['core'] for r in pilot],selected=selected,
        budget_cpu_node_hours=5,candidate_verification_target_cpu_node_hours=1,
        selection='Near/middle/far actual graph angle, adequate source count, minimum input rows in3deg bins; all frozen NGC cores; remaining angle/graph coverage. No new motion performance used.')
    save_json(output/'private_coordinate_plan.json',private)
    public=dict(base_commit=private['base_commit'],manifest_sha256=digest(manifest),graphs=parts,
        pilot=[{k:r[k] for k in ('core','window','graph_label','center_theta_deg','input_rows','research_sources')} for r in pilot],
        selected=[{k:r[k] for k in ('core','window','graph_label','center_theta_deg','input_rows','research_sources','ngc')} for r in selected],
        budget_cpu_node_hours=5,candidate_verification_target_cpu_node_hours=1,
        plan_sha256=digest(output/'private_coordinate_plan.json'),slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public);save_json(output/'v21_coordinate_registration_summary.json',public)
    print('COORDINATE_REGISTERED',[(r['core'],r['center_theta_deg']) for r in pilot],len(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('current',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute resources')
    run(**vars(a))
