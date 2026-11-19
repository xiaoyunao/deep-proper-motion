"""Freeze exact continuous-control members for existing night/band ablation tools."""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .exposure_catalogues import inside
from .independent_summary import standard_block
from .io import save_json,json_default
from .nsc_comparison import qso_matches
from .nsc_information import split_members,night_labels
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .astrometry import project


def freeze(work,science,geometry,region):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());spec=tomllib.loads(geometry.read_text())['fields'][region]
    source=(science/'source_receipt.json').resolve().parent;shared=(science/'reference_receipt.json').resolve().parent;graph=shared/'private_c0.npz';center=reference_frame(graph);solution,gauges=read_offsets(graph,with_gauge=True);gauge=json.loads((science/'nsc_science_catalogue_summary.json').read_text())['reference_gauge'];solution={k:v for k,v in solution.items() if gauges[k]==gauge}
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);support=calibration_support(keys,solution);nsc=data.survey.eq('NSC').to_numpy();physical=physical_candidates(data)
    strict=keep&(~nsc|physical);full=strict|(nsc&physical&support);night=night_labels(data)
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True);points=project(qso.ra,qso.dec,center)
    saved=json.loads((science/'private_known_qso_blocks.json').read_text());lookups={m:{r['validation_local_index']:r for r in saved[m]} for m in ('r_physical_quality','all_nsc')};truth={name:qso_matches(data,xy,mask,points)[1] for name,mask in [('quality_r',strict),('all_nsc_w0',full)]};rows=[];maximum=0.
    for k in sorted(set.intersection(*(set(v) for v in lookups.values()))):
        indices={name:table[k][support[table[k]]] for name,table in truth.items()};split_members(indices['quality_r'],indices['all_nsc_w0'],data,night);values={}
        for name,old in [('quality_r','r_physical_quality'),('all_nsc_w0','all_nsc')]:
            fit=evaluated_fit(indices[name],data,xy,time,sigma,keys,solution,center,config['reference_epoch']);value=(fit['pm_icrs_masyr'],fit['pm_error_masyr']);expected=(lookups[old][k]['motion_icrs']['C0'],lookups[old][k]['error_icrs']['C0'])
            np.testing.assert_allclose(value,expected,rtol=1e-9,atol=1e-8);maximum=max(maximum,float(np.max(abs(np.asarray(value)-np.asarray(expected)))));values[name]=value
        native=standard_block(lookups['r_physical_quality'][k],center)
        values.update({m:(native['motion'][m],native['error_icrs'][m]) for m in ('NSC','Xiao') if m in native['motion']})
        rows.append(dict(qso_local_index=k,magnitude=lookups['r_physical_quality'][k]['magnitude'],sdss_runs=data.exposure_id.iloc[indices['quality_r'][~nsc[indices['quality_r']]]].nunique(),values=values,members={name:data.detection_id.iloc[index].tolist() for name,index in indices.items()}))
    with gzip.open(work/'private_fixed_qso.json.gz','wt') as stream:json.dump(rows,stream,default=json_default,allow_nan=False)
    summary=dict(fixed_qso=len(rows),maximum_pm_and_error_replay_difference=maximum,slurm_job=os.environ['SLURM_JOB_ID'],scope='Current continuous science controls, explicit frozen spectral indices; physical-r and full NSC states/errors replayed before ablation. No source selected by PM or favourable new result. Existing ordinary/repeated C cores are explanatory diagnostics, not new blind validation. Every extra member is non-r NSC and SDSS/r members stay fixed.')
    check_aggregate(summary);save_json(work/'science_ablation_freeze_summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','science','geometry'):p.add_argument(name,type=Path)
    p.add_argument('region');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'science_receipt.json',verify_directory(a.science,('private_known_qso_blocks.json','nsc_science_catalogue_summary.json')))
    source=(a.science/'source_receipt.json').resolve().parent;shared=(a.science/'reference_receipt.json').resolve().parent
    bind(a.work,'source_receipt.json',verify_directory(source,('detections.csv','spectra_union.csv')));bind(a.work,'new_reference_receipt.json',verify_directory(shared,('private_c0.npz',)));bind(a.work,'geometry.toml',a.geometry)
    with run_stage(a.work,PRECISION,a.region,'continuous-science-ablation-v15',inputs=('science_receipt.json','source_receipt.json','new_reference_receipt.json','geometry.toml'),outputs=('private_fixed_qso.json.gz','science_ablation_freeze_summary.json')) as go:
        if go:freeze(**vars(a))


if __name__=='__main__':main()
