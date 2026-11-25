"""Metadata-only accounting and ancillary-input inventory for v2 diagnostics."""
import argparse
from collections import Counter
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from .blind_catalogue_quality import read
from .io import digest,save_json
from .ngc5466_demo import CENTER,footprints
from .public_summary import check_aggregate


def core_index(frozen):
    result={}
    for part in read(frozen/'private_continuation_manifest.json')['partitions']:
        paths=read(Path(part['owned'])/'fits.json')
        for core,window in part['region_by_core'].items():result[core]=dict(fit=paths[core],window=window,owned=part['owned'])
    return result


def run(previous,current,root,output):
    output.mkdir(parents=True,exist_ok=True)
    for frozen in (previous,current):
        if digest(frozen/'continuation_production_science_summary.json')!=read(frozen/'continuation_science_receipt.json')['combined_science_sha256']:raise ValueError('Changed scientific accounting inputs')
    old,new=[read(f/'continuation_production_science_summary.json')['cores'] for f in (previous,current)]
    before,after=core_index(previous),core_index(current)
    if not set(old)<=set(new):raise ValueError('A previous owner core is absent')
    changes=[]
    for core,prior in old.items():
        now=new[core]
        a=now['research_fits']-prior['research_fits'];r=now['physical_r_fits']-prior['physical_r_fits']
        if a or r:
            changes.append(dict(core=core,window=after[core]['window'],allband_delta=a,physical_r_delta=r,
                before_allband=prior['research_fits'],after_allband=now['research_fits'],
                before_physical_r=prior['physical_r_fits'],after_physical_r=now['physical_r_fits'],
                before=before[core],after=after[core]))
    selected=sorted(changes,key=lambda r:(-abs(r['physical_r_delta']),r['core']))[:8]
    save_json(output/'private_replay_accounting_plan.json',dict(previous=str(previous),current=str(current),
        previous_manifest_sha256=digest(previous/'private_continuation_manifest.json'),
        current_manifest_sha256=digest(current/'private_continuation_manifest.json'),
        changes=changes,selected=selected,selection='Largest absolute physical-r count changes; causal audit, not precision model selection.'))
    windows={}
    for window in sorted({r['window'] for r in changes}):
        rows=[r for r in changes if r['window']==window]
        windows[window]=dict(cores_with_count_change=len(rows),allband_delta=sum(r['allband_delta'] for r in rows),physical_r_delta=sum(r['physical_r_delta'] for r in rows))
    report=dict(implementation_sha256=digest(Path(__file__)),old_owner_cores=len(old),new_owner_cores=len(set(new)-set(old)),
        old_allband_net_change=sum(r['allband_delta'] for r in changes),
        old_physical_r_net_change=sum(r['physical_r_delta'] for r in changes),windows=windows,
        count_changed_cores=len(changes),selected_count_diagnostic_cores=len(selected),
        selected_physical_r_net_change=sum(r['physical_r_delta'] for r in selected),
        selected_physical_r_absolute_change=sum(abs(r['physical_r_delta']) for r in selected),
        all_physical_r_absolute_change=sum(abs(r['physical_r_delta']) for r in changes),
        slurm_job=os.environ['SLURM_JOB_ID'])
    ngc=[]
    for owned,specs,cores in footprints(previous):
        index=read(owned/'fits.json')
        ngc.extend(dict(core=c,fit=index[c],owned=str(owned),spec=specs[c]) for c in cores)
    save_json(output/'private_ngc_inputs.json',ngc)
    first=Path(ngc[0]['fit']);source=(first/'source_receipt.json').resolve().parent
    gaia=source/'validation/gaia.csv'
    report['ngc']=dict(cores=len(ngc),gaia_columns=list(pd.read_csv(gaia,nrows=0)) if gaia.exists() else [],
        validation_file_names=[p.name for p in (source/'validation').iterdir()] if (source/'validation').is_dir() else [])
    bricks=root.parent/'survey-bricks.fits.gz'
    with fits.open(bricks,memmap=False) as handle:
        t=handle[1].data
        near=(abs(t['RA']-CENTER[0])<.8)&(abs(t['DEC']-CENTER[1])<.7)
        choices=t[near]
        row=min(choices,key=lambda r:abs(r['RA']-CENTER[0])+abs(r['DEC']-CENTER[1]))
        name=str(row['BRICKNAME']).strip()
    path=root/'coadd'/name[:3]/name
    kinds=('maskbits','nexp-g','nexp-r','depth-g','depth-r')
    report['ngc']['pixel_products_available']={kind:(path/f'legacysurvey-{name}-{kind}.fits.fz').is_file() for kind in kinds}
    report['ngc']['local_bricks']=len(choices)
    report['ngc']['brick_metadata_columns']=list(choices.dtype.names)
    check_aggregate(report);save_json(output/'v20_input_accounting_summary.json',report)
    print('ACCOUNTING',report['old_allband_net_change'],report['old_physical_r_net_change'],'changed cores',len(changes),'top8 r delta',report['selected_physical_r_net_change'],flush=True)
    print('NGC_INPUTS',report['ngc'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','current','root','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(a))
