"""Keep the original v1.2 geographic ordinary/repeated definitions in v1.5."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .continuous_reference import bind,verify_directory
from .continuous_summary import geographic_sets,estimate
from .independent_summary import standard_block
from .io import digest,save_json
from .nsc_comparison import comparison
from .nsc_science_summary import METHODS,SETTINGS
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation_blocks import metrics

CONTRASTS=(('all_nsc','r_physical_quality'),('all_nsc','Xiao'),('all_nsc','NSC'))


def run(work,state,base_summary,geometry):
    fits=json.loads((state/'fits.json').read_text());baseline=json.loads(base_summary.read_text());spec=tomllib.loads(geometry.read_text())['fields'];sets=geographic_sets(fits,spec);records=[]
    for core,folder in fits.items():
        folder=Path(folder);receipt=verify_directory(folder,('private_known_qso_blocks.json',))
        if digest(receipt)!=baseline['fit_receipt_sha256'][core]:raise ValueError('Different science receipts in geographic readout')
        from .nsc_science_summary import plan_center
        center=plan_center(folder);raw=json.loads((folder/'private_known_qso_blocks.json').read_text());lookup={m:{r['validation_local_index']:standard_block(r,center) for r in raw[m]} for m in METHODS}
        for k in sorted(set.intersection(*(set(v) for v in lookup.values()))):
            row=lookup['r_legacy_quality'][k]
            if not {'NSC','Xiao'}<=row['motion'].keys():continue
            values={m:(lookup[m][k]['motion']['C0'],lookup[m][k]['error_icrs']['C0']) for m in METHODS};values.update({m:(row['motion'][m],row['error_icrs'][m]) for m in ('NSC','Xiao')});full=lookup['all_nsc'][k]
            records.append(dict(core=core,magnitude=row['magnitude'],values=values,exposures=full['exposures'],nsc_epoch_days=full['nsc_nights_utc18']))
    result={}
    for name,cores in sets.items():
        bins={}
        for low in (21,22):
            part=[r for r in records if r['core'] in cores and low<=r['magnitude']<low+1];stats=comparison([r['values'] for r in part],SETTINGS,CONTRASTS)
            sensitivity=[dict(exposures=r['exposures'],nsc_epoch_days=r['nsc_epoch_days'],motion={m:r['values'][m][0] for m in ('NSC','Xiao','r_physical_quality','all_nsc')}) for r in part]
            blocks=[dict(r,motion={'C0':r['values']['Xiao'][0],'C1':r['values']['all_nsc'][0]}) for r in part]
            bins['r'+str(low)]=dict(common=stats,shared_group_sensitivity=estimate(sensitivity,('NSC','Xiao','r_physical_quality','all_nsc'),9137,candidate='all_nsc'),connected_blocks=metrics(blocks))
            if name=='all':
                previous=baseline['known_stationary'][f'r{low}__all']['strict_published_common'];assert stats['n']==previous['n']
                for m in METHODS+('NSC','Xiao'):np.testing.assert_allclose(stats['methods'][m]['radial_abs68'],previous['methods'][m]['radial_abs68'],rtol=0,atol=0)
        result[name]=bins
    for low in (21,22):assert sum(result[name]['r'+str(low)]['common']['n'] for name in ('ordinary','transition','repeat_band'))==result['all']['r'+str(low)]['common']['n']
    payload=dict(cohorts=result,core_counts={k:len(v) for k,v in sets.items()},input_science_summary_sha256=digest(base_summary),source_code_commit=os.environ.get('EVALUATION_COMMIT'),slurm_job=os.environ['SLURM_JOB_ID'],scope='Exact original continuous_summary.geographic_sets: ordinary north<=-3 or south>=3; repeat_band within [-1,+1] Dec; remaining transition. Same current physical controls and common NSC/Xiao objects, individual vectors pooled. All-region count and P68 exactly replay the signed current summary. These geographic bands differ from one/multiple SDSS-run source strata; neither is a new blind region or a universal instrument/noise class.')
    check_aggregate(payload);save_json(work/'geographic_readout_summary.json',payload)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','state','base_summary','geometry'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    for name,path in [('fits.json',a.state/'fits.json'),('base_summary.json',a.base_summary),('geometry.toml',a.geometry)]:bind(a.work,name,path)
    with run_stage(a.work,a.geometry,'all','nsc-geographic-readout-v15',inputs=('fits.json','base_summary.json','geometry.toml'),outputs=('geographic_readout_summary.json',)) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
