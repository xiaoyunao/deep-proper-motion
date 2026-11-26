"""Inventory existing validation, CCD and photometry metadata before new queries."""
import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np

from .blind_catalogue_quality import read
from .continuous_instruments import chip_receipts
from .io import save_json
from .public_summary import check_aggregate
from .io import digest


def run(registration,root,output):
    plan=read(registration);output.mkdir(parents=True,exist_ok=True);report=dict(cores={},schemas={})
    selected=[r for r in plan['selected'] if r['ngc']][:1]+[r for r in plan['selected'] if r['core'] in plan['pilot']]
    for row in selected:
        source=Path(row['source']);chips,complete=chip_receipts(source/'run_manifest.json')
        columns={}
        for name in ('sdss.csv','parent_sdss.csv','validation/gaia.csv','chips.csv'):
            file=source/name
            if file.exists():columns[name]=list(pd.read_csv(file,nrows=0))
        report['cores'][row['core']]=dict(input_headers=columns,verified_chip_parts=len(chips),chip_parts_complete=complete,
            chip_columns=list(pd.read_csv(chips[0],nrows=0)) if chips else [],
            source_file_names=sorted(p.name for p in source.iterdir()))
    for name in ('nsc_schema.csv','native_schema.csv'):
        file=root/'cache'/name
        if not file.exists():continue
        table=pd.read_csv(file)
        # This is schema text only: column names/units/descriptions, no source rows.
        report['schemas'][name]=dict(columns=list(table),schema_records=table.astype(str).to_dict('records'))
    report['slurm_job']=os.environ['SLURM_JOB_ID'];check_aggregate(report)
    save_json(output/'v21_support_metadata_summary.json',report)
    print('SUPPORT_METADATA_DONE',len(selected),flush=True)


def gauge_audit(registration,output):
    """Audit the actual median-NSC gauge, not a fictitious single zero group."""
    from hashlib import sha256
    plan=read(registration);seen=set();graphs=[];private=[]
    for row in plan['selected']:
        path=Path(row['graph'])
        if str(path) in seen:continue
        seen.add(str(path))
        unfiltered=path.with_name('private_c0_unfiltered.npz');precut=unfiltered.exists()
        with np.load(unfiltered if precut else path,allow_pickle=False) as a:
            names=a['names'];nsc=np.char.startswith(names,'NSC:');points=a['position'];draws=a['draws']
            zero=(np.all(points==0,axis=1)&np.all(draws==0,axis=(1,2))).sum()
            private.append(dict(graph=str(path),available_pre_SE_gauge_universe=precut,nsc_groups=names[nsc].tolist()))
            graphs.append(dict(label=row['graph_label'],graph_sha256=digest(path),pre_SE_graph_available=precut,
                registered_NSC_groups=int(nsc.sum()),NSC_group_set_signature=sha256('\n'.join(sorted(names[nsc])).encode()).hexdigest(),
                median_NSC_offset_graph_mas=np.median(points[nsc],axis=0),
                maximum_abs_draw_NSC_median_graph_mas=float(np.max(abs(np.median(draws[nsc],axis=0)))),
                exactly_zero_single_group_count=int(zero)))
    output.mkdir(parents=True,exist_ok=True);save_json(output/'private_NSC_gauge_sets.json',private)
    r=dict(graphs=graphs,continuous_reference_source_sha256=digest(Path(__file__).with_name('continuous_reference.py')),
        actual_policy='Within each solved component, point offsets subtract the median NSC group point. Each bootstrap group subtracts its own draw mean, then all groups subtract the per-draw median of these centred NSC draws. These operations precede the12graph-mas SE selection. No individual physical observation group remains pinned to zero in serialized C0.',
        evidence_scope='If unfiltered graph unavailable, retained group set is only a post-SE diagnostic subset; its NSC medians need not be exactly zero. It cannot reconstruct the original pre-SE anchor universe. Bounded trial fixes external archived offsets and thus retains their gauge. Complete re-solve must register pre-SE NSC group universe and normalization; never create separate per-core zeros.',
        supersedes='v21_gauge_anchor_summary used a single-zero-group heuristic, found zero anchors, and is not a valid anchor registration.',slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(r);save_json(output/'v21_gauge_definition_summary.json',r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','root','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
