"""Bounded recovery of completed accounting blocks; never reload fitted cores."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np

from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate

ORIGINAL_IMPLEMENTATION='c8b865da29e09543235355744d1946fe471efcccf7011a12d2e3151b6daf5b36'


def pack(block,original,output,plan_sha):
    source=original/f'private_member_block_{block:03}.json.gz'
    dest=output/f'private_member_compact_{block:03}.npz'
    meta=output/f'compact_{block:03}_summary.json'
    ticket=dict(input_sha256=digest(source),plan_sha256=plan_sha,
                compactor_sha256=digest(Path(__file__)))
    if meta.exists():
        value=read(meta)
        if value['ticket']!=ticket or value['output_sha256']!=digest(dest):raise ValueError('Changed compact checkpoint')
        return value
    value=read(source)
    if value['ticket']!={'plan_sha256':plan_sha,'implementation_sha256':ORIGINAL_IMPLEMENTATION}:raise ValueError('Completed membership block has other inputs')
    arrays={}
    sizes={}
    for label in ('old','new'):
        entries=value[label]
        keys=np.frombuffer(b''.join(bytes.fromhex(k) for k in entries),dtype='V32')
        lengths=np.fromiter((len(e['record']['detection_ids']) for e in entries.values()),dtype=np.uint32,count=len(entries))
        ids=bytearray()
        for entry in entries.values():
            for identity in entry['record']['detection_ids']:ids.extend(hashlib.sha256(str(identity).encode()).digest()[:16])
        arrays[label+'_keys']=keys
        arrays[label+'_lengths']=lengths
        arrays[label+'_ids']=np.frombuffer(ids,dtype='V16')
        sizes[label]=len(entries)
    np.savez_compressed(dest,**arrays)
    result=dict(ticket=ticket,block=block,cores=value['cores'],before=value['before'],after=value['after'],
        common_same_origin=value['common_same_origin'],pending_counts=sizes,output_sha256=digest(dest),
        compact_bytes=dest.stat().st_size,original_bytes=source.stat().st_size)
    save_json(meta,result);print('COMPACT_BLOCK_DONE',block,sizes,flush=True)
    return result


def overlap_edges(old_ids,old_owner,new_ids,new_owner):
    """Hash-prefix partitions bound sort memory; ownership is checked per survey."""
    edges=Counter();shared=0
    for prefix in range(256):
        left=np.flatnonzero(old_ids.view('u1').reshape(-1,16)[:,0]==prefix)
        right=np.flatnonzero(new_ids.view('u1').reshape(-1,16)[:,0]==prefix)
        a,b=old_ids[left],new_ids[right]
        if len(np.unique(a))!=len(a) or len(np.unique(b))!=len(b):raise ValueError('A frozen physical detection has multiple source claims')
        matches,ia,ib=np.intersect1d(a,b,assume_unique=True,return_indices=True)
        shared+=len(matches)
        codes=(old_owner[left[ia]].astype(np.uint64)<<np.uint64(32))|new_owner[right[ib]].astype(np.uint64)
        pair,count=np.unique(codes,return_counts=True)
        edges.update({int(k):int(v) for k,v in zip(pair,count)})
    return edges,shared


def continuity(edges,old_lengths,new_lengths):
    if not edges:return dict(unique_reciprocal_continuities=0,ambiguous_or_weak_overlap_edges=0)
    code=np.fromiter(edges,dtype=np.uint64,count=len(edges));counts=np.fromiter(edges.values(),dtype=np.uint32,count=len(edges))
    old=(code>>np.uint64(32)).astype(np.int64);new=(code&np.uint64(0xffffffff)).astype(np.int64)
    maximum_old=np.zeros(len(old_lengths),np.uint32);maximum_new=np.zeros(len(new_lengths),np.uint32)
    np.maximum.at(maximum_old,old,counts);np.maximum.at(maximum_new,new,counts)
    best_old=counts==maximum_old[old];best_new=counts==maximum_new[new]
    ties_old=np.bincount(old[best_old],minlength=len(old_lengths));ties_new=np.bincount(new[best_new],minlength=len(new_lengths))
    second_old=np.zeros(len(old_lengths),np.uint32);second_new=np.zeros(len(new_lengths),np.uint32)
    np.maximum.at(second_old,old[~best_old],counts[~best_old]);np.maximum.at(second_new,new[~best_new],counts[~best_new])
    frac_old=counts/old_lengths[old];frac_new=counts/new_lengths[new]
    accepted=best_old&best_new&(ties_old[old]==1)&(ties_new[new]==1)&(frac_old>=.6)&(frac_new>=.6)
    accepted &= (counts.astype(float)-second_old[old])>=.2*old_lengths[old]
    accepted &= (counts.astype(float)-second_new[new])>=.2*new_lengths[new]
    return dict(unique_reciprocal_continuities=int(accepted.sum()),ambiguous_or_weak_overlap_edges=int((~accepted).sum()),
        accepted_fraction_before_p16_p50_p84=np.quantile(frac_old[accepted],[.16,.5,.84]) if accepted.any() else None,
        accepted_fraction_after_p16_p50_p84=np.quantile(frac_new[accepted],[.16,.5,.84]) if accepted.any() else None)


def run(plan_file,original,output):
    start=monotonic();output.mkdir(parents=True,exist_ok=True)
    plan_sha=digest(plan_file);plan=read(plan_file)
    # The completed driver had one 64-core block per affected origin; all block
    # caches must exist, including any new-origin transfers into old owners.
    files=sorted(original.glob('private_member_block_*.json.gz'))
    if not files:raise ValueError('No signed completed member checkpoints')
    if [p.name for p in files]!=[f'private_member_block_{i:03}.json.gz' for i in range(len(files))]:raise ValueError('Missing member block')
    blocks=[pack(i,original,output,plan_sha) for i in range(len(files))]
    arrays={}
    for label in ('old','new'):
        for tag in ('keys','lengths','ids'):
            parts=[]
            for i in range(len(files)):
                with np.load(output/f'private_member_compact_{i:03}.npz') as f:parts.append(f[label+'_'+tag])
            arrays[label+'_'+tag]=np.concatenate(parts)
    old_keys,new_keys=arrays['old_keys'],arrays['new_keys']
    if len(np.unique(old_keys))!=len(old_keys) or len(np.unique(new_keys))!=len(new_keys):raise ValueError('Repeated source membership across blocks')
    exact,left,right=np.intersect1d(old_keys,new_keys,assume_unique=True,return_indices=True)
    for label,common in [('old',left),('new',right)]:
        keep=np.ones(len(arrays[label+'_keys']),bool);keep[common]=False
        lengths=arrays[label+'_lengths']
        selected=np.repeat(keep,lengths)
        owner=np.repeat(np.arange(len(lengths),dtype=np.uint32),lengths)
        arrays[label+'_ids']=arrays[label+'_ids'][selected]
        arrays[label+'_owner']=owner[selected]
    edges,nshared=overlap_edges(arrays['old_ids'],arrays['old_owner'],arrays['new_ids'],arrays['new_owner'])
    linked=continuity(edges,arrays['old_lengths'],arrays['new_lengths'])
    old_count=sum(r['before'] for r in blocks);new_count=sum(r['after'] for r in blocks)
    common_same=sum(r['common_same_origin'] for r in blocks)
    assert common_same+len(old_keys)==old_count
    assert common_same+len(new_keys)==new_count
    expected=sum(r['allband_delta'] for r in plan['changes'])
    if new_count-old_count!=expected:raise ValueError('Recovered final owner counts do not reconcile')
    traces=[]
    for entry in plan['selected']:
        value=read(original/'r_trace'/entry['core']/'v20_r_count_transition_summary.json')
        if value['ticket']['implementation_sha256']!=ORIGINAL_IMPLEMENTATION:raise ValueError('Changed completed physical-r trace')
        if value['new_physical_r']-value['old_physical_r']!=entry['physical_r_delta']:raise ValueError('Changed physical-r count')
        traces.append(value)
    report=dict(blocks=blocks,exact_common_physical_member_sets=common_same+len(exact),
        changed_old_member_sets=len(old_keys)-len(exact),changed_new_member_sets=len(new_keys)-len(exact),
        shared_physical_measurements_in_changed_sets=nshared,member_continuity=linked,
        old_rows_without_unique_continuity=len(old_keys)-len(exact)-linked['unique_reciprocal_continuities'],
        new_rows_without_unique_continuity=len(new_keys)-len(exact)-linked['unique_reciprocal_continuities'],
        before_replayed_owned_sources=old_count,after_replayed_owned_sources=new_count,allband_net_change=expected,
        physical_r_traces=traces,physical_r_traced_net_change=sum(r['new_physical_r']-r['old_physical_r'] for r in traces),
        physical_r_full_net_change=sum(r['physical_r_delta'] for r in plan['changes']),
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],implementation_sha256=digest(Path(__file__)),
        scope='Recover only completed signed checkpoints. Exact member-set changes are not births/deaths of stars. Conditional continuity uses reciprocal unique physical detection overlap >=60% in each direction and >=20% competitor margin. Optical motion/object ID never defines a match. Unresolved rows retain ambiguity.')
    check_aggregate(report);save_json(output/'v20_replay_accounting_summary.json',report)
    print('RECOVERED_ACCOUNTING',common_same+len(exact),linked,expected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan_file','original','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(a))
