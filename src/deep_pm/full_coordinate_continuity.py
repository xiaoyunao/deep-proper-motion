"""Cross-core physical-member continuity after global ownership closes.

Identity overlap is an accounting link, never a stellar-truth assertion.
Exact common sources are removed before changed-member competition. All
remaining old/new hypotheses compete through globally identical detections.
"""
import argparse
from pathlib import Path
import os
import pandas as pd
from .blind_catalogue_quality import read
from .full_coordinate_catalogue import database
from .io import digest,save_json
from .public_summary import check_aggregate


def choose_pairs(overlaps):
    """Mutual unique maximum shared-member count above both 60% fractions."""
    q=overlaps.loc[(overlaps.shared>=.6*overlaps.old_n)&(overlaps.shared>=.6*overlaps.new_n)].copy()
    if q.empty:return q
    left=q.loc[q.shared.eq(q.groupby('old_src').shared.transform('max'))]
    right=q.loc[q.shared.eq(q.groupby('new_src').shared.transform('max'))]
    left=left.loc[left.groupby('old_src').old_src.transform('size').eq(1)]
    right=right.loc[right.groupby('new_src').new_src.transform('size').eq(1)]
    return left.merge(right[['old_src','new_src']],on=['old_src','new_src'],how='inner')


def run(root):
    dest=root/'continuity';dest.mkdir(exist_ok=True);claim_dir=dest/'claims';claim_dir.mkdir(exist_ok=True);db=database(root)
    exact=db.execute('SELECT old_src,new_src FROM exact_common').fetchdf();old_exact=set(exact.old_src);new_exact=set(exact.new_src)
    accepted=set(db.execute('SELECT src FROM accepted').fetchnumpy()['src'])
    plan=read(root/'registration/private_full_coordinate_plan.json');signatures=[]
    for e in plan['cores']:
        folder=root/'catalogue/index'/e['core'];source=folder/'private_changed_members.json.gz';target=claim_dir/(e['core']+'.parquet');receipt=target.with_suffix('.json')
        signature=digest(source);signatures.append(signature)
        ticket=dict(source_sha256=signature,closure_sha256=digest(root/'catalogue/private_closure_binding.json'))
        if receipt.exists():
            proof=read(receipt)
            if proof['ticket']!=ticket or proof['sha256']!=digest(target):raise ValueError('Changed member-continuity input')
            continue
        claims=[]
        for r in read(source):
            old='old_src' in r;ident=r['old_src'] if old else r['src']
            if (old and ident in old_exact) or (not old and (ident in new_exact or ident not in accepted)):continue
            claims.extend((d,ident,old,len(r['detection_ids'])) for d in r['detection_ids'])
        pd.DataFrame(claims,columns=['detection_id','src','old_side','n_members']).to_parquet(target,index=False,compression='zstd')
        save_json(receipt,dict(ticket=ticket,sha256=digest(target)))
    glob=str(claim_dir/'*.parquet');db.execute(f"CREATE OR REPLACE VIEW changed_claims AS SELECT * FROM read_parquet('{glob}')")
    # Partition on physical detection, not on origin core: boundary transfers
    # and candidates discovered in neighbouring halos still meet exactly once.
    buckets=dest/'detection_buckets';partition_receipt=dest/'private_partition.json'
    binding=dict(signatures=signatures,closure_sha256=digest(root/'catalogue/private_closure_binding.json'))
    if partition_receipt.exists():
        prior=read(partition_receipt)
        if prior['binding']!=binding:raise ValueError('Changed continuity partition')
        for name,checksum in prior['files'].items():
            if digest(dest/name)!=checksum:raise ValueError('Changed completed continuity partition')
    else:
        if buckets.exists():raise ValueError('Incomplete continuity partition requires a fresh recovery directory')
        db.execute(f"COPY (SELECT *,hash(detection_id)%64 bucket FROM changed_claims) TO '{buckets}' (FORMAT PARQUET,PARTITION_BY(bucket),COMPRESSION ZSTD)")
        save_json(partition_receipt,dict(binding=binding,files={str(p.relative_to(dest)):digest(p) for p in buckets.glob('**/*.parquet')}))
    outputs=[]
    for bucket in range(64):
        path=dest/f'private_overlap_{bucket:02d}.parquet';outputs.append(path)
        marker=path.with_suffix('.json')
        if marker.exists():
            prior=read(marker)
            if prior['binding_sha256']!=digest(partition_receipt) or prior['sha256']!=digest(path):raise ValueError('Changed continuity overlap checkpoint')
            continue
        inputs=str(buckets/f'bucket={bucket}'/'*.parquet')
        if not (buckets/f'bucket={bucket}').exists():continue
        temporary=path.with_suffix('.pending.parquet')
        db.execute(f'''COPY (SELECT o.src old_src,n.src new_src,count(*) shared,max(o.n_members) old_n,max(n.n_members) new_n
          FROM read_parquet('{inputs}') o JOIN read_parquet('{inputs}') n USING(detection_id)
          WHERE o.old_side AND NOT n.old_side 
          GROUP BY o.src,n.src) TO '{temporary}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
        temporary.replace(path);save_json(marker,dict(binding_sha256=digest(partition_receipt),sha256=digest(path)));print('CONTINUITY_BUCKET',bucket,flush=True)
    files=[str(p) for p in outputs if p.exists()]
    overlap=db.execute('SELECT old_src,new_src,sum(shared)::BIGINT shared,max(old_n) old_n,max(new_n) new_n FROM read_parquet(?) GROUP BY old_src,new_src',[files]).fetchdf()
    pairs=choose_pairs(overlap);pairs.to_parquet(dest/'private_unique_changed_links.parquet',index=False,compression='zstd')
    old_total=db.execute('SELECT count(*) FROM old').fetchone()[0];new_total=len(accepted)
    linked_old=set(pairs.old_src);linked_new=set(pairs.new_src)
    out=dict(exact_members=len(exact),changed_members_mutual_unique=len(pairs),old_retained=old_total,new_retained=new_total,
        old_without_exact_or_unique_continuity=old_total-len(old_exact)-len(linked_old),new_without_exact_or_unique_continuity=new_total-len(new_exact)-len(linked_new),
        old_changed_with_multiple_overlap_hypotheses=int((overlap.groupby('old_src').size()>1).sum()),
        new_changed_with_multiple_overlap_hypotheses=int((overlap.groupby('new_src').size()>1).sum()),
        overlap_pairs=len(overlap),cross_core_unique_changed_links=int(((pairs.old_src.to_numpy()>>32)!=(pairs.new_src.to_numpy()>>32)).sum()),
        minimum_both_member_fraction=.6,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='All changed-member hypotheses compete across origin cores by physical detection IDs after exact global identities and new quarantine are accounted for. Bidirectionally unique maximum shared-member count, >=60% each. Unlinked/changed membership is not a discovered or missing physical star. Split/merge counts describe competing overlaps, not confirmed astrophysical fragmentation.')
    check_aggregate(out);save_json(dest/'v22_global_member_continuity_summary.json',out);db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root)
