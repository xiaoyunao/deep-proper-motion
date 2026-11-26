"""Stream full-coordinate source indices and close global physical uniqueness.

Raw fits stay immutable. Shared detections quarantine their competing sources
until a separately recorded local identity resolution; unaffected sources close.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import os
import inspect
from pathlib import Path
import resource
from time import monotonic

import healpy as hp
import numpy as np
import pandas as pd

from .blind_catalogue_quality import read
from .continuous_summary import area
from .exposure_catalogues import inside
from .io import digest,save_json
from .public_summary import check_aggregate
from .full_coordinate_sources import write_rows
from .usability_accounting import index as old_index


def member_key(ids):return sha256('\0'.join(sorted(ids)).encode()).hexdigest()


def owners(rows,origin,specs):
    result=[origin]*len(rows)
    if not rows:return result
    xy=np.asarray([r['position_deg'] for r in rows]);outside=np.flatnonzero(~inside(*xy.T,specs[origin]))
    for i in outside:
        matches=[core for core,spec in specs.items() if inside(*xy[i],spec)]
        if len(matches)>1:raise ValueError('Overlapping common-epoch owners')
        result[i]=matches[0] if matches else None
    return result


def index_core(task):
    number,entry,root,old_entries=task;root=Path(root);core=entry['core'];folder=root/'sources'/core;dest=root/'catalogue/index'/core;dest.mkdir(parents=True,exist_ok=True)
    plan=read(root/'registration/private_full_coordinate_plan.json');specs={e['core']:e['spec'] for e in plan['cores']}
    proof=read(folder/'v22_source_core_summary.json');old_entry=old_entries[core]
    ticket=dict(source_summary_sha256=digest(folder/'v22_source_core_summary.json'),old_sha256=digest(Path(old_entry['path'])/'private_standard_motion.json.gz'),implementation_sha256=sha256((inspect.getsource(index_core)+inspect.getsource(owners)+inspect.getsource(member_key)).encode()).hexdigest())
    receipt=dest/'private_index_receipt.json'
    if receipt.exists():
        done=read(receipt)
        if done['ticket']!=ticket:raise ValueError('Changed catalogue index input')
        for name,checksum in done['files'].items():
            if digest(dest/name)!=checksum:raise ValueError('Changed completed catalogue index')
        return done
    start=monotonic();science=read(folder/'private_standard_motion.json.gz');ownership=owners(science,core,specs)
    compact=[];claims=[];changed=[];prior=[];older=read(Path(old_entry['path'])/'private_standard_motion.json.gz')
    old_by_key={member_key(r['detection_ids']):j for j,r in enumerate(older)};new_keys=set()
    for slot,(r,owner) in enumerate(zip(science,ownership)):
        ident=(number<<32)+slot;key=member_key(r['detection_ids']);new_keys.add(key);info=r['information'];cov=np.asarray(r['covariance'])
        if np.linalg.eigvalsh(cov).min()<=0 or not np.isfinite(cov).all():raise ValueError('Invalid ICRS covariance')
        if r['reference_epoch']!=2015.5:raise ValueError('Changed output epoch')
        ids=r['detection_ids']
        if len(set(ids))!=len(ids):raise ValueError('Repeated member in a science source')
        compact.append(dict(src=ident,core=core,slot=slot,owner=owner,member_sha256=key,n_members=len(ids),component=int(r['reference_component']),reference_model_sha256=r['reference_model_sha256'],reference_epoch=r['reference_epoch'],
            **{f'cov_{a}{b}':float(cov[a,b]) for a in range(4) for b in range(a,4)},
            independent_pm_variance_e=float(r['independent_icrs_covariance'][1][1]),independent_pm_variance_n=float(r['independent_icrs_covariance'][3][3]),
            reference_pm_variance_e=float(r['reference_icrs_covariance'][1][1]),reference_pm_variance_n=float(r['reference_icrs_covariance'][3][3]),
            ra_deg=r['position_deg'][0],dec_deg=r['position_deg'][1],pixel128=int(hp.ang2pix(128,*r['position_deg'],lonlat=True)),
            pm_e=r['pm_icrs_masyr'][0],pm_n=r['pm_icrs_masyr'][1],se_e=r['pm_error_masyr'][0],se_n=r['pm_error_masyr'][1],pm_cov_en=cov[1,3],
            kind=r['kind'],magnitude=r['magnitude'],sdss_runs=info['sdss_physical_runs'],nsc_nights=info['nsc_nights'],nsc_seasons=info['nsc_seasons'],
            joint_span=info['joint_span_years'],nsc_span=info['nsc_span_years'],maximum_group_leverage=max(info['largest_information_fraction']['physical_group']),
            flags=';'.join(sorted(r['quality_flags'])),bands=';'.join(r['bands']),instruments=';'.join(r['instruments'])))
        claims.extend((d,ident) for d in ids)
        if key not in old_by_key:changed.append(dict(src=ident,member_sha256=key,detection_ids=ids))
    for slot,r in enumerate(older):
        if old_entry['moves'].get(slot,core) not in specs:continue
        key=member_key(r['detection_ids']);ident=(number<<32)+slot
        prior.append(dict(src=ident,member_sha256=key,n_members=len(r['detection_ids'])))
        if key not in new_keys:changed.append(dict(old_src=ident,member_sha256=key,detection_ids=r['detection_ids']))
    columns='src core slot owner member_sha256 n_members component reference_model_sha256 reference_epoch independent_pm_variance_e independent_pm_variance_n reference_pm_variance_e reference_pm_variance_n ra_deg dec_deg pixel128 pm_e pm_n se_e se_n pm_cov_en kind magnitude sdss_runs nsc_nights nsc_seasons joint_span nsc_span maximum_group_leverage flags bands instruments'.split()+[f'cov_{a}{b}' for a in range(4) for b in range(a,4)]
    pd.DataFrame(compact,columns=columns).to_parquet(dest/'science.parquet',index=False,compression='zstd')
    pd.DataFrame(claims,columns=['detection_id','src']).to_parquet(dest/'claims.parquet',index=False,compression='zstd')
    pd.DataFrame(prior,columns=['src','member_sha256','n_members']).to_parquet(dest/'old.parquet',index=False,compression='zstd')
    write_rows(dest/'private_changed_members.json.gz',changed)
    result=dict(ticket=ticket,core=core,origin_science=len(science),inside_union=sum(v is not None for v in ownership),
        transferred=sum(v is not None and v!=core for v in ownership),exterior=sum(v is None for v in ownership),old_retained=len(prior),
        files={name:digest(dest/name) for name in ('science.parquet','claims.parquet','old.parquet','private_changed_members.json.gz')},
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save_json(receipt,result);print('SCIENCE_INDEX',core,len(science),flush=True);return result


def database(root,memory='24GB',threads=8):
    import duckdb
    dest=root/'catalogue';dest.mkdir(exist_ok=True);db=duckdb.connect(str(dest/'catalogue.duckdb'))
    db.execute('SET memory_limit=?',[memory]);db.execute('SET threads=?',[threads]);db.execute('SET temp_directory=?',[str(dest/'spill')]);db.execute('SET preserve_insertion_order=false')
    return db


def close(root):
    dest=root/'catalogue';plan=read(root/'registration/private_full_coordinate_plan.json');proofs=[]
    for e in plan['cores']:
        folder=dest/'index'/e['core'];r=read(folder/'private_index_receipt.json')
        for name,checksum in r['files'].items():
            if digest(folder/name)!=checksum:raise ValueError('Changed completed scientific index')
        proofs.append(digest(folder/'private_index_receipt.json'))
    ticket=dict(plan_sha256=digest(root/'registration/private_full_coordinate_plan.json'),input_signatures=proofs,implementation_sha256=digest(Path(__file__)))
    lock=dest/'private_closure_binding.json'
    if lock.exists() and read(lock)!=ticket:raise ValueError('Changed global closure input')
    save_json(lock,ticket);db=database(root)
    for name,file in [('science','science.parquet'),('claims','claims.parquet'),('old','old.parquet')]:
        glob=str(dest/'index/*'/file);db.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{glob}',union_by_name=true)")
    buckets=dest/'claim_buckets'
    marker=dest/'private_claim_partition.json'
    if not marker.exists():
        if buckets.exists():raise ValueError('Incomplete claim partition requires explicit independent recovery directory')
        db.execute(f"COPY (SELECT *,hash(detection_id)%64 bucket FROM claims) TO '{buckets}' (FORMAT PARQUET,PARTITION_BY(bucket),COMPRESSION ZSTD)")
        save_json(marker,dict(binding_sha256=digest(lock),files={str(p.relative_to(dest)):digest(p) for p in buckets.glob('**/*.parquet')}))
    previous=read(marker)
    if previous['binding_sha256']!=digest(lock):raise ValueError('Changed global claim partition')
    for name,checksum in previous['files'].items():
        if digest(dest/name)!=checksum:raise ValueError('Changed global detection claim')
    outputs=[]
    for folder in sorted(buckets.glob('bucket=*')):
        out=dest/('private_shared_'+folder.name.split('=')[1]+'.parquet');outputs.append(out)
        proof=out.with_suffix('.json')
        if proof.exists():
            prior=read(proof)
            if prior['binding_sha256']!=digest(lock) or prior['sha256']!=digest(out):raise ValueError('Changed completed source-claim bucket')
            continue
        temp=out.with_suffix('.pending.parquet')
        glob=str(folder/'*.parquet')
        db.execute(f'''COPY (SELECT * EXCLUDE(bucket) FROM read_parquet('{glob}',hive_partitioning=true)
            SEMI JOIN (SELECT detection_id FROM read_parquet('{glob}',hive_partitioning=true) GROUP BY detection_id HAVING count(*)>1) USING(detection_id)) TO '{temp}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
        temp.replace(out);save_json(proof,dict(binding_sha256=digest(lock),sha256=digest(out)))
        print('SCIENCE_CLAIM_BUCKET',folder.name,flush=True)
    glob=str(dest/'private_shared_*.parquet');db.execute(f"CREATE OR REPLACE VIEW shared AS SELECT * FROM read_parquet('{glob}')")
    db.execute('CREATE OR REPLACE TABLE competing AS SELECT DISTINCT src FROM shared')
    db.execute('CREATE OR REPLACE TABLE accepted AS SELECT s.* FROM science s ANTI JOIN competing USING(src) WHERE owner IS NOT NULL')
    # An identity ambiguity is archived explicitly; no detection is silently
    # offered to two high-confidence sources, and no QSO value selects a winner.
    db.execute('CREATE OR REPLACE TABLE exact_common AS SELECT a.src new_src,o.src old_src,a.member_sha256 FROM accepted a JOIN old o USING(member_sha256)')
    duplicates=db.execute('SELECT count(*) FROM (SELECT member_sha256 FROM accepted GROUP BY member_sha256 HAVING count(*)>1)').fetchone()[0]
    if duplicates:raise ValueError('Duplicate accepted physical source')
    stats=db.execute('SELECT count(*),count(DISTINCT pixel128) FROM accepted').fetchone()
    by_class=dict(db.execute('SELECT kind,count(*) FROM accepted GROUP BY kind').fetchall())
    by_core=db.execute('SELECT owner,count(*) FROM accepted GROUP BY owner').fetchall()
    result=dict(origin_research_solutions=sum(read(dest/'index'/e['core']/'private_index_receipt.json')['origin_science'] for e in plan['cores']),
        retained_common_epoch_solutions=stats[0],occupied_nside128_pixels=stats[1],by_class=by_class,
        nominal_registered_area_sqdeg=sum(area(e['spec']) for e in plan['cores']),
        exterior_archived=db.execute('SELECT count(*) FROM science WHERE owner IS NULL').fetchone()[0],
        competing_source_quarantine=db.execute('SELECT count(*) FROM competing').fetchone()[0],
        shared_detection_claims=db.execute('SELECT count(*) FROM shared').fetchone()[0],
        exact_physical_members_common=db.execute('SELECT count(*) FROM exact_common').fetchone()[0],
        old_retained_solutions=db.execute('SELECT count(*) FROM old').fetchone()[0],
        private_input_binding_sha256=digest(lock),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Full-source native ICRS/J2015.5 catalogue. Owner remapping leaves state/covariance/members unchanged. Exterior archived; any shared detection quarantines competing raw solutions pending explicit local audit. Occupied HEALPix pixels are not effective or certified area. Exact membership does not certify a true star; changed-member continuity follows separately.')
    if 'se_e' in {r[0] for r in db.execute('DESCRIBE accepted').fetchall()}:
        spatial=db.execute('''SELECT pixel128,count(*) source_count,median(se_e) median_SE_e,median(se_n) median_SE_n,
          avg(CASE WHEN maximum_group_leverage>=.8 THEN 1.0 ELSE 0.0 END) high_leverage_fraction,
          median(nsc_nights) median_nsc_nights,median(joint_span) median_joint_span,
          count(*) FILTER(WHERE kind='r_existing') r_existing,
          count(*) FILTER(WHERE kind='extra_r_anchor') extra_r_anchor,
          count(*) FILTER(WHERE kind='non_r_anchor') non_r_anchor
          FROM accepted GROUP BY pixel128''').fetchdf().to_dict('records')
        public=dict(nside=128,spatial=spatial,scope='Source-occupied pixels and conditional error/support diagnostics; neither geometric effective area nor certified spatial zero area.')
        check_aggregate(public);save_json(dest/'v22_catalogue_spatial_support_summary.json',public)
    check_aggregate(result);save_json(dest/'v22_catalogue_ownership_summary.json',result)
    save_json(dest/'private_owned_counts.json',dict(by_core));db.execute('CHECKPOINT');db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['index','close']);p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if a.mode=='close':close(a.root)
    else:
        plan=read(a.root/'registration/private_full_coordinate_plan.json');old=old_index(a.root.parent/'v1.9/batch/science')
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for done in pool.map(index_core,[(i,e,str(a.root),old) for i,e in enumerate(plan['cores'])]):pass
