"""External-memory global reference deduplication, peeling and graph closure."""
import argparse
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .blind_catalogue_quality import read
from .coordinate_metric import projected_covariance,native_covariance
from .astrometry import project
from .io import digest,save_json
from .public_summary import check_aggregate


def connect(path,memory,threads):
    import duckdb
    connection=duckdb.connect(str(path/'reference.duckdb'))
    connection.execute('SET memory_limit = ?', [memory])
    connection.execute('SET threads = ?', [threads])
    connection.execute('SET temp_directory = ?', [str(path/'spill')])
    connection.execute('SET preserve_insertion_order = false')
    return connection


def scalar(db,query):return db.execute(query).fetchone()[0]


def close_graph(root,memory='24GB',threads=8):
    start=monotonic();dest=root/'graph';dest.mkdir(parents=True,exist_ok=True)
    plan_path=root/'registration/private_full_coordinate_plan.json';plan=read(plan_path)
    proofs=[]
    for entry in plan['cores']:
        path=root/'input'/entry['core'];proof=read(path/'v22_reference_input_core_summary.json')
        if proof['parquet_sha256']!=digest(path/'reference.parquet'):raise ValueError('Changed input Parquet')
        proofs.append(proof['parquet_sha256'])
    ticket=dict(plan_sha256=digest(plan_path),shards=proofs,implementation_sha256=digest(Path(__file__)))
    lock=dest/'private_input_binding.json'
    if lock.exists() and read(lock)!=ticket:raise ValueError('Global graph input changed; use a new version directory')
    save_json(lock,ticket)
    db=connect(dest,memory,threads)
    def stage(name,sql):
        if scalar(db,f"SELECT count(*) FROM information_schema.tables WHERE table_name='{name}'")==0:
            db.execute(sql)
            db.execute('CHECKPOINT')
        print('GRAPH_STAGE',name,scalar(db,f'SELECT count(*) FROM {name}'),flush=True)
    glob=str(root/'input/*/reference.parquet')
    # Conflict policy preserves archived semantics. Exact halo duplicates are
    # removed before distinct physical exposure collisions are excluded.
    stage('bad_detection',f'''CREATE TABLE bad_detection AS SELECT detection_id FROM read_parquet('{glob}')
        GROUP BY detection_id HAVING count(DISTINCT reference)>1''')
    stage('unique_detection',f'''CREATE TABLE unique_detection AS
        SELECT * EXCLUDE(rn) FROM (
          SELECT *,row_number() OVER(PARTITION BY detection_id ORDER BY source_core) rn
          FROM read_parquet('{glob}') ANTI JOIN bad_detection USING(detection_id)) WHERE rn=1''')
    # Check the native input, not the obsolete per-domain projection coordinates.
    native_conflict=scalar(db,f'''SELECT count(*) FROM (
        SELECT detection_id FROM read_parquet('{glob}') GROUP BY detection_id
        HAVING count(DISTINCT "group")>1 OR max(ra_deg)-min(ra_deg)>1e-8
        OR max(dec_deg)-min(dec_deg)>1e-8 OR max(sigma_e)-min(sigma_e)>1e-5
        OR max(sigma_n)-min(sigma_n)>1e-5 OR count(DISTINCT is_holdout)>1)''')
    if native_conflict:raise ValueError(f'Conflicting physical measurement/split: {native_conflict}')
    stage('bad_exposure','''CREATE TABLE bad_exposure AS SELECT reference,exposure_id FROM unique_detection
        GROUP BY reference,exposure_id HAVING count(*)>1''')
    stage('clean','''CREATE TABLE clean AS SELECT u.* FROM unique_detection u
        ANTI JOIN bad_exposure b USING(reference,exposure_id)''')
    split_clash=scalar(db,'SELECT count(*) FROM (SELECT reference FROM clean GROUP BY reference HAVING count(DISTINCT is_holdout)>1)')
    if split_clash:raise ValueError('Physical reference appears in training and holdout')
    stage('active','CREATE TABLE active AS SELECT * FROM clean WHERE NOT is_holdout')
    trace=[]
    while True:
        n=scalar(db,'SELECT count(*) FROM active')
        db.execute('''CREATE OR REPLACE TABLE peel_groups AS SELECT "group" FROM active
            GROUP BY "group" HAVING count(DISTINCT reference)>=10''')
        db.execute('''CREATE OR REPLACE TABLE peel_refs AS SELECT reference FROM active
            SEMI JOIN peel_groups USING("group") GROUP BY reference HAVING count(DISTINCT "group")>=2''')
        db.execute('''CREATE OR REPLACE TABLE next_active AS SELECT a.* FROM active a
            SEMI JOIN peel_groups USING("group") SEMI JOIN peel_refs USING(reference)''')
        m=scalar(db,'SELECT count(*) FROM next_active');trace.append(dict(before=n,after=m))
        db.execute('DROP TABLE active');db.execute('ALTER TABLE next_active RENAME TO active');db.execute('CHECKPOINT')
        print('GRAPH_PEEL',n,m,flush=True)
        if n==m:break
    names=db.execute('SELECT DISTINCT "group" FROM active ORDER BY "group"').fetchnumpy()['group']
    group_table=pd.DataFrame({'group':names,'gid':np.arange(len(names),dtype='int32')})
    db.register('group_map_frame',group_table)
    db.execute('CREATE OR REPLACE TABLE group_map AS SELECT * FROM group_map_frame')
    # A star spanning tree per reference is enough for exact connectivity; no
    # quadratic all-pairs galaxy expansion and no component-local guessed zeros.
    edge=db.execute('''WITH base AS (SELECT reference,min(gid) root FROM active JOIN group_map USING("group") GROUP BY reference)
        SELECT DISTINCT root,gid FROM active JOIN base USING(reference) JOIN group_map USING("group")''').fetchnumpy()
    ncomponents,labels=connected_components(coo_matrix((np.ones(len(edge['gid']),dtype='int8'),(edge['root'],edge['gid'])),shape=(len(names),len(names))),directed=False)
    group_table['component']=labels
    db.unregister('group_map_frame');db.register('group_map_frame',group_table)
    db.execute('CREATE OR REPLACE TABLE group_map AS SELECT * FROM group_map_frame')
    components=[]
    for component in range(ncomponents):
        member=group_table.loc[group_table.component.eq(component),'group'].tolist()
        surveys={g.split(':')[0] for g in member}
        if surveys!={'NSC','SDSS'}:
            components.append(dict(component=component,status='no_cross_survey_link',groups=len(member)));continue
        statistics=db.execute('''SELECT count(*),count(DISTINCT reference),sum(cos(radians(dec_deg))*cos(radians(ra_deg))),
            sum(cos(radians(dec_deg))*sin(radians(ra_deg))),sum(sin(radians(dec_deg)))
            FROM active JOIN group_map USING("group") WHERE component=?''',[component]).fetchone()
        count,nref,x,y,z=statistics
        # One explicitly measured center per new component, bound in its output.
        center=[float(np.degrees(np.arctan2(y,x))%360),float(np.degrees(np.arctan2(z,np.hypot(x,y))))]
        theta=scalar(db,f'''SELECT degrees(acos(least(1.0,greatest(-1.0,min(dot))))) FROM (
            SELECT sin(radians(dec_deg))*sin(radians({center[1]}))+
            cos(radians(dec_deg))*cos(radians({center[1]}))*cos(radians(ra_deg-{center[0]})) dot
            FROM active JOIN group_map USING("group") WHERE component={component})''')
        if theta>=85:raise ValueError('New component exceeds registered stable gnomonic hemisphere; choose an explicit common projection strategy')
        components.append(dict(component=component,status='ready',groups=len(member),reference_sources=nref,measurements=count,
            center_deg=center,maximum_theta_deg=theta,nsc_gauge_group_count=sum(g.startswith('NSC:') for g in member)))
    group_table.to_parquet(dest/'private_group_components.parquet',index=False)
    summary=dict(plan_sha256=digest(plan_path),input_shard_measurements=sum(read(root/'input'/e['core']/'v22_reference_input_core_summary.json')['measurements'] for e in plan['cores']),
        conflict_detection_count=scalar(db,'SELECT count(*) FROM bad_detection'),unique_measurements=scalar(db,'SELECT count(*) FROM unique_detection'),
        physical_exposure_conflicts=scalar(db,'SELECT count(*) FROM bad_exposure'),clean_measurements=scalar(db,'SELECT count(*) FROM clean'),
        clean_holdout_measurements=scalar(db,'SELECT count(*) FROM clean WHERE is_holdout'),
        training_measurements=scalar(db,'SELECT count(*) FROM active'),peeling=trace,components=components,
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],support_definition='Graph peeling is pre-SE. No group-level planar SE rejection. Native12mas support evaluated at each target measurement.')
    check_aggregate(summary);save_json(dest/'v22_global_graph_summary.json',summary)
    db.execute('CHECKPOINT');db.close()


def export_component(root,number,memory='24GB',threads=8):
    dest=root/'reference'/f'component-{number:04d}';dest.mkdir(parents=True,exist_ok=True)
    binding=root/'graph/v22_global_graph_summary.json';meta=read(binding)
    entry=next(c for c in meta['components'] if c['component']==number)
    if entry['status']!='ready':raise ValueError('Not a cross-survey component')
    db=connect(root/'graph',memory,threads)
    names=db.execute('SELECT "group" FROM group_map WHERE component=? ORDER BY "group"',[number]).fetchnumpy()['group']
    db.register('local_groups_frame',pd.DataFrame({'group':names,'lg':np.arange(len(names),dtype='int32')}))
    db.execute('CREATE OR REPLACE TEMP TABLE local_refs AS SELECT reference,row_number() OVER(ORDER BY reference)-1 rid FROM active SEMI JOIN local_groups_frame USING("group") GROUP BY reference')
    n=entry['measurements'];nr=entry['reference_sources']
    groups=np.lib.format.open_memmap(dest/'groups.npy',mode='w+',dtype='int32',shape=(n,))
    value=np.lib.format.open_memmap(dest/'values.npy',mode='w+',dtype='float64',shape=(n,2))
    precision=np.lib.format.open_memmap(dest/'precision.npy',mode='w+',dtype='float64',shape=(n,3))
    # Source-centering prevents billion-mas absolute coordinates from polluting
    # normal RHS cancellation. Every bootstrap keeps this same translation.
    cursor=db.execute('''SELECT rid,lg,ra_deg,dec_deg,sigma_e,sigma_n FROM active
        JOIN local_groups_frame USING("group") JOIN local_refs USING(reference) ORDER BY rid,lg''')
    reader=cursor.fetch_record_batch(rows_per_batch=100000) if hasattr(cursor,'fetch_record_batch') else cursor.to_arrow_reader(batch_size=100000)
    counts=np.zeros(nr,dtype='int64');offset=0;last=-1;origin=None
    for batch in reader:
        f=batch.to_pandas();length=len(f);rid=f.rid.to_numpy();g=f.lg.to_numpy()
        xy=project(f.ra_deg,f.dec_deg,entry['center_deg'])
        first=np.r_[True,rid[1:]!=rid[:-1]];starts=np.flatnonzero(first)
        origins=xy[starts].copy()
        if rid[0]==last:origins[0]=origin
        xy-=origins[np.cumsum(first)-1]
        last=int(rid[-1]);origin=origins[-1].copy()
        inv=np.linalg.inv(projected_covariance(f.ra_deg,f.dec_deg,entry['center_deg'],native_covariance(f[['sigma_e','sigma_n']].to_numpy())))
        groups[offset:offset+length]=g;value[offset:offset+length]=xy;precision[offset:offset+length]=inv[:,[0,0,1],[0,1,1]]
        np.add.at(counts,rid,1);offset+=length
    if offset!=n or np.any(counts<2):raise ValueError('Numeric export row count or source linkage changed')
    ptr=np.r_[0,np.cumsum(counts)];np.save(dest/'ptr.npy',ptr)
    np.save(dest/'names.npy',np.asarray(names,dtype=str))
    for a in (groups,value,precision):a.flush()
    files={p.name:digest(p) for p in sorted(dest.glob('*.npy'))}
    save_json(dest/'private_numeric_binding.json',dict(graph_sha256=digest(binding),component=entry,files=files,
        precision='native covariance including50mas, projected and inverted full2D once',centering='per source first measurement'))
    db.close();print('COMPONENT_EXPORTED',number,n,nr,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['close','export']);p.add_argument('root',type=Path)
    p.add_argument('--component',type=int);p.add_argument('--memory',default='24GB');p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if a.mode=='close':close_graph(a.root,a.memory,a.threads)
    else:export_component(a.root,a.component,a.memory,a.threads)
