"""Bound provisional connectivity using only signed reference training measurements."""
from pathlib import Path
import argparse
import os,json
import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from deep_pm.io import digest,save_json

def run(work):
    old=work.parent/'v2.2/graph'
    index=pd.read_parquet(old/'private_group_components.parquet')
    markers=sorted((work/'reference-packs').glob('new-*/*/private_pack.json'))
    files=[]
    for marker in markers:
     r=json.loads(marker.read_text())
     files.extend(str(marker.parent/name) for name in r['files'])
    if len({p.parents[1].name for p in markers})!=len(markers):
     raise ValueError('Mixed pack versions need an explicit snapshot')
    dest=work/'chart-witness'/os.environ['SLURM_JOB_ID'];dest.mkdir(parents=True,exist_ok=False)
    db=duckdb.connect(str(dest/'sample.duckdb'));db.execute("SET memory_limit='4GB'");db.execute("SET threads=3")
    save_json(dest/'private_pack_bindings.json',{str(p.relative_to(work)):digest(p) for p in markers})
    db.from_parquet(files).create_view('snapshot')
    db.execute("""CREATE TABLE candidate AS SELECT DISTINCT reference,"group"
     FROM snapshot WHERE NOT is_holdout""")
    db.execute("""CREATE TABLE robust_group AS SELECT "group" FROM candidate GROUP BY "group" HAVING count(*)>=10""")
    db.execute("""CREATE TABLE eligible AS SELECT c.* FROM candidate c SEMI JOIN robust_group USING("group")""")
    groups=sorted(set(index['group'])|{s[0] for s in db.execute('SELECT DISTINCT "group" FROM eligible').fetchall()})
    ids={v:i for i,v in enumerate(groups)}
    db.register('names',pd.DataFrame({'group':groups,'gid':np.arange(len(groups),dtype=np.int32)}))
    edges=db.execute('''WITH represented AS (SELECT e.reference,n.gid FROM eligible e JOIN names n USING("group")),
     base AS (SELECT reference,min(gid) start_gid FROM represented GROUP BY reference HAVING count(*)>=2)
     SELECT DISTINCT start_gid,gid FROM represented JOIN base USING(reference)''').fetchnumpy()
    ii=[edges['start_gid']];jj=[edges['gid']]
    for comp,part in index.groupby('component'):
     resolved=part['group'].map(ids).to_numpy();ii.append(np.repeat(resolved[0],len(resolved)));jj.append(resolved)
    ii=np.concatenate(ii);jj=np.concatenate(jj)
    _,labels=connected_components(coo_matrix((np.ones(len(ii),dtype=np.int8),(ii,jj)),shape=(len(ids),len(ids))),directed=False)
    mapping={str(comp):int(labels[ids[part['group'].iloc[0]]]) for comp,part in index.groupby('component')}
    counts=np.bincount(labels)
    result=dict(signed_packs=len(markers),signed_pack_rows=sum(json.loads(m.read_text())['rows'] for m in markers),
                old_components_sharing_a_provisional_connection=[[int(k) for k in mapping if mapping[k]==c] for c in sorted(set(mapping.values()))],
                old_components_group_counts={k:int(counts[v]) for k,v in mapping.items()},
                added_group_names=len(groups)-len(index),largest_connected_physical_group_count=int(counts.max()),
                scope='Planning witness only. Existing physical components plus local training reference-group pairs with ten distinct references per group. '
                      'Global duplicate, exposure conflict, final peeling, full component geometry and gauge are still required.')
    save_json(dest/'v24_component_witness_summary.json',result);print(result,flush=True)
    db.close()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    args=parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or int(os.environ["SLURM_CPUS_PER_TASK"])<3:
        parser.error("Use a Slurm CPU allocation with at least three CPUs")
    run(args.work)
