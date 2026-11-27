"""Use actual predecessor directions to bound the angular extent of newly linked domains."""
import argparse
import json
import os
from pathlib import Path

import duckdb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .io import digest, save_json


def angular_bounds(sky):
    ra, dec = np.deg2rad(sky).T
    u = np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))
    _, opposite = cKDTree(u).query(-u)
    pairs = np.einsum("ij,ij->i", u, u[opposite])
    a = np.argmin(pairs)
    diameter = np.rad2deg(np.arccos(np.clip(pairs[a], -1., 1.)))
    # q is always a convex combination of actual directions. For ANY unit
    # centre c, min(c.u_i) <= c.q <= |q|. The bound remains valid if the
    # iterative solver stops early. A candidate centre supplies the other bound.
    q = .5*(u[a]+u[opposite[a]])
    best_centre = u[0].copy()
    best_min_dot = (u@best_centre).min()
    for _ in range(10000):
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            break
        dots = u@q
        i = np.argmin(dots)
        feasible = dots[i]/norm
        if feasible > best_min_dot:
            best_min_dot = feasible
            best_centre = q/norm
        if norm-feasible < 1e-6:
            break
        direction = u[i]-q
        step = np.clip(-np.dot(q,direction)/max(1e-30,np.dot(direction,direction)),0.,1.)
        q = q + step*direction
    lower = max(diameter/2, np.rad2deg(np.arccos(np.clip(np.linalg.norm(q),0.,1.))))
    upper = np.rad2deg(np.arccos(np.clip(best_min_dot,-1.,1.)))
    return dict(actual_pair_separation_deg=float(diameter), lower_bound_deg=float(lower),
                sample_upper_bound_deg=float(upper), candidate_center_deg=[
                    float(np.degrees(np.arctan2(best_centre[1],best_centre[0]))%360),
                    float(np.degrees(np.arcsin(np.clip(best_centre[2],-1.,1.))))])


def supported_links(db, old_index):
    import pandas as pd
    old = pd.read_parquet(old_index, columns=["group","component"])
    old["node"] = ["old-"+str(int(c)) for c in old.component]
    db.register("oldkey", old[["group","node"]])
    # A new edge is counted once per published reference identity. Requiring
    # many identities per edge tests whether a single provisional link dominates.
    edges = db.execute('''WITH membership AS (
      SELECT DISTINCT reference,coalesce(node,"group") AS node FROM eligible LEFT JOIN oldkey USING("group")),
      roots AS (SELECT reference,min(node) AS src FROM membership GROUP BY reference HAVING count(*)>1)
      SELECT src,node AS dst,count(*) AS witnesses FROM roots JOIN membership USING(reference)
      WHERE src<>node GROUP BY src,node''').fetchdf()
    nodes = sorted(set(edges.src)|set(edges.dst)|set(old.node))
    index = {n:i for i,n in enumerate(nodes)}
    result = {}
    for n in (1,10):
        used = edges[edges.witnesses>=n]
        r = used.src.map(index).to_numpy(dtype=int)
        c = used.dst.map(index).to_numpy(dtype=int)
        links = coo_matrix((np.ones(len(r),dtype=np.int8),(r,c)),shape=(len(nodes),len(nodes)))
        _, labels = connected_components(links, directed=False)
        groups = {}
        for k in sorted(set(old.component)):
            groups.setdefault(int(labels[index["old-"+str(int(k))]]), []).append(int(k))
        result[str(n)] = list(groups.values())
    return result


def run(work, partial):
    old = work.parent/"v2.2/graph"
    witness = partial/"v24_component_witness_summary.json"
    previous = old/"v22_global_graph_summary.json"
    groups = json.loads(witness.read_text())["old_components_sharing_a_provisional_connection"]
    partialdb = duckdb.connect(str(partial/"sample.duckdb"),read_only=True)
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    partialdb.execute("SET threads = ?",[cpus])
    partialdb.execute("SET memory_limit='6GB'")
    robust = supported_links(partialdb,old/"private_group_components.parquet")
    partialdb.close()
    db = duckdb.connect(str(old/"reference.duckdb"),read_only=True)
    db.execute("SET threads = ?",[cpus])
    db.execute("SET memory_limit='6GB'")
    output = []
    for comps in groups:
        if len(comps)<2:
            continue
        ids = ",".join(str(int(c)) for c in comps)
        # One actual pair per occupied half-degree cell. Taking a subset is
        # conservative for an impossibility result, and the full rows test any proposed centre.
        sample = db.execute(f'''WITH cells AS (
          SELECT min(struct_pack(a:=ra_deg,d:=dec_deg)) AS p FROM active JOIN group_map USING("group")
          WHERE component IN ({ids}) GROUP BY round(ra_deg*2),round(dec_deg*2))
          SELECT p.a,p.d FROM cells ORDER BY p.a,p.d''').fetchnumpy()
        angular = angular_bounds(np.column_stack(list(sample.values())))
        ra,dec = angular["candidate_center_deg"]
        n,theta = db.execute(f'''SELECT count(*),degrees(acos(least(1.,greatest(-1.,min(
          sin(radians(dec_deg))*sin(radians({dec}))+cos(radians(dec_deg))*cos(radians({dec}))*cos(radians(ra_deg-{ra})))))))
          FROM active JOIN group_map USING("group") WHERE component IN ({ids})''').fetchone()
        output.append(dict(predecessors=comps, actual_training_rows=n, independent_sky_cells=len(next(iter(sample.values()))),
                           candidate_maximum_on_all_old_rows_deg=theta, **angular))
    db.close()
    save_json(partial/"v24_linked_chart_geometry_summary.json",dict(slurm_job=os.environ["SLURM_JOB_ID"],
        provisional_binding_sha256=digest(witness),predecessor_graph_sha256=digest(previous),
        old_group_connectivity_by_minimum_reference_identities_per_edge=robust,angular_tests=output,
        scope="Provisional signed inputs, not the final globally deduplicated and peeled graph. Geometry uses actual predecessor training positions. A lower bound above 85 degrees rules out every single 85-degree chart for those positions."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("partial",type=Path)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Run on an allocated Slurm node")
    run(a.work,a.partial)
