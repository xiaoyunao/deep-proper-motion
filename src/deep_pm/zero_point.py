"""Relative visit offsets estimated from stationary reference objects."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

from .io import digest


def read_checkpoint(path):
    checksum = path.with_suffix(".sha256")
    if not path.is_file() or not checksum.is_file():
        return None
    if digest(path) != checksum.read_text().strip():
        raise ValueError(f"Changed reference checkpoint {path.name}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def write_checkpoint(path, **arrays):
    pending = path.with_suffix(".part")
    with pending.open("wb") as stream:
        np.savez(stream, **arrays)
    pending.replace(path)
    checksum = path.with_suffix(".sha256")
    pending = checksum.with_suffix(".sha-part")
    pending.write_text(digest(path) + "\n")
    pending.replace(checksum)


def fit_offsets(refs, xy, sigma, group, min_refs, draws, seed, workers=1,
                parents=None, shrinkage_mas=50., overlaps=None, features=None,
                checkpoint=None):
    # Optional C0-only residual design: returns (offsets, coefficients, draws).
    # Default calls retain the original numerical solve and dict result.
    if features is not None and (parents or overlaps is not None):
        raise ValueError('Residual features are a separate C0 trial, not a C1 hierarchy')
    # Peel groups without independent references and objects without a link.
    pairs = pd.DataFrame(dict(ref=np.asarray(refs), group=group))
    active = pairs.copy()
    while len(active):
        count = active.groupby("group").ref.transform("nunique")
        reduced = active.loc[count >= min_refs]
        linked = reduced.groupby("ref").group.transform("nunique")
        reduced = reduced.loc[linked >= 2]
        if len(reduced) == len(active):
            break
        active = reduced
    if active.empty:
        if features is not None:
            raise ValueError('No reference support for a residual-feature model')
        return {}
    def index(rows):
        r, sources = pd.factorize(rows.ref)
        groups = list(pd.unique(rows.group))
        if parents:
            groups.extend(p for p in dict.fromkeys(parents[k] for k in groups if k in parents)
                          if p not in groups)
        lookup = {name:j for j,name in enumerate(groups)}
        g = rows.group.map(lookup).to_numpy()
        child = np.array([lookup[name] for name in groups if parents and name in parents],dtype=int)
        parent = np.array([lookup[parents[groups[j]]] for j in child],dtype=int)
        return r,g,sources,groups,child,parent

    r,g,sources,groups,child,parent = index(active)
    nr, ng = len(sources), len(groups)
    graph = coo_matrix((np.ones(len(g)+len(child)),
                        (np.r_[r,nr+child],nr+np.r_[g,parent])),shape=(nr+ng,nr+ng))
    ncomponents, component = connected_components(graph, directed=False)
    # Different connected components have different gauges. Only a shared
    # component can define a cross-survey proper motion.
    candidates = [(sum(component[:nr] == k), k) for k in range(ncomponents)
                  if len({str(groups[j]).split(":")[0]
                          for j in np.flatnonzero(component[nr:] == k)}) == 2]
    if not candidates:
        if features is not None:
            raise ValueError('No cross-survey reference component for residual features')
        return {}
    selected = max(candidates)[1]
    active = active.loc[component[r] == selected]
    r,g,sources,groups,child,parent = index(active)
    nr, ng = len(sources), len(groups)
    i = np.arange(len(g))
    design = coo_matrix((np.ones(len(g)*2), (np.r_[i, i], np.r_[r, nr + g])),
                        shape=(len(g), nr + ng)).tocsr()[:, :-1]
    nfeatures = 0
    if features is not None:
        full_features = np.asarray(features, float)
        if full_features.ndim != 2 or len(full_features) != len(refs) or not np.isfinite(full_features).all():
            raise ValueError('Features must be finite, aligned to the original reference rows')
        extra = full_features[active.index]
        nfeatures = extra.shape[1]
        # A band constant is already an exposure zero; a per-source constant is
        # already a galaxy position. Reject these degeneracies before fitting.
        norms = np.sqrt(np.asarray(design.power(2).sum(axis=0)).ravel())
        scale = np.divide(1., norms, out=np.ones_like(norms), where=norms>0)
        projection = design.multiply(scale).tocsr()
        orthogonal = []
        for column in extra.T:
            fitted = lsmr(projection, column, atol=1e-10, btol=1e-10, maxiter=4000)
            if fitted[1] not in (0,1,2,4,5):
                raise RuntimeError('Residual-feature rank check did not converge')
            residual = column-projection@fitted[0]
            if np.linalg.norm(residual) <= 1e-5*max(np.linalg.norm(column), 1.):
                raise ValueError('Residual feature is degenerate with source positions or group zeros')
            orthogonal.append(residual/np.linalg.norm(residual))
        if nfeatures and np.linalg.matrix_rank(np.column_stack(orthogonal), tol=1e-5) < nfeatures:
            raise ValueError('Residual features are mutually degenerate')
        design = hstack((design, coo_matrix(extra)), format='csr')
    if len(child):
        h = np.arange(len(child))
        penalty = coo_matrix((np.r_[np.ones(len(h)),-np.ones(len(h))]/shrinkage_mas,
                             (np.r_[h,h],nr+np.r_[child,parent])),
                             shape=(len(h),nr+ng)).tocsr()[:,:-1]
    values = xy[active.index].copy()
    # Center each object's positions to keep the numerical scale in mas.
    for j, indexes in active.groupby("ref", sort=False).indices.items():
        values[indexes] -= np.median(values[indexes], axis=0)
    independent = sigma[active.index]

    overlap_design = None
    if overlaps is not None and len(overlaps):
        names = {name: j for j, name in enumerate(groups)}
        source_ids = {name: j for j, name in enumerate(sources)}
        edges = overlaps.loc[overlaps.ref.isin(source_ids)
                             & overlaps["first"].isin(names) & overlaps.second.isin(names)]
        if len(edges):
            e = np.arange(len(edges))
            first, second = (edges[k].map(names).to_numpy() for k in ("first", "second"))
            overlap_ref = edges.ref.map(source_ids).to_numpy()
            overlap_design = coo_matrix(
                (np.r_[np.ones(len(e)), -np.ones(len(e))], (np.r_[e, e], nr + np.r_[first, second])),
                shape=(len(e), nr + ng)).tocsr()[:, :-1]
            overlap_value = edges[["east", "north"]].to_numpy()
            overlap_sigma = edges[["sigma_e", "sigma_n"]].to_numpy()
            overlap_base = edges.weight.to_numpy()
    overlap_weight = overlap_base.copy() if overlap_design is not None else None

    def solve(weights, edge_weights):
        coeff = []
        for axis in range(2):
            w = np.sqrt(weights) / independent[:, axis]
            matrix, rhs = design.multiply(w[:, None]).tocsr(), values[:,axis]*w
            if len(child):
                matrix, rhs = vstack((matrix,penalty),format="csr"),np.r_[rhs,np.zeros(len(child))]
            if overlap_design is not None:
                w_edge = np.sqrt(edge_weights) / overlap_sigma[:, axis]
                matrix = vstack((matrix, overlap_design.multiply(w_edge[:, None])), format="csr")
                rhs = np.r_[rhs, overlap_value[:, axis] * w_edge]
            # Balance columns with very different reference counts. This is
            # an invertible reparameterization of the same weighted graph.
            norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel())
            scale = np.divide(1., norms, out=np.ones_like(norms), where=norms > 0)
            matrix.data *= scale[matrix.indices]
            answer = lsmr(matrix,rhs,
                          atol=1e-9, btol=1e-9, maxiter=4000)
            if answer[1] not in (0, 1, 2, 4, 5):
                raise RuntimeError("Stationary reference solve did not converge")
            coeff.append(scale*answer[0])
        return np.column_stack(coeff)

    state = None
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        state = read_checkpoint(checkpoint/"robust.npz")
    if state is None:
        weight = np.ones(len(g))
        for _ in range(6):
            optimum = solve(weight, overlap_weight)
            residual = (values - design @ optimum) / independent
            weight = np.minimum(1, 6 / (4 + (residual**2).sum(axis=1)))
            if overlap_design is not None:
                residual = (overlap_value - overlap_design @ optimum) / overlap_sigma
                overlap_weight = overlap_base * np.minimum(1, 6 / (4 + (residual**2).sum(axis=1)))
        if checkpoint is not None:
            state = dict(optimum=optimum, weight=weight)
            if overlap_weight is not None:
                state["overlap_weight"] = overlap_weight
            write_checkpoint(checkpoint/"robust.npz", **state)
    else:
        optimum, weight = state["optimum"], state["weight"]
        overlap_weight = state.get("overlap_weight")
        if (optimum.shape != (design.shape[1], 2) or weight.shape != (len(g),)
                or (overlap_design is None) != (overlap_weight is None)
                or (overlap_design is not None and overlap_weight.shape != overlap_base.shape)):
            raise ValueError("Reference checkpoint does not match the fitted graph")
        print("Reusing robust reference fit", flush=True)
    random = np.random.default_rng(seed)
    # Bayesian bootstrap by reference object preserves shared visit covariance
    # and the connected graph, including groups with relatively few references.
    reference_weights = random.exponential(size=(draws, nr))

    def draw(item):
        number, ref_weight = item
        if checkpoint is not None:
            path = checkpoint/f"draw-{number:03d}.npz"
            saved = read_checkpoint(path)
            if saved is not None:
                result = saved["offsets"]
                if result.shape != (ng - 1 + nfeatures, 2) or not np.isfinite(result).all():
                    raise ValueError("Reference draw checkpoint has an invalid shape or value")
                return result
        edge_weights = None if overlap_weight is None else overlap_weight * ref_weight[overlap_ref]
        result = solve(weight * ref_weight[r], edge_weights)[nr:].copy()
        if checkpoint is not None:
            write_checkpoint(path, offsets=result)
            print(f"Completed reference bootstrap draw {number+1}/{draws}", flush=True)
        return result

    if workers>1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = np.asarray(list(pool.map(draw, enumerate(reference_weights))))
    else:
        samples = np.asarray(list(map(draw, enumerate(reference_weights))))
    support = active.groupby("group").ref.nunique()
    positions = np.r_[optimum[nr:nr+ng-1], np.zeros((1, 2))]
    feature_point = optimum[nr+ng-1:]
    feature_samples = samples[:,ng-1:]
    samples = np.concatenate((samples[:,:ng-1], np.zeros((draws, 1, 2))), axis=1)
    offsets = {name: (positions[j], samples[:, j], int(support.get(name, 0)))
               for j, name in enumerate(groups)}
    return (offsets, feature_point, feature_samples) if features is not None else offsets
