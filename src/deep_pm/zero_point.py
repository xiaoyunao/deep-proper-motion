"""Relative visit offsets estimated from stationary reference objects."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr


def fit_offsets(refs, xy, sigma, group, min_refs, draws, seed, workers=1,
                parents=None, shrinkage_mas=50.):
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
        return {}
    selected = max(candidates)[1]
    active = active.loc[component[r] == selected]
    r,g,sources,groups,child,parent = index(active)
    nr, ng = len(sources), len(groups)
    i = np.arange(len(g))
    design = coo_matrix((np.ones(len(g)*2), (np.r_[i, i], np.r_[r, nr + g])),
                        shape=(len(g), nr + ng)).tocsr()[:, :-1]
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

    def solve(weights):
        coeff = []
        for axis in range(2):
            w = np.sqrt(weights) / independent[:, axis]
            matrix, rhs = design.multiply(w[:, None]).tocsr(), values[:,axis]*w
            if len(child):
                matrix, rhs = vstack((matrix,penalty),format="csr"),np.r_[rhs,np.zeros(len(child))]
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

    weight = np.ones(len(g))
    for _ in range(6):
        optimum = solve(weight)
        residual = (values - design @ optimum) / independent
        weight = np.minimum(1, 6 / (4 + (residual**2).sum(axis=1)))
    random = np.random.default_rng(seed)
    # Bayesian bootstrap by reference object preserves shared visit covariance
    # and the connected graph, including groups with relatively few references.
    reference_weights = random.exponential(size=(draws, nr))

    def draw(ref_weight):
        return solve(weight * ref_weight[r])[nr:]

    if workers>1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = np.asarray(list(pool.map(draw, reference_weights)))
    else:
        samples = np.asarray(list(map(draw, reference_weights)))
    support = active.groupby("group").ref.nunique()
    positions = np.r_[optimum[nr:], np.zeros((1, 2))]
    samples = np.concatenate((samples, np.zeros((draws, 1, 2))), axis=1)
    return {name: (positions[j], samples[:, j], int(support.get(name, 0)))
            for j, name in enumerate(groups)}
