"""Relative visit offsets estimated from stationary reference objects."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr


def fit_offsets(refs, xy, sigma, group, min_refs, draws, seed, workers=1):
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
    r, sources = pd.factorize(active.ref)
    g, groups = pd.factorize(active.group)
    nr, ng = len(sources), len(groups)
    graph = coo_matrix((np.ones(len(g)), (r, nr + g)), shape=(nr + ng, nr + ng))
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
    r, sources = pd.factorize(active.ref)
    g, groups = pd.factorize(active.group)
    nr, ng = len(sources), len(groups)
    i = np.arange(len(g))
    design = coo_matrix((np.ones(len(g)*2), (np.r_[i, i], np.r_[r, nr + g])),
                        shape=(len(g), nr + ng)).tocsr()[:, :-1]
    values = xy[active.index].copy()
    # Center each object's positions to keep the numerical scale in mas.
    for j, indexes in active.groupby("ref", sort=False).indices.items():
        values[indexes] -= np.median(values[indexes], axis=0)
    independent = sigma[active.index]

    def solve(weights):
        coeff = []
        for axis in range(2):
            w = np.sqrt(weights) / independent[:, axis]
            answer = lsmr(design.multiply(w[:, None]), values[:, axis] * w,
                          atol=1e-9, btol=1e-9, maxiter=4000)
            if answer[1] not in (0, 1, 2, 4, 5):
                raise RuntimeError("Stationary reference solve did not converge")
            coeff.append(answer[0])
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
    return {name: (np.r_[optimum[nr:], np.zeros((1, 2))][j],
                   np.concatenate((samples, np.zeros((draws, 1, 2))), axis=1)[:, j],
                   int(support[name])) for j, name in enumerate(groups)}
