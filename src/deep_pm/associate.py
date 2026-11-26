"""Static and motion-aware association on the same detections."""
from collections import defaultdict
from dataclasses import dataclass, replace
from time import monotonic

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .astrometry import fit_motion, seasons


@dataclass
class Track:
    indices: tuple
    score: float
    status: str
    groups: int


def one_per_exposure(indices, distance, exposure):
    kept = {}
    for index, _ in sorted(zip(indices, distance), key=lambda item: (item[1], item[0])):
        kept.setdefault(exposure[index], index)
    return tuple(sorted(kept.values()))


def static_tracks(xy, exposure, radius, eps=None, sky_center=None):
    if not len(xy):
        return []
    if sky_center is not None and eps is not None:raise ValueError('Spherical fixed-radius route does not silently reinterpret planar DBSCAN epsilon')
    if sky_center is None:
        distance=lambda points,center:np.linalg.norm(points-center,axis=-1)
        tree_type=cKDTree
    else:
        from .coordinate_metric import AngularTree,angular_distance
        distance=lambda points,center:angular_distance(points,center,sky_center)
        tree_type=lambda points:AngularTree(points,sky_center)
    if eps is None:
        labels = np.full(len(xy), -1)
    else:
        labels = DBSCAN(eps=eps, min_samples=3).fit_predict(xy)
    groups = []
    for label in np.unique(labels[labels >= 0]):
        group = np.flatnonzero(labels == label)
        for index in np.setdiff1d(group, one_per_exposure(
                group, np.linalg.norm(xy[group] - np.median(xy[group], axis=0), axis=1), exposure)):
            labels[index] = -1
        group = np.flatnonzero(labels == label)
        groups.append(list(group))
    used = labels >= 0
    tree = tree_type(xy)
    centers = [np.median(xy[group], axis=0) for group in groups]
    if centers:
        center_tree = tree_type(centers)
        for index in np.flatnonzero(~used):
            possibilities = center_tree.query_ball_point(xy[index], radius)
            for label in sorted(possibilities, key=lambda j: float(distance(xy[index],centers[j])[0]) if sky_center is not None else float(distance(xy[index],centers[j]))):
                if exposure[index] not in exposure[groups[label]]:
                    groups[label].append(index)
                    used[index] = True
                    break
    for index in np.flatnonzero(~used):
        if used[index]:
            continue
        candidates = np.array(tree.query_ball_point(xy[index], radius) if sky_center is None else tree.query_ball_index(index, radius), dtype=int)
        candidates = candidates[~used[candidates]]
        separation = distance(xy[candidates],xy[index]) if sky_center is None else tree.distances_to_index(candidates,index)
        chosen = one_per_exposure(candidates, separation, exposure)
        used[list(chosen)] = True
        groups.append(chosen)
    return [tuple(sorted(group)) for group in groups]


def components(candidates):
    members = defaultdict(list)
    for i, track in enumerate(candidates):
        for index in track.indices:
            members[index].append(i)
    seen = set()
    for i in range(len(candidates)):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        group = []
        visited_detections = set()
        while stack:
            j = stack.pop()
            group.append(j)
            for index in candidates[j].indices:
                if index in visited_detections:
                    continue
                visited_detections.add(index)
                for other in members[index]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
        yield group, visited_detections


def choose_tracks(candidates, max_component=None, time_limit=5, audit=None):
    """Solve each conflict component and retain validated time-limit incumbents.

    The positional size argument is retained for callers of the prototype.
    Component size is diagnostic. The solver budget is a wall-time limit.
    """
    selected = []
    difficult = 0
    for group, detections in components(candidates):
        start = monotonic()
        local = [candidates[i] for i in group]
        scores = np.array([track.score for track in local])
        gap = None
        solver_status = 0
        if len(local) == 1:
            winners = np.flatnonzero(scores > 0)
            state = "optimal"
        else:
            row = {i: j for j, i in enumerate(sorted(detections))}
            entries = [(row[i], j) for j, track in enumerate(local) for i in track.indices]
            rows, columns = np.array(entries, dtype=np.int32).T.copy()
            matrix = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(row), len(local))).tocsr()
            answer = milp(-scores, integrality=np.ones(len(local)), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, 0, 1),
                          options={"time_limit": time_limit})
            solver_status = answer.status
            raw = answer.x
            feasible = raw is not None and np.isfinite(raw).all()
            if feasible:
                binary = np.rint(raw)
                feasible = (np.all((binary == 0) | (binary == 1))
                            and np.allclose(raw, binary, rtol=0, atol=1e-6)
                            and np.all(matrix @ binary <= 1))
            state = {0: "optimal", 1: "feasible_incumbent"}.get(solver_status, "unresolved")
            if not feasible:
                state = "unresolved"
            winners = np.flatnonzero(binary) if state != "unresolved" else np.empty(0, dtype=int)
            gap = getattr(answer, "mip_gap", None)
            if gap is not None and not np.isfinite(gap):
                gap = None

        difficult += state != "optimal"
        if audit is not None:
            audit.append(dict(candidates=group, size=len(group), detections=len(detections),
                              state=state, scipy_status=solver_status, mip_gap=gap,
                              selected=[group[j] for j in winners], seconds=monotonic() - start))
        for j in winners:
            track = local[j]
            if state == "feasible_incumbent":
                selected.append(replace(track, status="tentative_solver"))
                continue
            if len(local) > 1:
                own = set(track.indices)
                relevant = matrix[[row[i] for i in track.indices]].indices
                for k in np.unique(relevant):
                    if k == j or scores[k] < track.score - 1:
                        continue
                    alternative = set(local[k].indices)
                    if len(own.symmetric_difference(alternative)) > 1:
                        track = replace(track, status="ambiguous")
                        break
            selected.append(track)
    used = {i for track in selected for i in track.indices}
    return selected, difficult, used


def exposure_density(xy, exposure, radius=5000):
    """Local sources per square mas using each detection's own exposure."""
    density = np.empty(len(xy))
    exposure_rows = defaultdict(list)
    for i, visit in enumerate(exposure):
        exposure_rows[visit].append(i)
    area = np.pi * radius ** 2
    for group in exposure_rows.values():
        positions = xy[group]
        counts = cKDTree(positions).query_ball_point(positions, radius, return_length=True)
        # One pseudo-count gives a finite background estimate even in empty fields.
        density[group] = counts / area
    return density


def merge_candidates(tracks, models, build_track, exposure, config):
    """Refit near-duplicates with consistent trajectory and one member per visit."""
    if not tracks:
        return [], []
    position_gate = config.get("merge_position_mas", 80)
    motion_gate = config.get("merge_motion_masyr", 12)
    features = np.array([m.coeff / [[position_gate], [motion_gate]] for m in models]).reshape(-1, 4)
    index = cKDTree(features)
    retained = []
    mapping = [-1] * len(tracks)
    threshold = config.get("merge_overlap", .8)
    for i in sorted(range(len(tracks)), key=lambda j: tracks[j].score, reverse=True):
        if mapping[i] >= 0:
            continue
        current = tracks[i]
        representative = len(retained)
        mapping[i] = representative
        near = index.query_ball_point(features[i], 1, p=np.inf)
        for j in near:
            if mapping[j] >= 0:
                continue
            proposed = set(current.indices).union(tracks[j].indices)
            shared = len(current.indices) + len(tracks[j].indices) - len(proposed)
            if shared / len(proposed) < threshold:
                continue
            union = tuple(sorted(proposed))
            if len(set(exposure[list(union)])) != len(union):
                continue
            merged = build_track(union)
            if merged is not None and merged[0].score >= max(current.score, tracks[j].score):
                current = merged[0]
                mapping[j] = representative
        retained.append(current)
    return retained, mapping


def motion_tracks(xy, time, sigma, exposure, provisional, config, audit=None):
    bucket = np.floor((time - 2000) * 365.25 / config["min_season_days"]).astype(int)
    epoch = config["reference_epoch"]
    mu_max = config["mu_max"]
    gate = config["match_sigma"]
    margin = config["model_margin_mas"]
    spread = np.linalg.norm(sigma, axis=1)
    max_error = spread.max(initial=0)
    temporal = []
    for group in np.unique(bucket):
        rows = np.flatnonzero(bucket == group)
        midpoint = np.median(time[rows])
        delta = np.max(np.abs(time[rows] - midpoint))
        temporal.append((rows, cKDTree(xy[rows]), midpoint, delta, spread[rows].max()))
    density = exposure_density(xy, exposure)
    seeds = set(range(len(xy)))
    if config.get("season_seed_representatives", False):
        seeds.clear()
        for group in provisional:
            g = np.array(group)
            for value in np.unique(bucket[g]):
                k = g[bucket[g] == value]
                seeds.add(int(k[np.argmin(spread[k])]))
    seed_rows = np.array(sorted(seeds), dtype=int)
    seed_tree = cKDTree(xy[seed_rows])
    last = time.max(initial=0)
    candidates = {}
    generated = set()
    screened_seen = set()
    seed_sets = set()
    pairs = 0

    def build_track(indices):
        if len(indices) < 2:
            return None
        i = np.asarray(indices)
        if np.ptp(time[i]) < config["min_seed_years"]:
            return None
        model = fit_motion(xy[i], time[i], sigma[i], epoch)
        speed = np.linalg.norm(model.coeff[1])
        if speed > mu_max + gate * np.sqrt(sum(model.cov[:, 1, 1])):
            return None
        residual = np.linalg.norm(model.residual / sigma[i], axis=1)
        if np.any(residual > gate * 1.5):
            return None
        n_group = len(set(seasons(time[i], config["min_season_days"])))
        nights = len(set(np.floor(51544.5 + (time[i] - 2000) * 365.25)))
        # Gaussian signal relative to a uniform local background for the same visit.
        # The per-object cost means disjoint fragments pay twice for the same model.
        evidence = -np.log(2 * np.pi * np.prod(sigma[i], axis=1) * density[i]) - residual ** 2 / 2
        score = float(evidence.sum() - config.get("track_model_penalty", 8))
        if score <= 0:
            return None
        error = np.sqrt(sum(model.cov[:, 1, 1]))
        status = "insufficient_time_information"
        if (n_group >= 3 and nights >= 3 and model.span >= config.get("secure_min_span_years", 2)
                and error <= config.get("secure_max_pm_error", 100)
                and score >= config.get("secure_min_score", 10)):
            status = "secure"
        return Track(indices, score, status, n_group), model

    def add(indices):
        if len(indices) < 2 or indices in screened_seen:
            return
        screened_seen.add(indices)
        generated.add(indices)
        built = build_track(indices)
        if built is not None:
            candidates[indices] = built

    for group in provisional:
        add(group)
    for i in seed_rows:
        # The outer query is conservative. The pair-specific gate uses actual epochs.
        outer = mu_max * max(0, last - time[i]) + gate * (spread[i] + max_error) + margin
        for j in seed_rows[seed_tree.query_ball_point(xy[i], outer)]:
            dt = time[j] - time[i]
            if dt < config["min_seed_years"]:
                continue
            dv = xy[j] - xy[i]
            if np.linalg.norm(dv) > mu_max * dt + gate * np.hypot(spread[i], spread[j]) + margin:
                continue
            pairs += 1
            if pairs > config["max_seed_pairs"]:
                break
            vector = dv / dt
            support = []
            distance = []
            for rows, group_tree, midpoint, delta, largest in temporal:
                mid_alpha = (midpoint - time[i]) / dt
                margin_alpha = delta / dt
                wide_anchor = np.hypot((abs(1 - mid_alpha) + margin_alpha) * spread[i],
                                        (abs(mid_alpha) + margin_alpha) * spread[j])
                wide_radius = gate * np.hypot(largest, wide_anchor) + margin + np.linalg.norm(vector) * delta
                near = rows[group_tree.query_ball_point(xy[i] + (midpoint - time[i]) * vector, wide_radius)]
                alpha = (time[near] - time[i]) / dt
                anchor = np.hypot((1 - alpha) * spread[i], alpha * spread[j])
                uncertainty = gate * np.hypot(spread[near], anchor) + margin
                res = np.linalg.norm(xy[near] - xy[i] - (time[near] - time[i])[:, None] * vector,
                                     axis=1)
                keep = res <= uncertainty
                support.extend(near[keep])
                distance.extend(res[keep] / uncertainty[keep])
            idx = one_per_exposure(support, distance, exposure)
            if idx in seed_sets:
                continue
            seed_sets.add(idx)
            k = np.array(idx)
            if len(k) < 2 or np.ptp(time[k]) == 0:
                continue
            fit = fit_motion(xy[k], time[k], sigma[k], epoch)
            ok = np.linalg.norm(fit.residual / sigma[k], axis=1) < gate
            generated.add(idx)
            screened = tuple(k[ok])
            add(screened)
        if pairs > config["max_seed_pairs"]:
            break
    accepted, models = zip(*candidates.values()) if candidates else ([], [])
    merged, mapping = merge_candidates(accepted, models, build_track, exposure, config)
    records = []
    selected, difficult, used = choose_tracks(merged, time_limit=config.get("solver_time_limit", 5),
                                              audit=records)
    exhausted = pairs > config["max_seed_pairs"]
    if exhausted:
        selected = [replace(track, status="tentative_search") for track in selected]
    unresolved = {i for row in records if row["state"] == "unresolved" for j in row["candidates"]
                  for i in merged[j].indices}
    fallback = [list(set(track) & (unresolved - used)) for track in provisional]
    if audit is not None:
        audit.update(generated=sorted(generated), screened=list(candidates), near_duplicate_map=mapping,
                     components=records, provisional_unresolved=[sorted(g) for g in fallback if g])
    return selected, merged, sorted(set(range(len(xy))) - used), {
        "seed_detections": len(seed_rows), "seed_pairs": min(pairs, config["max_seed_pairs"]),
        "seed_budget_exhausted": exhausted, "generated": len(generated), "screened": len(candidates),
        "candidates": len(merged), "difficult_components": difficult,
        "solver_states": {state: sum(r["state"] == state for r in records)
                          for state in ("optimal", "feasible_incumbent", "unresolved")},
    }
