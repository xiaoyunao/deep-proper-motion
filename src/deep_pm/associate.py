"""Static and motion-aware association on the same detections."""
from collections import defaultdict
from dataclasses import dataclass

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


def static_tracks(xy, exposure, radius, eps=None):
    if not len(xy):
        return []
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
    tree = cKDTree(xy)
    centers = [np.median(xy[group], axis=0) for group in groups]
    if centers:
        center_tree = cKDTree(centers)
        for index in np.flatnonzero(~used):
            possibilities = center_tree.query_ball_point(xy[index], radius)
            for label in sorted(possibilities, key=lambda j: np.linalg.norm(xy[index] - centers[j])):
                if exposure[index] not in exposure[groups[label]]:
                    groups[label].append(index)
                    used[index] = True
                    break
    for index in np.flatnonzero(~used):
        if used[index]:
            continue
        candidates = np.array(tree.query_ball_point(xy[index], radius), dtype=int)
        candidates = candidates[~used[candidates]]
        chosen = one_per_exposure(candidates, np.linalg.norm(xy[candidates] - xy[index], axis=1),
                                  exposure)
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


def choose_tracks(candidates, max_component):
    selected = []
    difficult = 0
    for group, detections in components(candidates):
        local = [candidates[i] for i in group]
        if len(local) == 1:
            selected.extend(local)
            continue
        if len(local) > max_component:
            difficult += 1
            continue
        row = {i: j for j, i in enumerate(sorted(detections))}
        entries = [(row[i], j) for j, track in enumerate(local) for i in track.indices]
        rows, columns = np.array(entries, dtype=np.int32).T.copy()
        matrix = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(row), len(local)))
        scores = np.array([track.score for track in local])
        answer = milp(-scores, integrality=np.ones(len(local)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, 0, 1), options={"time_limit": 5})
        if not answer.success:
            difficult += 1
            continue
        for j in np.flatnonzero(answer.x > 0.5):
            track = local[j]
            conflicting = [other for k, other in enumerate(local)
                           if k != j and set(track.indices).intersection(other.indices)]
            if any(other.score >= track.score - 1 and
                   len(set(other.indices).symmetric_difference(track.indices)) > 1
                   for other in conflicting):
                track = Track(track.indices, track.score, "ambiguous", track.groups)
            selected.append(track)
    used = {i for track in selected for i in track.indices}
    return selected, difficult, used


def motion_tracks(xy, time, sigma, exposure, provisional, config):
    tree = cKDTree(xy)
    season = seasons(time, config["min_season_days"])
    epoch = config["reference_epoch"]
    mu_max = config["mu_max"]
    gate = config["match_sigma"]
    margin = config["model_margin_mas"]
    spread = np.linalg.norm(sigma, axis=1)
    max_error = spread.max(initial=0)
    temporal = []
    for group in np.unique(season):
        rows = np.flatnonzero(season == group)
        midpoint = np.median(time[rows])
        delta = np.max(np.abs(time[rows] - midpoint))
        temporal.append((rows, cKDTree(xy[rows]), midpoint, delta, spread[rows].max()))
    density = (tree.query_ball_point(xy, 5000, return_length=True) - 1) / 25
    seeds = set(range(len(xy)))
    if config.get("season_seed_representatives", False):
        seeds.clear()
        for group in provisional:
            g = np.array(group)
            for value in np.unique(season[g]):
                k = g[season[g] == value]
                seeds.add(int(k[np.argmin(spread[k])]))
    seed_rows = np.array(sorted(seeds), dtype=int)
    seed_tree = cKDTree(xy[seed_rows])
    last = time.max(initial=0)
    candidates = {}
    seed_sets = set()
    pairs = 0

    def add(indices):
        if len(indices) < 2 or indices in candidates:
            return
        i = np.array(indices)
        if np.ptp(time[i]) < config["min_seed_years"]:
            return
        model = fit_motion(xy[i], time[i], sigma[i], epoch)
        speed = np.linalg.norm(model.coeff[1])
        if speed > mu_max + gate * np.sqrt(sum(model.cov[:, 1, 1])):
            return
        residual = np.linalg.norm(model.residual / sigma[i], axis=1)
        if np.any(residual > gate * 1.5):
            return
        n_group = len(set(season[i]))
        status = "secure" if n_group >= 3 else "insufficient_time_information"
        penalty = np.log1p(np.median(density[i]))
        score = 3 * (n_group - 1) + np.log(len(i)) - np.mean(residual ** 2) / 2 - penalty
        if score > 0:
            candidates[indices] = Track(indices, score, status, n_group)

    for group in provisional:
        add(group)
    for i in seed_rows:
        # The outer query is conservative. The pair-specific gate uses actual epochs.
        outer = mu_max * max(0, last - time[i]) + gate * (spread[i] + max_error) + margin
        for j in seed_rows[seed_tree.query_ball_point(xy[i], outer)]:
            dt = time[j] - time[i]
            if dt < config["min_seed_years"] or season[i] == season[j]:
                continue
            dv = xy[j] - xy[i]
            if np.linalg.norm(dv) > mu_max * dt + gate * np.hypot(spread[i], spread[j]) + margin:
                continue
            pairs += 1
            if pairs > config["max_seed_pairs"]:
                raise RuntimeError("Seed-pair budget exceeded. Use a smaller region or increase the explicit budget.")
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
            add(tuple(k[ok]))
    selected, difficult, used = choose_tracks(list(candidates.values()), config["max_component"])
    return selected, list(candidates.values()), sorted(set(range(len(xy))) - used), {
        "seed_detections": len(seed_rows), "seed_pairs": pairs, "candidates": len(candidates), "difficult_components": difficult,
    }
