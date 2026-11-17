"""Relative field offsets measured from overlapping SDSS field reductions."""
import numpy as np

from .validation_audit import sdss_overlap_rows


def overlap_constraints(pairs, refs, sigma, options):
    """A repeat reduction reuses a physical visit and its primary position."""
    overlaps = pairs.copy()
    train_names = set(refs.loc[~refs.is_holdout, "reference"])
    held_names = set(refs.loc[refs.is_holdout, "reference"])
    if not set(overlaps.ref) <= train_names - held_names:
        raise ValueError("Overlap constraints must exclude all held galaxy identities")
    # Do not reuse a counterpart if more than one selected primary claims it.
    overlaps = overlaps.loc[~overlaps.counterpart.duplicated(keep=False)].copy()
    native = overlaps[["err_e", "err_n"]].to_numpy(dtype=float)
    good = np.isfinite(native).all(axis=1) & (native > 0).all(axis=1)
    overlaps = overlaps.loc[good].copy()
    errors = sigma[overlaps["index"].to_numpy(dtype=int)]
    combined = np.hypot(np.hypot(native[good], errors), options["error_floor_mas"])
    overlaps[["sigma_e", "sigma_n"]] = combined
    multiplicity = overlaps.groupby(["ref", "run"]).run.transform("size")
    overlaps["weight"] = options["training_weight"] / multiplicity
    return overlaps


def overlap_trial(sas, refs, data, xy, sigma, keys, offsets, center, options):
    pairs, counts = sdss_overlap_rows(
        sas, refs, data, xy, keys, offsets, center, holdout=False,
        max_fields=options["max_neighbor_fields"], match_mas=options["match_mas"],
        margin_mas=options["neighbor_margin_mas"])
    overlaps = overlap_constraints(pairs, refs, sigma, options)
    counts.update(usable_pairs=len(overlaps), reference_count=overlaps.ref.nunique(),
                  usable_runs=overlaps.run.nunique(),
                  field_links=len(set(zip(overlaps["first"], overlaps["second"]))))
    return overlaps, counts


def assess_boundary(rows, counts, offsets, options, seed):
    valid = np.logical_and.reduce([
        rows["first"].isin(model) & rows.second.isin(model) for model in offsets[1:]])
    common = rows.loc[valid]
    counts = dict(counts, common_pairs=len(common), common_references=common.ref.nunique(),
                  common_runs=common.run.nunique())
    if common.empty:
        return counts
    raw = common[["east", "north"]].to_numpy()
    values = [raw]
    for model in offsets[1:]:
        correction = np.array([model[a][0]-model[b][0]
                               for a, b in zip(common["first"], common.second)])
        values.append(raw-correction)
    values = np.stack(values, axis=1)
    radial = np.linalg.norm(values, axis=2)
    p68, p90, p95 = np.quantile(radial, [.68, .90, .95], axis=0)
    counts.update(radial_p68_mas=p68, radial_p90_mas=p90, radial_p95_mas=p95,
                  median_mas=np.median(values, axis=0))
    blocks = list(common.groupby("run").indices.values())
    if len(blocks) >= options["min_independent_blocks"]:
        rng = np.random.default_rng(seed)
        bootstrap = []
        for draw in rng.integers(len(blocks), size=(options["resample_draws"], len(blocks))):
            sample = np.concatenate([blocks[i] for i in draw])
            estimate = np.quantile(radial[sample], .68, axis=0)
            if estimate[0] > 0:
                bootstrap.append(estimate[1:]/estimate[0]-1)
        if bootstrap:
            counts["relative_loss95_vs_c0"] = np.quantile(bootstrap, [.025, .975], axis=0).T
    return counts


def pair_audit(training, held, models):
    """Keep per-field geometries and errors on NERSC for reproduction and audit."""
    output = {}
    for name, records in (("training", training), ("held", held)):
        for keys, rows in records.groupby(["first", "second"], sort=True):
            a, b = sorted(keys)
            sign = 1 if keys == (a, b) else -1
            entry = output.setdefault(f"{a}__{b}", {})
            raw = sign * rows[["east", "north"]].to_numpy()
            if name not in entry:
                entry[name] = []
            entry[name].extend(raw.tolist())
    for key, entry in output.items():
        a, b = key.split("__")
        solution = {}
        for name, model in zip(("C1", "C1b"), models[1:]):
            if a in model and b in model:
                delta = model[a][0]-model[b][0]
                draws = model[a][1]-model[b][1]
                solution[name] = dict(difference_mas=delta, uncertainty_mas=np.std(draws, axis=0, ddof=1))
        entry["calibration"] = solution
        for name in ("training", "held"):
            raw = np.asarray(entry.get(name, []))
            if len(raw):
                entry[name] = dict(count=len(raw), raw_median_mas=np.median(raw, axis=0),
                                   raw_radial_p68_mas=np.quantile(np.linalg.norm(raw, axis=1), .68))
            else:
                entry[name] = dict(count=0)
    return output
