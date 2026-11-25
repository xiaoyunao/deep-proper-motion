"""Describe published morphology values without changing frozen selection."""
import numpy as np


def morphology_state(values, threshold=.8):
    values=np.asarray(values,float)
    finite=np.isfinite(values)
    valid=finite&(values>=0)&(values<=1)
    invalid=finite&~valid
    legacy=np.median(values) if len(values) else np.nan
    median=np.median(values[valid]) if valid.any() else None
    if invalid.any():state='finite_out_of_range'
    elif not finite.any():state='all_missing'
    elif not finite.all():state='mixed_valid_missing'
    else:state='valid_pointlike' if median>=threshold else 'valid_below_threshold'
    return dict(state=state,total_count=len(values),valid_count=int(valid.sum()),
        missing_count=int((~finite).sum()),invalid_count=int(invalid.sum()),
        unknown_fraction=float((~valid).mean()) if len(values) else 1.,
        legacy_median=float(legacy) if np.isfinite(legacy) else None,
        valid_median=float(median) if median is not None else None,
        legacy_pass=bool(not legacy<threshold),valid_only_pass=bool(median is not None and median>=threshold))
