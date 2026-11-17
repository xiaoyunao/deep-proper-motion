import numpy as np
import pandas as pd
import pytest

from deep_pm.overlap_calibration import overlap_constraints


def test_fitting_repeats_drops_shared_counterparts_and_caps_weight_by_source_run():
    refs = pd.DataFrame(dict(reference=["fit", "held"], is_holdout=[False, True]))
    rows = pd.DataFrame(dict(ref=["fit"] * 5, counterpart=["a", "b", "c", "c", "d"],
                             run=[1, 1, 1, 1, 2], index=[0] * 5,
                             err_e=[30.] * 5, err_n=[40.] * 5))
    options = dict(error_floor_mas=20, training_weight=1)
    sigma = np.full((2, 2), 10.)
    overlap = overlap_constraints(rows, refs, sigma, options)
    assert list(overlap.counterpart) == ["a", "b", "d"]
    np.testing.assert_equal(overlap.weight.to_numpy(), [.5, .5, 1.])
    np.testing.assert_allclose(overlap.sigma_e, np.sqrt(30**2 + 20**2 + 10**2))
    rows.loc[0, "ref"] = "held"
    with pytest.raises(ValueError, match="held galaxy"):
        overlap_constraints(rows, refs, sigma, options)
