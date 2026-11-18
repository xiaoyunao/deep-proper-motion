import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from deep_pm.astrometry import project
from deep_pm.independent_summary import run, standard_block
from deep_pm.paired_control import catalogue_motion
from deep_pm.standard_motion import icrs_motion


def test_published_inverse_conversion_restores_native_motion_at_high_declination():
    center=(150.,35.)
    source=pd.Series(dict(ra=146.,dec=39.,pmra=123.,pmdec=-76.))
    tangent=catalogue_motion(source,center)
    position=project([source.ra],[source.dec],center)[0]
    c0=icrs_motion(position,tangent,center)['pm_icrs_masyr']
    row=dict(position_deg=[source.ra,source.dec],motion={'C0':tangent,'NSC':tangent},
             motion_icrs={'C0':c0,'NSC':[999.,999.]},nsc_nights_utc18=[50000])
    converted=standard_block(row,center)
    np.testing.assert_allclose(converted['motion']['NSC'],[123.,-76.],atol=3e-4)  # historical catalogue_motion uses a one-arcsec forward step
    assert converted['nsc_epoch_days']==[50000]


def test_sealed_region_filtered_before_any_outcome_access():
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/'tasks.tsv').write_text('held_00\tnorth_two\n')
        protocol=root/'protocol.toml'
        protocol.write_text('final_test_box = "north_two"\n')
        output=root/'summary.json'
        with patch.dict(os.environ,SLURM_JOB_ID='test'), patch('deep_pm.independent_summary.pool',return_value={'cohorts':{}}) as pool:
            run(root,protocol,output)
        pool.assert_called_once_with({}, {}, {}, {})
        result=json.loads(output.read_text())
        assert result['test_results_read'] is False
        assert result['sealed_core_count']==1
