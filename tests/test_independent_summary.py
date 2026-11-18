import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from deep_pm.astrometry import project
from deep_pm.independent_summary import run, standard_block
from deep_pm.paired_control import catalogue_motion
from deep_pm.standard_motion import icrs_motion


def test_invalid_release_cannot_open_any_test_output(tmp_path):
    from deep_pm.io import digest
    protocol=tmp_path/'protocol.toml';protocol.write_text('final_test_box = "north_two"\n')
    (tmp_path/'selection.json').write_text('{}')
    release=tmp_path/'release.json'
    release.write_text(json.dumps(dict(selection_sha256=digest(tmp_path/'selection.json'))))
    with patch('deep_pm.independent_summary.pool') as pool,pytest.raises(ValueError):
        run(tmp_path,protocol,tmp_path/'output.json',release)
    pool.assert_not_called()


def test_shape_diagnostic_partitions_the_same_common_controls_without_dropping_unknowns():
    from deep_pm.independent_summary import comparison_cohorts
    rows=[]
    for shape in [.9]*5+[.2,None]:
        rows.append(dict(magnitude=21.5,median_nsc_r_class_star=shape,
            motion={'NSC':[10.,1.],'Xiao':[5.,1.],'C0':[3.,1.]},
            error_icrs={key:[2.,2.] for key in ('NSC','Xiao','C0')},
            exposures=['SDSS:42','NSC:a'],nsc_epoch_days=[50000],nsc_span_years=1.,
            sdss_span_years=0.,joint_span_years=8.,calendar_year_count=3))
    result=comparison_cohorts(rows)['r21_22']['conditions']
    assert result['all']['common']['n']==7
    assert result['pointlike_qso']['common']['n']==5
    assert result['nonpointlike_qso']['common']['n']==1
    assert result['shape_unavailable']['common']['n']==1


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
        with patch.dict(os.environ,SLURM_JOB_ID='test'), patch('deep_pm.independent_summary.pool',return_value={'cohorts':{},'cores':{}}) as pool:
            run(root,protocol,output)
        pool.assert_called_once_with({}, {}, {}, {})
        result=json.loads(output.read_text())
        assert result['test_results_read'] is False
        assert result['sealed_core_count']==1


def test_historical_unknown_blind_membership_is_not_zero_recovery():
    from deep_pm.independent_summary import blind_cohorts
    result=blind_cohorts([dict(magnitude=21.5)])['r21_22']
    assert result['status']=='unavailable_in_historical_private_records'
    assert 'blind_recovered' not in result


def test_blind_quality_is_compared_on_the_identical_recovered_subset():
    from deep_pm.independent_summary import blind_cohorts
    rows=[]
    for i in range(6):
        rows.append(dict(magnitude=21.5,blind_track_ids={'C0':i if i<5 else None},
            motion={'NSC':[8.,0.],'Xiao':[5.,0.],'C0':[3.,0.]},error_icrs={k:[2.,2.] for k in ('NSC','Xiao','C0')},
            blind_C0=([3.,0.],[2.,2.]) if i<5 else None,exposures=['SDSS:1','NSC:a'],nsc_epoch_days=[50000],
            blind_visits=['SDSS:1','NSC:a','NSC:b'],blind_nights=[50000,50100]))
    result=blind_cohorts(rows)['r21_22']
    assert result['blind_recovered']['n']==5
    assert result['blind_recovered']['denominator']==6
    sample=result['strict_common_comparison']
    assert sample['n']==5
    np.testing.assert_allclose(sample['paired']['C0_blind_vs_C0_known']['fraction_improved'][2],0)
    assert sample['paired']['C0_blind_vs_C0_known']['fraction_improved'][1] is None
    np.testing.assert_allclose(sample['paired']['C0_blind_vs_Xiao']['fraction_improved'][2],.4)
