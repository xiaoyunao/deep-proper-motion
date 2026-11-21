import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from deep_pm.astrometry import project
from deep_pm.continuous_reference import solve,read_offsets
from deep_pm.io import digest, save_json
from deep_pm.joint_reference import reproject,identity_split,source_projection


def test_reprojection_keeps_raw_measurement_identity_split_and_errors():
    rows=pd.DataFrame(dict(ra_deg=[187.9,188.1],dec_deg=[33.,33.],east=[1.,2.],north=[3.,4.],
        reference=['a','b'],detection_id=['x','y'],is_holdout=[True,False],sigma_e=[40.,50.],sigma_n=[60.,70.]))
    converted=reproject(rows,(188.,33.))
    assert identity_split(converted)==identity_split(rows)
    np.testing.assert_allclose(converted[['east','north']],project(rows.ra_deg,rows.dec_deg,(188.,33.)))
    assert rows.east.tolist()==[1.,2.]


def test_legacy_reference_projection_requires_consistent_archived_coordinates():
    frame=pd.DataFrame(dict(ra_deg=[345.,355.],dec_deg=[-2.,1.]))
    frame[['east','north']]=project(frame.ra_deg,frame.dec_deg,(350.,0.))
    assert source_projection(frame,{})==(350.,0.)
    frame.loc[0,'east']+=.01
    with pytest.raises(ValueError,match='legacy coordinates'):
        source_projection(frame,{})


def test_group_trace_saves_se_rejection_without_changing_retained_offsets(tmp_path):
    source=tmp_path/'input';source.mkdir()
    rows=[]
    for i in range(12):
        for group in ('SDSS:1:1','NSC:one','NSC:two'):
            rows.append(dict(reference=str(i),detection_id=f'{group}:{i}',exposure_id=group,group=group,
                is_holdout=False,ra_deg=10.1,dec_deg=.1,east=1.,north=2.,sigma_e=50.,sigma_n=50.))
    pd.DataFrame(rows).to_csv(source/'reference.csv.gz',index=False)
    (source/'reference_summary.json').write_text(json.dumps(dict(projection_center_deg=[10.,0.])))
    files = ("reference.csv.gz", "reference_summary.json")
    save_json(source/"run_manifest.json", {"stages": {"reference": [{
        "state": "completed",
        "outputs": {name: digest(source/name) for name in files},
    }]}})
    random=np.arange(48)-23.5
    values={'SDSS:1:1':(np.ones(2),np.zeros((48,2)),12),'NSC:one':(np.ones(2),np.zeros((48,2)),12),
            'NSC:two':(np.ones(2),np.column_stack([100*random,100*random]),12)}
    # With two NSC groups, the gauge makes all groups uncertain. Add many
    # zero-draw NSC anchors so only the intended group exceeds 12 mas.
    for i in range(4):values[f'NSC:anchor{i}']=(np.ones(2),np.zeros((48,2)),12)
    a=tmp_path/'plain';b=tmp_path/'trace';a.mkdir();b.mkdir()
    with patch('deep_pm.continuous_reference.fit_offsets',return_value=values):
        solve(a,{'core':source},{'core':dict(ra=[10.,11.],dec=[0.,1.])},True)
        solve(b,{'core':source},{'core':dict(ra=[10.,11.],dec=[0.,1.])},True,True)
    x,y=read_offsets(a/'private_c0.npz'),read_offsets(b/'private_c0.npz')
    assert x.keys()==y.keys() and 'NSC:two' not in x
    for key in x:
        for left,right in zip(x[key],y[key]):np.testing.assert_array_equal(left,right)
    trace=pd.read_csv(b/'private_reference_group_selection.csv.gz')
    assert trace.loc[trace.group.eq('NSC:two'),'status'].iloc[0]=='offset_se_rejected'
