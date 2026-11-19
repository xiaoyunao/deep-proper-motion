import json
import numpy as np
import pandas as pd
from astropy.time import Time

from deep_pm import allband_blind_year as module
from deep_pm.astrometry import years,unproject
from deep_pm.precision_v03 import calibration_groups


def test_held_target_positions_cannot_change_allband_or_nonr_training(tmp_path,monkeypatch):
    rows=[]
    for source,x in enumerate([0.,20000.]):
        for year,band in [(2001,'r'),(2014,'g'),(2014,'VR'),(2018,'g'),(2018,'Y'),(2015,'z')]+([(2014,'r'),(2018,'r')] if source==0 else []):
            survey='SDSS' if year==2001 else 'NSC';stamp=Time(f'{year}-06-01',scale='utc').tt.mjd
            rows.append(dict(survey=survey,band=band,detection_id=f'{source}:{year}:{band}',exposure_id=f'{survey}:{year}_{band}',detector_id=1,
                observation_time=stamp,sigma_east_mas=50.,sigma_north_mas=50.,quality_flags=0,magnitude=21.5+source*.7,magnitude_error=.1,
                morphology=6. if survey=='SDSS' else .99,xx=x))
    data=pd.DataFrame(rows);sky=unproject(np.column_stack([data.xx,np.zeros(len(data))]),[0,0]);data['ra_deg']=sky[:,0];data['dec_deg']=sky[:,1]
    xy=np.column_stack([data.xx,np.zeros(len(data))]);t=years(data.observation_time);sigma=np.full((len(data),2),np.hypot(50,50));keep=data.band.eq('r').to_numpy()
    solution={key:(np.zeros(2),np.tile(np.arange(48)[:,None],(1,2)),100) for key in calibration_groups(data)}
    monkeypatch.setattr(module,'reference_frame',lambda path:[0.,0.])
    monkeypatch.setattr(module,'read_offsets',lambda path,with_gauge=False:(solution,{k:'0' for k in solution}))
    source=tmp_path/'source';source.mkdir();pd.DataFrame(dict(ra=sky[[0,8],0],dec=[0,0])).to_csv(source/'spectra_union.csv',index=False)
    chip=dict(exposure=['2015_z'],instrument=['c4d'],filter=['z'],mjd=[Time('2015-06-01',scale='utc').mjd])
    for k,(ra,dec) in enumerate([(-.01,-.01),(.01,-.01),(.01,.01),(-.01,.01)],1):chip['vertex_ra'+str(k)]=[ra%360];chip['vertex_dec'+str(k)]=[dec]
    monkeypatch.setattr(module.access,'checked_spatial',lambda *a,**kw:pd.DataFrame(chip))
    spec=dict(ra=[-.01,.01],dec=[-.01,.01])
    training=[]
    for i,shift in enumerate([0,5000]):
        positions=xy.copy();positions[data.observation_time.eq(Time('2015-06-01',scale='utc').tt.mjd),0]+=shift
        monkeypatch.setattr(module,'quality_detections',lambda *a,**kw:(data,positions,t,sigma,keep))
        work=tmp_path/str(i);work.mkdir();monkeypatch.setenv('SLURM_JOB_ID','synthetic')
        module.run(work,source,tmp_path/'graph',spec)
        training.append(json.loads((work/'private_training.json').read_text()))
    assert len(training[0])==2
    assert {r['kind'] for r in training[0]}=={'r_existing','non_r_anchor'}
    assert all(2015 not in r['training_years'] for r in training[0])
    assert training[0]==training[1]

    # Membership is a position/detection identity test, even for a catastrophically wrong PM.
    from deep_pm import allband_member_audit as membership
    trial=tmp_path/'1';saved=json.loads((trial/'private_training.json').read_text());saved[0]['pm_icrs_masyr']=[1000.,-1000.]
    (trial/'private_training.json').write_text(json.dumps(saved))
    (source/'run_manifest.json').write_text('{}')
    graph=tmp_path/'graph';graph.mkdir();(graph/'run_manifest.json').write_text('{}')
    (trial/'source_receipt.json').symlink_to(source/'run_manifest.json')
    (trial/'reference_receipt.json').symlink_to(graph/'run_manifest.json')
    monkeypatch.setattr(membership,'reference_frame',lambda path:[0.,0.])
    monkeypatch.setattr(membership,'quality_detections',lambda *a,**kw:(data,positions,t,sigma,keep))
    output=tmp_path/'audit';output.mkdir();membership.audit(output,trial)
    audit=json.loads((output/'allband_member_audit_summary.json').read_text())
    assert audit['unique_spectral_sources']==2
    for cohort in audit['by_class'].values():
        assert cohort['consistent_sources']['value']==1
        assert cohort['known_competing_member_fraction']['wilson95'][1]>0
    # The original r subset and all-band fit predict the exact same held detection.
    from deep_pm import fixed_year_multiband as paired
    frozen=tmp_path/'0'
    (frozen/'source_receipt.json').symlink_to(source/'run_manifest.json')
    (frozen/'reference_receipt.json').symlink_to(graph/'run_manifest.json')
    monkeypatch.setattr(paired,'reference_frame',lambda path:[0.,0.])
    monkeypatch.setattr(paired,'read_offsets',lambda path,with_gauge=False:(solution,{k:'0' for k in solution}))
    monkeypatch.setattr(paired,'quality_detections',lambda *a,**kw:(data,xy,t,sigma,keep))
    out=tmp_path/'paired';out.mkdir();paired.run(out,frozen)
    comparison=json.loads((out/'fixed_year_multiband_summary.json').read_text())
    assert comparison['common_held_position_count']==1
    assert comparison['cohorts']['r21__held_z']['source_count']==1
    assert comparison['maximum_replayed_allband_pm_difference_masyr']==0

    # A zero color model must reproduce the same frozen annual residuals; real
    # graph readers return tuples while the feature-model reader stores a list.
    from deep_pm import chromatic_validation as chromatic
    colors=dict(base=solution,gauges={k:'0' for k in solution},roles={k:False for k in solution},
                coefficient={},feature_draws={},color_reference=1.2,center=[0.,0.])
    monkeypatch.setattr(chromatic,'read_model',lambda path:colors)
    monkeypatch.setattr(chromatic,'reference_frame',lambda path:(0.,0.))
    monkeypatch.setattr(chromatic,'quality_detections',lambda *a,**kw:(data,xy,t,sigma,keep))
    model_file=tmp_path/'color-model';model_file.write_text('synthetic fixed model')
    output=tmp_path/'chromatic';output.mkdir();chromatic.year_trial(output,frozen,model_file)
    held=json.loads((output/'private_chromatic_held.json').read_text())
    assert len(held)==2
    for row in held:
        np.testing.assert_allclose(row['residual'][0],row['residual'][1],atol=1e-8)
        np.testing.assert_allclose(row['error'][0],row['error'][1],atol=1e-8)
    import pytest
    monkeypatch.setattr(chromatic,'reference_frame',lambda path:(0.,1.))
    with pytest.raises(ValueError,match='same reference coordinate basis'):
        chromatic.year_trial(output,frozen,model_file)

    monkeypatch.setattr(chromatic,'reference_frame',lambda path:(0.,0.))
    zero=tmp_path/'chromatic-zero';zero.mkdir();chromatic.blind_zero_trial(zero,frozen,model_file)
    result=json.loads((zero/'chromatic_blind_zero_summary.json').read_text())
    assert result['frozen_blind_source_count']==2
    for cohort in result['cohorts'].values():
        assert cohort['known_competing_member_fraction']['n']==0

    from deep_pm import nsc_information as information
    monkeypatch.setattr(information,'reference_frame',lambda path:(0.,0.))
    monkeypatch.setattr(information,'read_offsets',lambda path,with_gauge=False:(solution,{k:'0' for k in solution}))
    monkeypatch.setattr(information,'quality_detections',lambda *a,**kw:(data,xy,t,sigma,keep))
    info_work=tmp_path/'time-information';info_work.mkdir();information.year(info_work,frozen)
    ablation=json.loads((info_work/'nsc_information_summary.json').read_text())
    assert ablation['fixed_r_training_sources']==1
    assert ablation['cohorts']['r21']['B_new_nights_vs_A_r']['source_count']==1
