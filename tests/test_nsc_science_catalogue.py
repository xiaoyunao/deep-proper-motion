import gzip
import json

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from deep_pm import nsc_science_catalogue as module
from deep_pm.astrometry import unproject,years
from deep_pm.precision_v03 import calibration_groups


@pytest.mark.parametrize('missing_spectral_r',[False,True])
def test_three_research_classes_keep_unique_real_measurements_and_full_covariance(tmp_path,monkeypatch,missing_spectral_r):
    records=[]
    for source,visits in enumerate([[(2001,'r'),(2014,'r'),(2018,'r'),(2018,'i')],[(2001,'r'),(2014,'r'),(2018,'z')],[(2001,'r'),(2014,'Y'),(2018,'VR')]]):
        for year,band in visits:
            survey='SDSS' if year==2001 else 'NSC'
            records.append(dict(survey=survey,band=band,instrument='c4d' if survey=='NSC' else 'SDSS',detection_id=f'{source}:{year}:{band}',exposure_id=f'{survey}:{year}_{band}',detector_id=1,observation_time=Time(f'{year}-06-01',scale='utc').tt.mjd,sigma_east_mas=50.,sigma_north_mas=50.,quality_flags=0,magnitude=21.5+source*.3,magnitude_error=.1,morphology=6. if survey=='SDSS' else .99,xx=source*20000.))
    data=pd.DataFrame(records);xy=np.c_[data.xx,np.zeros(len(data))];sky=unproject(xy,[0,0]);data['ra_deg']=sky[:,0];data['dec_deg']=sky[:,1]
    time=years(data.observation_time);sigma=np.full((len(data),2),np.hypot(50.,50.));keep=data.band.eq('r').to_numpy()
    solution={k:(np.zeros(2),np.tile(np.arange(48)[:,None],(1,2)),100) for k in calibration_groups(data)}
    monkeypatch.setattr(module,'reference_frame',lambda p:(0.,0.));monkeypatch.setattr(module,'read_offsets',lambda *a,**kw:(solution,{k:'0' for k in solution}))
    monkeypatch.setattr(module,'quality_detections',lambda *a:(data,xy,time,sigma,keep))
    qso=pd.DataFrame(dict(ra=[0.],dec=[0.],reference_r=[np.nan])) if missing_spectral_r else pd.DataFrame(columns=['ra','dec','reference_r'])
    monkeypatch.setattr(module,'stationary_controls',lambda *a:({},qso,{'all_nsc':{}}))
    monkeypatch.setenv('SLURM_JOB_ID','synthetic');shared=tmp_path/'shared';shared.mkdir();(shared/'private_c0.npz').write_text('synthetic-bound-graph')
    work=tmp_path/'out';work.mkdir();module.run(work,tmp_path/'source',shared,dict(ra=[-.1,.1],dec=[-.1,.1]),tmp_path/'legacy')
    with gzip.open(work/'private_standard_motion.json.gz','rt') as stream:rows=json.load(stream)
    assert len(rows)==3
    assert {r['kind'] for r in rows}=={'r_existing','extra_r_anchor','non_r_anchor'}
    members=[v for r in rows for v in r['detection_ids']];assert len(members)==len(set(members))
    nonr=next(r for r in rows if r['kind']=='non_r_anchor');assert nonr['magnitude_source']=='SDSS_r';assert nonr['bands']==['VR','Y']
    for row in rows:
        covariance=np.asarray(row['covariance']);assert covariance.shape==(4,4)
        np.testing.assert_allclose(np.sqrt(np.diag(covariance)[[1,3]]),row['pm_error_masyr'])
        assert row['information']['sdss_physical_runs']==1
        assert 'spatial_zero_uncertified' in row['quality_flags']

    blind=json.loads((work/'private_blind_qso.json').read_text())
    summary=json.loads((work/'nsc_science_catalogue_summary.json').read_text())
    assert summary['spectral_matches_without_reference_r']==int(missing_spectral_r)
    if missing_spectral_r:assert len(blind)==1 and blind[0]['magnitude'] is None
