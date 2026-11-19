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
