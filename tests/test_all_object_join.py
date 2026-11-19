import pandas as pd
from deep_pm.exposure_inputs import tap_join


def test_all_object_join_removes_r_selection_without_changing_legacy_default():
    class Cache:
        def __init__(self):self.conditions=[]
        def checked_native_window(self,columns,table,where,cap):
            self.conditions.append(where)
            return pd.DataFrame(dict(measid=['g-only'],objectid=['no-r']))
    archive=pd.DataFrame(dict(ra=[10.],dec=[1.],ndet=[1]))
    options=dict(box=dict(ra=[9.9,10.1],dec=[.9,1.1]),halo=.01,service='nsc',workers=1,native_window=True)
    old=Cache();new=Cache()
    a=tap_join(old,archive,**options);b=tap_join(new,archive,all_objects=True,**options)
    assert a.equals(b)
    assert 'o.ndetr>0' in old.conditions[0] and 'o.rmag<24' in old.conditions[0]
    assert 'ndetr' not in new.conditions[0] and 'rmag' not in new.conditions[0]
    assert 'o.ra' in new.conditions[0] and 'q3c_radial_query' in new.conditions[0]


import pytest


@pytest.mark.parametrize('raw_sdss_available',[False,True])
def test_allband_input_preserves_sdss_and_physically_separates_index_labels(tmp_path,monkeypatch,raw_sdss_available):
    import json
    import numpy as np
    from deep_pm import nsc_allband_inputs as module,normalize
    parent=tmp_path/'parent';parent.mkdir();work=tmp_path/'work';work.mkdir()
    raw=pd.DataFrame(dict(measid=['r-old','g-new'],objectid=['with-r','no-r'],exposure=['r-exp','g-exp'],ccdnum=[1,1],mjd=[57000.,58000.],ra=[10.,10.001],dec=[1.,1.001],raerr=[.05,.08],decerr=[.06,.09],filter=['r','g'],mag_auto=[21.5,22.],magerr_auto=[.1,.1],class_star=[.99,.99],flags=[0,0],fwhm=[1.,1.],x=[50,60],y=[60,70]))
    old=normalize.nsc(raw.iloc[:1]);early=old.iloc[:1].copy();early['survey']='SDSS';early['detection_id']='SDSS:frozen';early['exposure_id']='SDSS:run1';early['original_object_id']='physical-run1';early['observation_time']=52000.;early['ra_deg']=10.0001
    prior=pd.concat([early,old],ignore_index=True);prior[['detection_id','original_object_id']].to_csv(parent/'external_labels.csv',index=False);prior.drop(columns='original_object_id').to_csv(parent/'detections.csv',index=False)
    for relative in ('spectra_union.csv','spectra_veto.csv','validation/gaia.csv')+ (('sdss.csv',) if raw_sdss_available else ()):
        p=parent/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('ra,dec\n')
    objects=pd.DataFrame(dict(objectid=['with-r','no-r'],ra=[10,10.001],dec=[1,1.001],pmra=[0,0],pmdec=[0,0],pmraerr=[1,1],pmdecerr=[1,1],deltamjd=[1000,1000],ndet=[1,1]))
    chips=pd.DataFrame(dict(exposure=['r-exp','g-exp'],ccdnum=[1,1],instrument=['c4d','c4d'],airmass=[1.2,1.3]))
    def query(cache,service,columns,source,*args):return objects if source.endswith('.object') else chips
    monkeypatch.setattr(module.access,'checked_spatial',query);monkeypatch.setattr(module,'tap_join',lambda *a,**kw:raw.copy());monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    module.acquire(work,parent,dict(ra=[9.9,10.1],dec=[.9,1.1]),dict(halo_deg=.06,native_row_cap=80000),1)
    result=pd.read_csv(work/'detections.csv');assert 'original_object_id' not in result
    assert set(result.detection_id)=={'SDSS:frozen','NSC:r-old','NSC:g-new'}
    actual=result.loc[result.survey.eq('SDSS')].iloc[0]
    assert actual.ra_deg==early.ra_deg.iat[0] and actual.observation_time==early.observation_time.iat[0]
    summary=json.loads((work/'acquisition_summary.json').read_text());assert summary['new_nsc_measures']==1
    assert summary['raw_sdss_photometry_available']==raw_sdss_available
    assert (work/'sdss.csv').is_file()==raw_sdss_available
