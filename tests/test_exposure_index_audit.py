import json
import numpy as np
import pandas as pd
from deep_pm import exposure_index_audit as module


def test_exposure_route_counts_nonr_object_index_losses_without_using_ids_as_truth(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();work=tmp_path/'work';work.mkdir()
    pd.DataFrame(dict(detection_id=['NSC:old_early','NSC:old_late'])).to_csv(source/'detections.csv',index=False)
    pd.DataFrame(dict(id=['original'])).to_csv(source/'official_nsc.csv',index=False)
    chip=dict(exposure=['early','late'],ccdnum=[1,1],instrument=['c4d']*2,filter=['Y']*2,mjd=[57000.,58000.])
    for k,(ra,dec) in enumerate([(-.02,-.02),(.02,-.02),(.02,.02),(-.02,.02)],1):chip['vertex_ra'+str(k)]=[ra%360]*2;chip['vertex_dec'+str(k)]=[dec]*2
    queried=[]
    def request(cache,service,columns,table,where,*args):
        if table=='nsc_dr2.chip':return pd.DataFrame(chip)
        assert table=='nsc_dr2.meas' and 'ccdnum=1' in where and 'exposure=' in where
        exp='early' if "'early'" in where else 'late';queried.append(exp)
        return pd.DataFrame(dict(measid=['old_'+exp,'new_'+exp],objectid=['original','nonr'],exposure=[exp]*2,ccdnum=[1]*2,mjd=[57000.]*2,ra=[0,.001],dec=[0,0],raerr=[.05]*2,decerr=[.05]*2,filter=['Y']*2,mag_auto=[22.]*2,magerr_auto=[.1]*2,class_star=[.99]*2,flags=[0]*2,fwhm=[1]*2,x=[100]*2,y=[100]*2))
    monkeypatch.setattr(module.access,'checked_spatial',request)
    monkeypatch.setattr(module.access.Cache,'checked',lambda *a,**kw:pd.DataFrame(dict(objectid=['nonr'],ndetr=[0],nphotr=[0],rmag=[99.99])))
    monkeypatch.setattr(module,'read_offsets',lambda path:{'NSC:early':None,'NSC:late':None})
    monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    module.run(work,source,tmp_path/'graph',dict(ra=[-.01,.01],dec=[-.01,.01]))
    result=json.loads((work/'index_route_audit_summary.json').read_text())
    assert queried==['early','late']
    assert result['by_filter']['Y']['missing_physical_reference_supported']==2
    assert result['missing_object_filter_diagnostics']['no_r_detection_objects']==1
    assert result['missing_object_filter_diagnostics']['missing_measurements_from_original_index_objects']==0
