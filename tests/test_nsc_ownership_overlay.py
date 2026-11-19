import copy
import gzip
import json

import pytest

from deep_pm.nsc_ownership_overlay import transfer,record_hash,owned_records,STANDARD
from deep_pm.io import digest

SPECS=dict(a=dict(ra=[359.,360.],dec=[-1.,1.]),b=dict(ra=[0.,1.],dec=[-1.,1.]))


def test_boundary_owner_changes_without_moving_astrometry_or_covariance():
    record=dict(position_deg=[.00001,0.],pm_icrs_masyr=[20.,-2.],covariance=[[1.,.2],[.2,2.]],detection_ids=['real:physical:1'])
    before=copy.deepcopy(record);move=transfer(record,'a',7,SPECS)
    assert move['owner_core']=='b' and move['source_slot']==7
    assert record==before and move['unchanged_record_sha256']==record_hash(before)
    changed=copy.deepcopy(record);changed['covariance'][0][1]+=.00001
    assert record_hash(changed)!=move['unchanged_record_sha256']
    assert transfer(record,'b',7,SPECS) is None
    assert transfer(dict(record,position_deg=[0.,0.]),'a',7,SPECS)['owner_core']=='b'
    with pytest.raises(ValueError):transfer(dict(record,position_deg=[2.,0.]),'a',7,SPECS)
    with pytest.raises(ValueError):transfer(record,'a',7,dict(SPECS,c=SPECS['b']))


def test_owned_reader_is_exclusive_and_detects_changed_member(tmp_path,monkeypatch):
    from deep_pm import nsc_ownership_overlay as m
    monkeypatch.setattr(m,'verify_directory',lambda *args:None)
    row=dict(position_deg=[.01,0.],covariance=[[1.]],detection_ids=['physical:1'])
    folders={}
    for core,records in [('a',[row]),('b',[])]:
        folder=tmp_path/core;folder.mkdir();folders[core]=str(folder)
        with gzip.open(folder/STANDARD,'wt') as stream:json.dump(records,stream)
    fits=tmp_path/'fits.json';fits.write_text(json.dumps(folders));overlay=dict(fits_manifest_sha256=digest(fits),transfers=[transfer(row,'a',0,SPECS)])
    (tmp_path/'private_owner_overlay.json').write_text(json.dumps(overlay))
    assert list(owned_records(tmp_path,'a'))==[]
    assert list(owned_records(tmp_path,'b'))==[('a',0,row)]
    row['detection_ids']=['different:physical']
    with gzip.open(tmp_path/'a'/STANDARD,'wt') as stream:json.dump([row],stream)
    with pytest.raises(ValueError):list(owned_records(tmp_path,'b'))
