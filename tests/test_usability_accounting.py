import gzip,json
from deep_pm.usability_accounting import key,retained


def test_source_accounting_uses_physical_members_instead_of_motion():
    a=dict(detection_ids=['c','a','b'],pm_icrs_masyr=[1,2])
    b=dict(detection_ids=['a','b','c'],pm_icrs_masyr=[100,-200])
    assert key(a)==key(b)
    assert key(a)!=key(dict(detection_ids=['a','b','d']))


def test_retained_accounting_obeys_final_owner_and_archived_exterior(tmp_path,monkeypatch):
    import deep_pm.usability_accounting as module
    rows=[dict(detection_ids=[str(i)]) for i in range(4)]
    with gzip.open(tmp_path/module.STANDARD,'wt') as f:json.dump(rows,f)
    monkeypatch.setattr(module,'verify_directory',lambda *a:None)
    entry=dict(path=str(tmp_path),moves={1:None,2:'new_core',3:'old_core'})
    result=retained(entry,'origin',{'origin','old_core'})
    assert set(result)=={key(rows[0]),key(rows[3])}
    assert retained(None,'missing',{'origin'})=={}
