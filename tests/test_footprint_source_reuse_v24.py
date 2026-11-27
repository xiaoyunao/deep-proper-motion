import json

from deep_pm.full_coordinate_sources import ARCHIVED_TAN,reuse_archived_source,write_rows
from deep_pm.io import digest,save_json


def test_reuse_only_same_complete_physical_model(tmp_path,monkeypatch):
    prior=tmp_path/'old/sources/test';prior.mkdir(parents=True)
    dest=tmp_path/'new/sources/test';dest.mkdir(parents=True)
    fields=('source_receipt_sha256','precision_sha256','spatial_sha256','selection_sha256','funnel_sha256','method')
    ticket=dict.fromkeys(fields,'frozen')
    ticket.update(ARCHIVED_TAN,graphs={'1':'original'})
    files=['private_standard_motion.json.gz','private_r_control.json.gz','private_fixed_blind_qso.json.gz']
    for p in files:write_rows(prior/p,[1])
    save_json(prior/'private_blind_qso.json',[2]);files.append('private_blind_qso.json')
    source=prior/'v22_source_core_summary.json'
    save_json(source,dict(ticket=ticket,private_output_sha256={p:digest(prior/p) for p in files}))
    save_json(tmp_path/'old/catalogue/index/test/private_index_receipt.json',dict(ticket=dict(source_summary_sha256=digest(source))))
    monkeypatch.setenv('SLURM_JOB_ID','test')
    entry=dict(core='test',predecessor=str(prior))
    new=dict(ticket,save_r_comparison=False,implementation_sha256='wide-capable')
    assert reuse_archived_source(entry,dict(new,graphs={'1':'different'}),dest) is None
    result=reuse_archived_source(entry,new,dest)
    assert (dest/files[0]).resolve()==prior/files[0]
    assert not (dest/files[1]).is_symlink()
    assert json.loads((dest/'private_blind_qso.json').read_text())==[]
    assert result['reused_archived_source_sha256']==digest(source)
