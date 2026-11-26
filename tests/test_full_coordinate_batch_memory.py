import json
from pathlib import Path

import pandas as pd
import pytest

from deep_pm.full_coordinate_batch import DISPATCH_DEFINITION,memory_queue,physical_dispatch
from deep_pm.io import digest,save_json


def test_recovery_reuses_only_unchanged_actual_target_dispatch(tmp_path,monkeypatch):
    source=tmp_path/'input';source.mkdir();save_json(source/'run_manifest.json',dict(version=1))
    e=dict(core='a',source=str(source));folder=tmp_path/'sources/a';folder.mkdir(parents=True)
    payload=dict(source_sha256=digest(source/'run_manifest.json'),group_map_sha256='map1',components=[0,2],definition=DISPATCH_DEFINITION)
    save_json(folder/'private_dispatch.json',payload);before=(folder/'private_dispatch.json').read_bytes()
    monkeypatch.setattr(pd,'read_csv',lambda *args,**kwargs:pytest.fail('Unchanged signed dispatch must not scan the giant CSV again'))
    actual,proof,reused=physical_dispatch(e,tmp_path,None,'map1')
    assert actual==[0,2] and reused and proof==payload
    assert (folder/'private_dispatch.json').read_bytes()==before
    with pytest.raises(ValueError,match='Changed physical-dispatch'):physical_dispatch(e,tmp_path,None,'map2')
    save_json(source/'run_manifest.json',dict(version=2))
    with pytest.raises(ValueError,match='Changed physical-dispatch'):physical_dispatch(e,tmp_path,None,'map1')


def test_measured_memory_envelope_requires_sufficient_safe_source_receipts(tmp_path):
    scientific=Path(__import__('deep_pm.full_coordinate_sources',fromlist=['one']).__file__)
    entries=[]
    for i in range(50):
        source=tmp_path/f'data/{i}';source.mkdir(parents=True)
        (source/'detections.csv').write_bytes(b'normalised input')
        (source/'private_standard_motion.json.gz').write_bytes(b'old source archive')
        entries.append(dict(core=str(i),source=str(source),fit=str(source)))
        save_json(tmp_path/f'input/{i}/v22_reference_input_core_summary.json',dict(peak_rss_mib=8192))
        save_json(tmp_path/f'sources/{i}/v22_source_core_summary.json',dict(ticket=dict(implementation_sha256=digest(scientific),graphs={'0':'signed-reference'}),peak_rss_mib=1024))
        save_json(tmp_path/f'sources/{i}/private_dispatch.json',dict(components=[0]))
    plan=dict(cores=entries)
    smaller,_,audit=memory_queue(tmp_path,plan,range(49),'receipt-validated')
    assert audit['used']=='reference-input-conservative'
    selected,legacy,audit=memory_queue(tmp_path,plan,range(50),'receipt-validated')
    assert audit['used']=='receipt-validated' and all(memory<legacy[i] for i,memory,size in selected)
    path=tmp_path/'sources/49/v22_source_core_summary.json';proof=json.loads(path.read_text());proof['peak_rss_mib']=4096;save_json(path,proof)
    selected,legacy,audit=memory_queue(tmp_path,plan,range(50),'receipt-validated')
    assert audit['used']=='reference-input-conservative'
    assert all(memory==legacy[i] for i,memory,size in selected)
