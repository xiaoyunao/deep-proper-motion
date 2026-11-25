import gzip
import json
from pathlib import Path

import pytest

from deep_pm.io import digest
from deep_pm.recover_wide_science import existing_readout
from deep_pm.resumable_wide_readout import cached_value, content_hash


def test_completed_wide_readout_rejects_changed_scope(tmp_path):
    manifest = tmp_path/'manifest.json'
    index = tmp_path/'index.json.gz'
    summary = tmp_path/'summary.json'
    manifest.write_text(json.dumps({'partitions':[{'region_by_core':{'core_a':'window_a'}}]}))
    with gzip.open(index,'wt') as stream:
        json.dump({'partition_manifest_sha256':digest(manifest),'pixels':{}},stream)
    summary.write_text(json.dumps({'cores':{'core_a':{}},'healpix_index_sha256':digest(index)}))
    assert existing_readout(manifest,summary,index)
    manifest.write_text(json.dumps({'partitions':[{'region_by_core':{'core_b':'window_b'}}]}))
    with pytest.raises(ValueError,match='different signed inputs'):
        existing_readout(manifest,summary,index)


def test_statistic_cache_rejects_changed_output(tmp_path):
    ticket = {'input_sha256':'input','implementation_sha256':'version'}
    value = {'n':25,'methods':{'all_nsc':{'radial_abs68':4.2}}}
    path = tmp_path/'sample.json'
    path.write_text(json.dumps({'ticket':ticket,'summary':value,'summary_sha256':content_hash(value)}))
    assert cached_value(tmp_path,'sample',ticket)==value
    changed = json.loads(path.read_text())
    changed['summary']['methods']['all_nsc']['radial_abs68']=1.0
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='checkpoint changed'):
        cached_value(tmp_path,'sample',ticket)
