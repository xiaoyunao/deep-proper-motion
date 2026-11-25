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


def test_spectral_identity_survives_float_roundtrip_and_rejects_competitors():
    import pandas as pd
    from deep_pm.production_replay_readout import spectral_identity_mask

    spectra = pd.DataFrame({'ra':[211.36370000000003,212.0,213.0],
                            'dec':[28.5344]*3,'origin':['DR16Q']*3})
    recovered = {('DR16Q',211.3637,28.5344),('other',212.0,28.5344)}
    mask,audit = spectral_identity_mask(spectra,recovered)
    assert mask.tolist()==[True,False,False]
    assert audit['exact_float_matches']==0
    assert audit['unique_matches']==1
    assert audit['maximum_accepted_separation_mas']<1.e-5

    competitors = pd.DataFrame({'ra':[211.3637,211.36370000001],
                                 'dec':[28.5344]*2,'origin':['DR16Q']*2})
    mask,audit=spectral_identity_mask(competitors,recovered)
    assert not mask.any()
    assert audit['ambiguous_reference_positions']==1
