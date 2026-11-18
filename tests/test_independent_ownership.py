import json
from unittest.mock import patch
from deep_pm.independent_ownership import audit


def test_sealed_catalogues_are_not_opened_even_for_geometry(tmp_path):
    (tmp_path/'tasks.tsv').write_text('held_00\tnorth_two\n')
    protocol=tmp_path/'protocol.toml';protocol.write_text('final_test_box="north_two"\n')
    output=tmp_path/'summary.json'
    with patch.dict('os.environ',SLURM_JOB_ID='test'):
        audit(tmp_path,protocol,output)
    result=json.loads(output.read_text())
    assert result['test_results_read'] is False
    assert result['catalogue_count']==0 and result['core_count']==0
