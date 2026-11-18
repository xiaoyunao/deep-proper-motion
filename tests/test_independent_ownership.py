import json
from unittest.mock import patch
from deep_pm.independent_ownership import audit
from deep_pm.io import digest


def test_sealed_catalogues_are_not_opened_even_for_geometry(tmp_path):
    (tmp_path/'tasks.tsv').write_text('held_00\tnorth_two\n')
    protocol=tmp_path/'protocol.toml';protocol.write_text('final_test_box="north_two"\n')
    output=tmp_path/'summary.json'
    with patch.dict('os.environ',SLURM_JOB_ID='test'):
        audit(tmp_path,protocol,output)
    result=json.loads(output.read_text())
    assert result['test_results_read'] is False
    assert result['catalogue_count']==0 and result['core_count']==0


def test_completed_zero_source_core_keeps_numeric_coordinate_schema(tmp_path):
    import pandas as pd
    (tmp_path/'tasks.tsv').write_text('empty\ta\n')
    protocol=tmp_path/'protocol.toml';protocol.write_text('final_test_box="north_two"\n')
    (tmp_path/'batch-a.toml').write_text('projection_center_deg=[137.0,0.0]\n[fields.empty]\nra=[132.0,133.0]\ndec=[0.0,1.0]\n')
    graph=tmp_path/'shared/a/private_c0.npz';graph.parent.mkdir(parents=True);graph.write_bytes(b'fixture')
    work=tmp_path/'outputs/empty';work.mkdir(parents=True)
    (work/'continuous_core_summary.json').write_text(json.dumps(dict(shared_solution_sha256=digest(graph),owned_w0=0,reference_gauge='0')))
    pd.DataFrame(columns=['ra','dec','pm_east','pm_north']).to_csv(work/'private_catalogue.csv.gz',index=False)
    output=tmp_path/'summary.json'
    with patch.dict('os.environ',SLURM_JOB_ID='test'),patch('deep_pm.independent_ownership.verify_directory',return_value=work/'continuous_core_summary.json'):
        audit(tmp_path,protocol,output)
    result=json.loads(output.read_text())
    assert result['catalogue_count']==0 and result['core_count']==1
    assert result['outside_common_epoch_owner']==0
