import json
from deep_pm.io import digest
from deep_pm.production_dispatch import run


def test_dispatch_serial_tail_uses_one_node_and_resumes_without_duplicate(tmp_path,monkeypatch):
    ref=tmp_path/'reference';ref.mkdir()
    plan=ref/'private_continuation_reference_plan.json';plan.write_text('{}')
    (ref/'v19_reference_plan_summary.json').write_text(json.dumps(dict(plan_sha256=digest(plan),
        forecast_cpu_node_hours=20,budget_cpu_node_hours=48,
        physical_partitions=dict(a=dict(new_cores=3,replayed_cores=2),b=dict(new_cores=1,replayed_cores=0)))))
    commands=[]
    def submit(command,text):
        commands.append(command);return str(100+len(commands))+'\n'
    monkeypatch.setattr('deep_pm.production_dispatch.subprocess.check_output',submit)
    args=(tmp_path,tmp_path,'account','regular','python','runtime','window')
    run(*args);n=len(commands);run(*args);assert len(commands)==n
    state=json.loads((tmp_path/'private_production_jobs.json').read_text())['jobs']
    assert state['science-merge']['nodes']==state['science-quality']['nodes']==1
    assert state['science-merge']['dependencies']==[state['science-partitions']['job']]
    assert state['science-quality']['dependencies']==[state['science-merge']['job']]
    assert sum(state[k]['nodes']*state[k]['requested_wall_minutes'] for k in state if k.startswith('science-'))==180
