import numpy as np

from deep_pm.full_coordinate_catalogue import owners,member_key


def test_common_epoch_rehomes_boundary_without_changing_physical_rows():
    specs={'a':dict(ra=[359,360],dec=[0,1]),'b':dict(ra=[0,1],dec=[0,1])}
    rows=[dict(position_deg=[359.9,.5],detection_ids=['x','y']),dict(position_deg=[.0001,.5],detection_ids=['z']),dict(position_deg=[2,.5],detection_ids=['w'])]
    before=[member_key(r['detection_ids']) for r in rows]
    assert owners(rows,'a',specs)==['a','b',None]
    assert [member_key(r['detection_ids']) for r in rows]==before
    assert member_key(['x','y'])==member_key(['y','x'])


def test_physical_claim_quarantine_preserves_unaffected_sources(tmp_path,monkeypatch):
    import pandas as pd
    from deep_pm.full_coordinate_catalogue import close
    from deep_pm.io import save_json,digest
    from deep_pm.blind_catalogue_quality import read
    monkeypatch.setenv('SLURM_JOB_ID','test')
    root=tmp_path;registration=root/'registration';registration.mkdir();dest=root/'catalogue/index/a';dest.mkdir(parents=True)
    save_json(registration/'private_full_coordinate_plan.json',dict(cores=[dict(core='a',spec=dict(ra=[0,1],dec=[0,1]))]))
    pd.DataFrame(dict(src=[0,1,2],owner=['a','a','a'],member_sha256=['first','second','third'],pixel128=[4]*3,kind=['r_existing']*3)).to_parquet(dest/'science.parquet')
    pd.DataFrame(dict(src=[0,0,1,1,2,2],detection_id=['shared','a','shared','b','c','d'])).to_parquet(dest/'claims.parquet')
    pd.DataFrame(dict(src=[4],member_sha256=['third'],n_members=[2])).to_parquet(dest/'old.parquet')
    save_json(dest/'private_index_receipt.json',dict(files={name:digest(dest/name) for name in ('science.parquet','claims.parquet','old.parquet')},origin_science=3))
    close(root)
    out=read(root/'catalogue/v22_catalogue_ownership_summary.json')
    assert out['competing_source_quarantine']==2 and out['retained_common_epoch_solutions']==1
    assert out['exact_physical_members_common']==1 and out['shared_detection_claims']==2
    close(root) # signed bucket recovery, identical result


def test_identical_member_halo_clones_close_once_without_losing_source(tmp_path,monkeypatch):
    import pandas as pd
    from deep_pm.full_coordinate_catalogue import close
    from deep_pm.io import save_json,digest
    from deep_pm.blind_catalogue_quality import read
    monkeypatch.setenv('SLURM_JOB_ID','test')
    registration=tmp_path/'registration';registration.mkdir();d=tmp_path/'catalogue/index/a';d.mkdir(parents=True)
    save_json(registration/'private_full_coordinate_plan.json',dict(cores=[dict(core='a',spec=dict(ra=[0,1],dec=[0,1]))]))
    pd.DataFrame(dict(src=[0,1],core=['a','other'],owner=['a','a'],member_sha256=['same','same'],pixel128=[4,4],kind=['r_existing']*2,component=[0,0],pm_e=[1,1],pm_n=[2,2],se_e=[1,1],se_n=[2,2],cov_00=[100,100],maximum_group_leverage=[.2,.2],nsc_nights=[4,4],joint_span=[10,10])).to_parquet(d/'science.parquet')
    pd.DataFrame(dict(src=[0,0,1,1],detection_id=['x','y','x','y'])).to_parquet(d/'claims.parquet')
    pd.DataFrame(dict(src=[9],member_sha256=['same'],n_members=[2])).to_parquet(d/'old.parquet')
    save_json(d/'private_index_receipt.json',dict(files={n:digest(d/n) for n in ('science.parquet','claims.parquet','old.parquet')},origin_science=2))
    close(tmp_path);r=read(tmp_path/'catalogue/v22_catalogue_ownership_summary.json')
    assert r['retained_common_epoch_solutions']==1 and r['exact_identical_member_state_clones_collapsed']==1
    assert r['competing_source_quarantine']==0 and r['exact_physical_members_common']==1
