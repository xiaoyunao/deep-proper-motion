import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('duckdb')
from deep_pm.full_reference_graph import close_graph,export_component
from deep_pm.io import save_json,digest


def test_global_duplicates_connectivity_and_numeric_export(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','synthetic');root=tmp_path
    (root/'registration').mkdir();entries=[]
    for core in ('a','b'):
        folder=root/'input'/core;folder.mkdir(parents=True);entries.append({'core':core})
        records=[]
        # Same physical inputs loaded twice across halos, split by object.
        for r in range(20):
            for e,group in enumerate(('NSC:1','NSC:2','SDSS:1:1')):
                records.append(dict(reference=str(r),is_holdout=r>=15,detection_id=f'd{r}_{e}',exposure_id=f'e{e}',
                    ra_deg=359.9+r/1000+e/3600000,dec_deg=.1,sigma_e=60.,sigma_n=70.,group=group,source_core=core))
        frame=pd.DataFrame(records);frame.to_parquet(folder/'reference.parquet',index=False)
        save_json(folder/'v22_reference_input_core_summary.json',dict(measurements=len(frame),parquet_sha256=digest(folder/'reference.parquet')))
    save_json(root/'registration/private_full_coordinate_plan.json',dict(cores=entries))
    close_graph(root,memory='128MB',threads=1)
    report=json.loads((root/'graph/v22_global_graph_summary.json').read_text())
    assert report['unique_measurements']==60 and report['training_measurements']==45
    assert len(report['components'])==1 and report['components'][0]['groups']==3
    export_component(root,0,memory='128MB',threads=1)
    dest=root/'reference/component-0000'
    stamp=(dest/'groups.npy').stat().st_mtime_ns
    export_component(root,0,memory='128MB',threads=1)
    assert (dest/'groups.npy').stat().st_mtime_ns==stamp
    assert np.load(dest/'groups.npy').shape==(45,)
    np.testing.assert_array_equal(np.diff(np.load(dest/'ptr.npy')),3)
    assert np.isfinite(np.load(dest/'precision.npy')).all()
    # Every object's first datum is centred, including external Arrow batches.
    values=np.load(dest/'values.npy');np.testing.assert_allclose(values[::3],0,atol=1e-10)


def test_common_hemisphere_uses_geometry_across_ra_wrap():
    from deep_pm.full_reference_graph import hemisphere_center
    from deep_pm.coordinate_metric import sky_basis
    sky=np.array([[350.,20.],[10.,25.],[170.,40.],[190.,35.]])
    center=hemisphere_center(sky)
    dot=sky_basis(*sky.T)[:,0,:]@sky_basis(*center)[0]
    assert np.min(dot)>.3


def test_hemisphere_does_not_reject_feasible_84_degree_equatorial_chart():
    from deep_pm.full_reference_graph import hemisphere_center
    from deep_pm.coordinate_metric import sky_basis
    # A cube-bounded LP can choose Dec=-45 deg and spuriously exceed 85 deg.
    sky=np.array([[84.,0.],[276.,0.]])
    center=hemisphere_center(sky)
    dots=sky_basis(*sky.T)[:,0,:]@sky_basis(*center)[0]
    np.testing.assert_allclose(np.rad2deg(np.arccos(dots)),84.,atol=1e-7)
    with pytest.raises(ValueError,match='hemisphere'):
        hemisphere_center(np.array([[0.,0.],[180.,0.]]))


def test_held_reference_transports_old_offsets_and_checks_actual_physical_members(tmp_path):
    from deep_pm.astrometry import project,unproject
    from deep_pm.footprint_held_reference_v24 import archived_positions
    base=tmp_path/"v2.4/predecessor-held"
    base.mkdir(parents=True)
    old_center=(352.,8.);new_center=(22.,4.)
    ra=np.array([359.99,.01]);dec=np.array([12.,12.])
    adjusted=unproject(project(ra,dec,old_center)-np.array([70.,-80.]),old_center)
    f=pd.DataFrame(dict(reference=['A','B'],detection_id=['d1','d2'],group=['g','h'],ra_deg=ra,dec_deg=dec))
    archived=f.assign(old_ra=adjusted[:,0],old_dec=adjusted[:,1],old_supported=[True,False])
    output=base/'held-0000.parquet';archived.to_parquet(output)
    save_json(base/'v24_previous_held_reference_summary.json',dict(components={'0':dict(sha256=digest(output))}))
    f.loc[2]=['C','d3','g',1.,13.]
    result,keep,n,_=archived_positions(tmp_path,f,new_center)
    assert n==2 and keep.tolist()==[True,False,False]
    np.testing.assert_allclose(unproject(result[keep],new_center),adjusted[:1],atol=1e-12)
    assert np.isnan(result[~keep]).all()


def test_component_reuse_requires_identity_order_positions_covariance_and_seed():
    from copy import deepcopy
    from deep_pm.footprint_component_v24 import same_archived_input
    binding={"component":{"component":2,"center_deg":[355.,0.],"measurements":120,"groups":5,"reference_sources":30},
             "files":{"precision.npy":"offset-a","private_reference_index.parquet":"identity-order"}}
    assert same_archived_input(binding,deepcopy(binding))
    for key,val in (("component",1),("center_deg",[356.,0.])):
        changed=deepcopy(binding)
        changed["component"][key]=val
        assert not same_archived_input(binding,changed)
    for key in binding["files"]:
        changed=deepcopy(binding)
        changed["files"][key]="other"
        assert not same_archived_input(binding,changed)
