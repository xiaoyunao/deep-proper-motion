"""Inventory actual native survey geometry for the next wide production cohort."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
import pandas as pd

from deep_pm import access
from deep_pm.__main__ import tomllib
from deep_pm.astrometry import project,sdss_tai_to_tt,years
from deep_pm.continuous_summary import area
from deep_pm.exposure_preflight import where_box
from deep_pm.footprints import sdss_window_mask, sdss_window_selection
from deep_pm.independent_selection import intersects_previous
from deep_pm.io import digest, save_json
from deep_pm.nsc_opportunity_area import equal_area_grid,estimate,scan_members
from deep_pm.regional_preflight import CHIP_VERTICES
from deep_pm.public_summary import check_aggregate

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run geometry on a Slurm compute node')
root, window_path, past, prior = map(Path, sys.argv[1:])
summary = prior/'continuation_production_science_summary.json'
proof = json.loads((prior/'continuation_science_receipt.json').read_text())
if digest(summary) != proof['combined_science_sha256']:
    raise ValueError('Frozen catalogue scope changed')
existing = json.loads(summary.read_text())
excluded_windows = set(existing['regions'])
root.mkdir(parents=True,exist_ok=True)
old = [dict(ra=row['core_bounds_deg']['ra_bounds_deg'], dec=row['core_bounds_deg']['dec_bounds_deg'])
       for row in existing['cores'].values()]
with fits.open(window_path,memmap=True) as f:
    t=f[1].data
    v=sdss_window_mask(t,301)
    columns=['RA','DEC','RUN','CAMCOL','NODE','INCL','MU_START','MU_END','NU_START','NU_END']
    frame=pd.DataFrame({k:np.asarray(t[k][v],float) for k in columns})
    tai=np.asarray(t['TAI'][v],float)
    if tai.ndim==2:tai=tai[:,2]
    frame['epoch']=years(sdss_tai_to_tt(np.where((tai>=4e9)&(tai<6e9),tai,np.nan)))
    frame=frame.loc[np.isfinite(frame.epoch)].reset_index(drop=True)
arrays={k:frame[k].to_numpy() for k in ('RA','DEC')}
ranks=root/'sdss_windows_summary.json'
if not ranks.exists():ranks=past/'sdss_windows_summary.json'
if ranks.exists():
    ordered=[r for r in json.loads(ranks.read_text())['windows'] if r['name'] not in excluded_windows]
else:
    queues=defaultdict(list)
    for a in range(10,350,10):
        for d in range(-20,70,10):
            box=dict(ra=[a,a+10],dec=[d,d+10])
            center=(a+5,d+5)
            near=frame.loc[sdss_window_selection(arrays,center,8)]
            if len(near)<400:continue
            sky,unit=equal_area_grid(box,.33)
            xy=project(*sky.T,center)
            members,_=scan_members(near,sky,xy,center,.4)
            count=np.zeros(len(sky),int)
            for ix in members.values():count[ix]+=1
            covered=(count>0).mean()
            if covered<.50:continue
            label=f'w{a:03d}_{d+90:03d}'
            if label in excluded_windows:continue
            record=dict(name=label,footprint=box,geometry_sdss_fraction=float(covered),
                        geometry_sdss_multi_fraction=float((count>1).mean()),area_sqdeg=area(box))
            category=('south' if d<0 else 'equatorial' if d<20 else 'north')+str(a//90)
            queues[category].append(record)
    for q in queues.values():
        q.sort(key=lambda row:-(row['geometry_sdss_fraction']*row['area_sqdeg']))
    ordered=[]
    while any(queues.values()):
        for k,queue in sorted(queues.items()):
            if queue:ordered.append(queue.pop(0))
    print('SDSS_WINDOW_RANKS',len(ordered),[p['name'] for p in ordered[:24]],flush=True)
    save_json(root/'sdss_windows_summary.json',dict(windows=ordered,slurm_job=os.environ['SLURM_JOB_ID']))

def qualify(row):
    name,box=row['name'],row['footprint']
    center=tuple(np.mean(box[k]) for k in ('ra','dec'))
    work=root/name
    work.mkdir(exist_ok=True)
    output=work/'window_summary.json'
    if output.exists():return json.loads(output.read_text())
    early=frame.loc[sdss_window_selection(arrays,center,8)]
    pad=.70
    cols=['exposure','ccdnum','instrument','filter','mjd',*CHIP_VERTICES]
    old_chips = past/name/'private_chips.csv.gz'
    if old_chips.is_file():
        chips = pd.read_csv(old_chips, dtype={'exposure':str})
    else:
        chips=access.checked_spatial(access.Cache(work/'cache'),'nsc',','.join(cols),
               'nsc_dr2.chip',where_box(box,pad),box['dec'][0]-pad-1e-7,box['dec'][1]+pad+1e-7).drop_duplicates()
    if chips.duplicated(['exposure','ccdnum']).any():raise ValueError('Conflicting chip metadata')
    chips=chips.loc[chips.mjd.between(50000,62000)]
    chips.to_csv(work/'private_chips.csv.gz',index=False)
    keys={'NSC:'+str(i) for i in chips.exposure}|{f'SDSS:{int(i)}:{int(j)}' for i,j in zip(early.RUN,early.CAMCOL)}
    parents=[]
    group_keys=set()
    for a in range(box['ra'][0],box['ra'][1],2):
        for d in range(box['dec'][0],box['dec'][1],2):
            spec=dict(ra=[a,a+2],dec=[d,d+2])
            if intersects_previous(spec,old):continue
            # Conservative rectangular metadata crop includes enclosing CCDs.
            relevant=(chips[[f'vertex_ra{i}' for i in range(1,5)]].min(axis=1)<a+2)&(
              chips[[f'vertex_ra{i}' for i in range(1,5)]].max(axis=1)>a)&(
              chips[[f'vertex_dec{i}' for i in range(1,5)]].min(axis=1)<d+2)&(
              chips[[f'vertex_dec{i}' for i in range(1,5)]].max(axis=1)>d)
            cad=chips.loc[relevant]
            if cad.exposure.nunique()<2:continue
            field_center=(a+1,d+1)
            neighbours=early.loc[sdss_window_selection(early,field_center,1.5)]
            result=estimate(cad,neighbours,spec,keys,.12)
            late=result['layers']['all']['geometric_joint_opportunity_area_sqdeg']
            if late<.55*area(spec):continue
            # Conservative possible group overlap for costing. Reference sources
            # later determine which of these physical groups actually connect.
            group_keys.update('NSC:'+str(e) for e in cad.exposure)
            group_keys.update(f'SDSS:{int(r)}:{int(c)}' for r,c in zip(neighbours.RUN,neighbours.CAMCOL))
            parents.append(dict(name=f'{name}_{a-box["ra"][0]}{d-box["dec"][0]}',box=name,
                ra_bounds_deg=spec['ra'],dec_bounds_deg=spec['dec'],area_sqdeg=area(spec),
                pre_reference_opportunity_sqdeg=late,instruments=sorted(map(str,cad.instrument.dropna().unique())),
                exposures=int(cad.exposure.nunique()), possible_sdss_runs=int(neighbours.RUN.nunique()),
                nsc_window_span_years=float(np.ptp(cad.mjd)/365.25),
                nsc_window_utc_nights=len(set(np.floor(cad.mjd-.75).astype(int))),
                nsc_window_calendar_years=len(set((2000+(cad.mjd-51544.5)/365.25).astype(int)))))
    result={k:v for k,v in row.items() if k!='footprint'}
    result.update(window_bounds_deg=dict(ra_bounds_deg=box['ra'],dec_bounds_deg=box['dec']),
          selected_parents=parents,suitable_nominal_sqdeg=sum(x['area_sqdeg'] for x in parents),
          pre_reference_opportunity_sqdeg=sum(x['pre_reference_opportunity_sqdeg'] for x in parents))
    save_json(work/'private_possible_groups.json',sorted(group_keys))
    check_aggregate(result);save_json(output,result)
    return result

checked=[]
failures=[]
for start in range(0,len(ordered),4):
    with ThreadPoolExecutor(1) as pool:
        futures=[(r,pool.submit(qualify,r)) for r in ordered[start:start+4]]
        for row,future in futures:
            try:
                result=future.result()
            except Exception as exc:
                failures.append(dict(window=row['name'],error_type=type(exc).__name__))
                print('WINDOW_ERROR',row['name'],repr(exc),flush=True)
                continue
            checked.append(result)
            print('WINDOW',result['name'],result['suitable_nominal_sqdeg'],result['pre_reference_opportunity_sqdeg'],flush=True)
    result=dict(windows=checked,errors=failures,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Prospective continuation from independent native SDSS and NSC geometry. Excludes all frozen existing cores and registered wide windows. No new motion results used.')
    check_aggregate(result);save_json(root/'wide_geometry_summary.json',result)
    if sum(w['pre_reference_opportunity_sqdeg'] for w in checked)>=1250:break
if sum(w['suitable_nominal_sqdeg'] for w in checked)<500:
    raise ValueError('Less than 500 square degrees of useful new candidate geometry')
