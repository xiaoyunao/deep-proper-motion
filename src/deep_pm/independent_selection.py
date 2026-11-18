"""Select geometrically eligible new cells without reading proper-motion results."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_summary import area
from .exposure_catalogues import inside
from .footprints import wrap
from .io import digest,save_json
from .public_summary import check_aggregate


HISTORY = ('configs/production-v05.toml','configs/regional-v07.toml',
           'configs/spatial-v08-fields.toml','configs/precision-v03.toml',
           'configs/exposure-v09-fields.toml','configs/validation-v10.toml',
           'configs/continuous-v12.toml')


def previous_footprints():
    result=[]
    for filename in HISTORY:
        config=tomllib.loads(Path(filename).read_text())
        for value in config.get('fields',config.get('regions',{})).values():
            if {'ra','dec'} <= value.keys():
                result.append(value)
    return result


def intersects_previous(box,previous):
    """Conservative spherical bounding boxes; excludes entire intersecting cells."""
    ra,dec = np.mean(box['ra']),np.mean(box['dec'])
    w,h = np.ptp(box['ra'])/2,np.ptp(box['dec'])/2
    for old in previous:
        if np.isscalar(old['ra']):
            radius=old.get('radius',0.)+.06
            latitude=min(89.,max(abs(dec),abs(old['dec']))+radius+h)
            old_w=radius/np.cos(np.deg2rad(latitude))
            if abs(wrap(ra-old['ra'])) <= w+old_w and abs(dec-old['dec']) <= h+radius:
                return True
        else:
            a,d=np.mean(old['ra']),np.mean(old['dec'])
            if abs(wrap(ra-a)) < w+np.ptp(old['ra'])/2+.06 and abs(dec-d) < h+np.ptp(old['dec'])/2+.06:
                return True
    return False


def select(old_root,new_root,catalogues,config_path,work):
    config=tomllib.loads(config_path.read_text())
    spectrum=pd.read_csv(catalogues/'spectra_union.csv',usecols=['ra','dec','reference_r'])
    old=previous_footprints()
    candidates,receipts=[],{}
    for name in config['boxes']:
        path=(old_root if name in ('a','b') else new_root)/name/'geometry_summary.json'
        geometry=json.loads(path.read_text())
        receipts[name]=dict(geometry_sha256=digest(path),instrument_band_ccd_counts=geometry['ccd_coverage_by_instrument_band'])
        for i,tile in enumerate(geometry['tiles']):
            box=tile['region']
            has_old=intersects_previous(box,old)
            on=inside(spectrum.ra,spectrum.dec,box)
            qsos=int((on&spectrum.reference_r.between(21,23,inclusive='left')).sum())
            eligible=(not has_old and tile['joint']>=config['minimum_joint_fraction']
                      and tile['joint_time']>=config['minimum_nsc_time_fraction'] and qsos>=10)
            candidates.append(dict(name=f'{name}_{i:02d}',box=name,ra_bounds_deg=box['ra'],dec_bounds_deg=box['dec'],
                area_sqdeg=area(box),overlaps_previous_science=has_old,eligible=bool(eligible),
                joint_fraction=tile['joint'],joint_timed_fraction=tile['joint_time'],
                long_sdss_repeat_fraction=tile['long_sdss_repeat'],sdss_visits=tile['sdss_visits'],
                nsc_r_visits=tile['nsc_r_visits'],spectroscopic_qso_21_23_count=qsos,
                objects=tile['objects'],approximate_nsc_measurements=tile['approximate_nsc_measurements']))
    # Interleave candidate windows; no precision measurement enters this ordering.
    queues={name:sorted([r for r in candidates if r['eligible'] and r['box']==name],key=lambda r:(-r['joint_timed_fraction'],r['name']))
            for name in config['boxes']}
    chosen=[]
    total=0.
    while any(queues.values()):
        for name in sorted(queues):
            if not queues[name]:
                continue
            value=queues[name].pop(0)
            if total+value['area_sqdeg']<=config['target_new_area_max_deg2']:
                chosen.append(value);total+=value['area_sqdeg']
    selected={r['name'] for r in chosen}
    work.mkdir(parents=True,exist_ok=True)
    for box in config['boxes']:
        center=[float(np.mean(config['boxes'][box][k])) for k in ('ra','dec')]
        lines=[f'projection_center_deg = {center}', 'halo_deg = 0.06', 'seed = 9137',
               'native_join = true','native_count_window = true','native_row_cap = 80000','download_workers = 1']
        for r in chosen:
            if r['box']==box:
                lines.extend([f"[fields.{r['name']}]",f"ra = {r['ra_bounds_deg']}",f"dec = {r['dec_bounds_deg']}"])
        (work/f'selected-{box}.toml').write_text('\n'.join(lines)+'\n')
    for row in candidates:
        row['selected_for_reference_precheck']=row['name'] in selected
    output=dict(selection_scope='Geometry and spectral positions only; no new proper-motion catalogue read.',
        candidate_receipts=receipts,candidates=candidates,provisional_new_area_sqdeg=total,
        target_area_supported=bool(total>=config['target_new_area_min_deg2']),
        selected_count=len(chosen),reference_graph_support='Must verify on first acquired batches before final field lock.',
        selected_approximate_nsc_measurements=sum(r['approximate_nsc_measurements'] for r in chosen),
        excluded_previous_cell_count=sum(r['overlaps_previous_science'] for r in candidates))
    check_aggregate(output)
    save_json(work/'independent_geometry_summary.json',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old_root','new_root','catalogues','config','work'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    select(a.old_root,a.new_root,a.catalogues,a.config,a.work)


if __name__=='__main__':
    main()
