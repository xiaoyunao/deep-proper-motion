"""Read-only, Slurm-side plotting aggregates for the frozen v1.7 catalogue."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from deep_pm.__main__ import tomllib
from deep_pm.continuous_reference import read_offsets, reference_frame, verify_directory
from deep_pm.continuous_summary import estimate
from deep_pm.io import digest, save_json
from deep_pm.nsc_ownership_overlay import STANDARD, owned_records
from deep_pm.public_summary import check_aggregate

SNAPSHOT = '2e4a04a902151ea97e2dd2c3dba647efa9f92503'
METHODS = ('NSC', 'Xiao', 'r_physical_quality', 'all_nsc')
CONFIG = dict(qso_nsides=[32, 64], qso_display_min=30, qso_public_min=5,
              magnitude_edges=np.arange(14, 26.01, .25).tolist(),
              comparison_windows=[[-50., 50., -25., 25.], [-200., 200., -100., 100.]],
              comparison_bins=140)
CASES = ('single_run', 'multiple_runs', 'extra_band', 'downweighted', 'high_leverage')


def read(path):
    return json.loads(path.read_text())


def export(path, value):
    check_aggregate(value)
    save_json(path, value)


class FrozenCatalogue:
    def __init__(self, directory):
        self.path = directory
        self.receipt = read(directory/'continuation_science_receipt.json')
        self.science = directory/'continuation_production_science_summary.json'
        if digest(self.science) != self.receipt['combined_science_sha256']:
            raise ValueError('The frozen science receipt changed')
        self.summary = read(self.science)
        index = directory/'private_nside128_index.json.gz'
        manifest = directory/'private_continuation_manifest.json'
        if digest(index) != self.summary['healpix_index_sha256']:
            raise ValueError('The final owner index changed')
        with gzip.open(index, 'rt') as stream:
            index_header = json.load(stream)
        if (index_header['partition_manifest_sha256'] != digest(manifest)
                or index_header['nside'] != 128 or index_header['ordering'] != 'RING'):
            raise ValueError('The final owner index belongs to another plan')
        self.pixel_counts = {int(k): len(v) for k, v in index_header['pixels'].items()}
        assert sum(self.pixel_counts.values()) == self.summary['research_fits']
        self.partitions = read(manifest)['partitions']
        self.parts_by_core = {}
        for part in self.partitions:
            owned = Path(part['owned'])
            stage = verify_directory(owned, ('continuous_science_summary.json', 'private_owner_overlay.json', 'fits.json'))
            if digest(owned/'continuous_science_summary.json') != self.receipt['graphs'][part['label']]['science_receipt_sha256']:
                raise ValueError('The archived ownership solution changed')
            part['folders'] = {k: Path(v) for k, v in read(owned/'fits.json').items()}
            part['specs'] = tomllib.loads(Path(part['geometry']).read_text())['fields']
            part['stage_sha256'] = digest(stage)
            for core in part['region_by_core']:
                if core in self.parts_by_core:
                    raise ValueError('Two owners for an archived core')
                self.parts_by_core[core] = part
        assert self.parts_by_core.keys() == self.summary['cores'].keys()


def metrics(vectors):
    med = np.median(vectors, axis=0)
    nmad = 1.4826 * np.median(np.abs(vectors - med), axis=0)
    return dict(p68=np.quantile(np.linalg.norm(vectors, axis=-1), .68, axis=0), median=med, nmad=nmad)


def coverage(vectors, errors):
    z = vectors / errors
    centered = (vectors-np.median(vectors, axis=0))/errors
    x = np.linspace(0, 4.5, 181)
    edges = np.linspace(-5, 5, 101)
    answer = dict(n=len(z), thresholds=x, density_edges=edges)
    for tag, v in [('original', z), ('median_removed', centered)]:
        answer[tag] = dict(cdf=[np.searchsorted(np.sort(abs(axis)), x, side='right')/len(v) for axis in v.T],
                           density_counts=[np.histogram(axis, edges)[0] for axis in v.T],
                           beyond_display=[int(np.count_nonzero(abs(axis)>5)) for axis in v.T])
    return answer


def qso_statistics(data, output):
    positions, rows = [], []
    for part in data.partitions:
        owned = Path(part['owned'])
        pool_file = Path(part['private_pool'])
        pool = read(pool_file)
        frozen = read(owned/'continuous_science_summary.json')
        if digest(owned.parent/'science_preownership_summary.json') != pool['science_summary_sha256']:
            raise ValueError('Unexpected archived spectrum pool signature')
        if pool['fit_receipt_sha256'] != frozen['fit_receipt_sha256']:
            raise ValueError('Spectrum pool and owner receipts disagree')
        by_core = defaultdict(list)
        for record in pool['known']:
            by_core[record['core']].append(record)
        for core, known in by_core.items():
            folder = part['folders'][core]
            receipt = verify_directory(folder, ('private_known_qso_blocks.json',))
            if digest(receipt) != pool['fit_receipt_sha256'][core]:
                raise ValueError('The frozen QSO route belongs to another fit')
            raw = read(folder/'private_known_qso_blocks.json')
            lookup = {method: {r['validation_local_index']: r for r in raw[method]} for method in raw}
            indices = sorted(set.intersection(*(set(v) for v in lookup.values())))
            if len(indices) != len(known):
                raise ValueError('Frozen common-object order changed')
            for local, record in zip(indices, known):
                if not set(METHODS) <= record['values'].keys():
                    continue
                all_nsc = lookup['all_nsc'][local]
                np.testing.assert_allclose(all_nsc['motion_icrs']['C0'], record['values']['all_nsc'][0], atol=1e-8, rtol=1e-8)
                assert record['joint_span']==all_nsc['joint_span_years']
                positions.append(all_nsc['position_deg'])
                rows.append(record)
    sky = np.asarray(positions)
    mags = np.array([r['magnitude'] for r in rows])
    values = np.asarray([[r['values'][m][0] for m in METHODS] for r in rows])
    std = np.asarray([r['values']['all_nsc'][1] for r in rows])
    out = dict(snapshot=SNAPSHOT, science_sha256=digest(data.science), config=CONFIG,
               source_counts=data.pixel_counts, bands={}, slurm_job=os.environ['SLURM_JOB_ID'])
    pv = np.array(list(data.pixel_counts))
    lon, lat = hp.pix2ang(128,pv,lonlat=True)
    qso_ref=data.summary['aggregate']['known_stationary']
    for low in (21,22):
        mask=(mags>=low)&(mags<low+1)
        published=qso_ref[f'r{low}__all']['strict_published_common']
        result=metrics(values[mask])
        assert np.count_nonzero(mask)==published['n']
        for j, key in enumerate(METHODS):
            method=published['methods'][key]
            np.testing.assert_allclose(result['p68'][j], method['radial_abs68'], rtol=1e-13)
            np.testing.assert_allclose(result['median'][j], method['median'], atol=1e-13)
            np.testing.assert_allclose(result['nmad'][j], method['nmad'], atol=1e-10)
        band=dict(n=int(mask.sum()), all=result, halves=[], spatial={}, conditions={}, coverage={})
        for lower in (low, low+.5):
            sub=(mags>=lower)&(mags<lower+.5)
            band['halves'].append(dict(lower=lower,n=int(sub.sum()),**metrics(values[sub])))
        assert sum(d['n'] for d in band['halves'])==band['n']
        for nside in CONFIG['qso_nsides']:
            pixels=hp.ang2pix(nside,*sky[mask].T,lonlat=True)
            vectors=values[mask]
            work=defaultdict(list)
            for j, pixel in enumerate(pixels):
                work[int(pixel)].append(j)
            support={int(p) for p in hp.ang2pix(nside,lon,lat,lonlat=True)}
            cells={}
            for pix in sorted(support|work.keys()):
                idx=work.get(pix, [])
                n=len(idx)
                cell=dict(n=n, has_owned_sources=pix in support, display=n>=CONFIG['qso_display_min'])
                if n>=CONFIG['qso_public_min']:
                    result=metrics(vectors[idx])
                    cell.update(result)
                    cell['gain']=100*(1-result['p68'][-1]/result['p68'][:-1])
                    cell['zero_change']=result['median'][-1]-result['median'][-2]
                cells[pix]=cell
            assert sum(c['n'] for c in cells.values())==band['n']
            band['spatial'][str(nside)]=cells
        in_band=[r for j,r in enumerate(rows) if mask[j]]
        points=np.array([r['shape_class']=='pointlike' for r in in_band])
        band['pointlike']=dict(n=int(points.sum()),**metrics(values[mask][points]))
        joint=np.array([r['joint_span'] for r in in_band])
        for tag, sub in [('all', np.ones(len(joint), dtype=bool)),
                         ('joint_lt12',joint<12), ('joint_ge12',joint>=12)]:
            band['coverage'][tag]=coverage(values[mask][sub,-1],std[mask][sub])
            for axis in range(2):
                if tag=='all':
                    actual=band['coverage'][tag]['original']['cdf'][axis]
                    np.testing.assert_allclose(actual[[40,80]], [published['methods']['all_nsc'][f'fraction_within_{n}sigma'][axis] for n in (1,2)])
            if tag!='all':
                block=[r for r, good in zip(in_band, sub) if good]
                shared=[dict(exposures=r['exposures'], nsc_epoch_days=r['nsc_epoch_days'],
                             motion={m:r['values'][m][0] for m in ('Xiao','all_nsc')}) for r in block]
                sensitivity=estimate(shared,('Xiao','all_nsc'),9137,candidate='all_nsc')['paired_radial_gain']['Xiao']['visit_weighted']
                band['conditions'][tag]=dict(n=len(block), **metrics(values[mask][sub]), sensitivity=sensitivity)
        out['bands'][str(low)]=band
    export(output/'qso_plot_summary.json', out)


@lru_cache(maxsize=4)
def xiao_catalogue(path):
    table = pd.read_csv(path, dtype={'objID':str})
    keep = table.PM_SOURCE.str.strip().eq('DESI')
    numeric = table[['PMRA','PMDEC','PMRA_ERR','PMDEC_ERR']].to_numpy()
    keep &= np.isfinite(numeric).all(axis=1)&(numeric[:,2:]>0).all(axis=1)&(np.abs(numeric[:,:2])<1e5).all(axis=1)
    selected = table.loc[keep, ['objID','PMRA','PMDEC']].drop_duplicates()
    counts = selected.objID.value_counts()
    return {k:(x,y) for k,x,y in selected.itertuples(index=False,name=None) if counts[k]==1}, set(counts[counts>1].index)


@lru_cache(maxsize=6)
def nsc_catalogue(source):
    verify_directory(source, ('official_nsc.csv','external_labels.csv'))
    p = pd.read_csv(source/'official_nsc.csv',dtype={'id':str})
    columns = ['pmra','pmdec','pmraerr','pmdecerr','deltamjd']
    vals = p[columns].to_numpy(float)
    keep = np.isfinite(vals).all(axis=1)&(vals[:,2:]>0).all(axis=1)&(np.abs(vals[:,:2])<1e5).all(axis=1)
    use = p.loc[keep,['id','pmra','pmdec']].drop_duplicates()
    counts = use.id.value_counts()
    official = {k:(x,y) for k,x,y in use.itertuples(index=False,name=None) if counts[k]==1}
    labels = pd.read_csv(source/'external_labels.csv',dtype=str).dropna().drop_duplicates()
    if labels.detection_id.duplicated().any():
        raise ValueError('A physical detection has multiple archived NSC labels')
    return official, set(counts[counts>1].index), dict(labels.itertuples(index=False,name=None))


def membership_match(record, external, stats, key):
    xiao,xiao_amb,nsc,nsc_amb,labels = external
    sdss={d.partition(':')[2] for d in record['detection_ids'] if d.startswith('SDSS:')}
    late=[d for d in record['detection_ids'] if d.startswith('NSC:')]
    archive={labels.get(d) for d in late}
    xhits = {ident for ident in sdss if ident in xiao}
    if len(xhits)>1 or sdss&xiao_amb:
        stats['Xiao_multiple_member_ids']+=1
    elif len(xhits)==1:
        ident=next(iter(xhits))
        yield 'Xiao', (ident, key, record['magnitude'], *xiao[ident], *record['pm_icrs_masyr'])
    else:
        stats['Xiao_missing_member_match']+=1
    if len(archive-{None})>1 or archive&nsc_amb:
        stats['NSC_multiple_member_ids']+=1
    elif len(archive)==1 and next(iter(archive),None) in nsc:
        ident=next(iter(archive))
        yield 'NSC', (ident, key, record['magnitude'], *nsc[ident], *record['pm_icrs_masyr'])
    else:
        stats['NSC_missing_member_match']+=1


def case_scores(record, spectral):
    info=record['information']
    mag=record['magnitude']
    if mag is None or not 20.5<=mag<22.5 or not 5<=info['detections']<=55:
        return
    run=info['sdss_physical_runs']
    group=max(info['largest_information_fraction']['physical_group'])
    weight=min(record['robust_weight'])
    candidates=(run==1 and info['nsc_seasons']>=3 and group<.65,
                run>=3 and info['joint_span_years']>=16 and group<.65,
                record['kind']=='extra_r_anchor' and len(record['bands'])>=3,
                weight<.30 and group<.70,
                group>=.85 or info['nsc_span_years']<1.2,
                spectral)
    for tag, good in zip((*CASES,'spectral_qso'),candidates):
        if good:
            yield tag, abs(mag-21.5)+.01*abs(info['detections']-25)


def catalogue_shard(data, output, index, total):
    counts=Counter()
    cases=defaultdict(list)
    histogram=defaultdict(lambda: np.zeros(len(CONFIG['magnitude_edges'])-1,dtype=int))
    pairs={'Xiao': [], 'NSC': []}
    source_pixel=Counter()
    for core in sorted(data.parts_by_core)[index::total]:
        counts['cores']+=1
        part=data.parts_by_core[core]
        external={}
        per_core=defaultdict(list)
        positions=[]
        for origin, slot, record in owned_records(Path(part['owned']),core):
            if origin not in external:
                folder=part['folders'][origin]
                source=(folder/'source_receipt.json').resolve().parent
                legacy=(folder/'legacy.csv').resolve()
                expected=read(folder/'run_manifest.json')['stages']['nsc-science-catalogue-v15'][-1]['inputs']['legacy.csv']
                if cached_digest(legacy)!=expected:
                    raise ValueError('The signed external Xiao cache changed')
                blind_receipt=verify_directory(folder,('private_blind_qso.json',))
                external[origin]=((*xiao_catalogue(legacy),*nsc_catalogue(source)),
                                  {v['source_slot'] for v in read(folder/'private_blind_qso.json')}, digest(blind_receipt))
            catalogues,spectral,_=external[origin]
            row_id=f'{origin}/{slot}'
            counts['owned']+=1
            positions.append(record['position_deg'])
            counts[f'class_{record["kind"]}']+=1
            mag=record['magnitude']
            label=record['magnitude_source']
            if mag is not None and np.isfinite(mag):
                per_core[label].append(mag)
            else:
                counts['magnitude_missing']+=1
            for method, pair in membership_match(record,catalogues,counts,row_id):
                pairs[method].append(pair)
            for tag, score in case_scores(record,slot in spectral):
                options=cases[tag]
                if len(options)<6 or score<options[-1][0]:
                    options.append((score, core, origin, slot))
                    options.sort()
                    del options[6:]
        if len(positions)!=data.summary['cores'][core]['research_fits']:
            raise ValueError('The owner ledger lost catalogue sources')
        if positions:
            source_pixel.update(map(int,hp.ang2pix(128,*np.asarray(positions).T,lonlat=True)))
        for name, mags in per_core.items():
            histogram[name]+=np.histogram(mags,CONFIG['magnitude_edges'])[0]
            counts[f'magnitude_measured_{name}']+=len(mags)
    for method, values in pairs.items():
        ids=np.asarray([r[0] for r in values],dtype='U')
        keys=np.asarray([r[1] for r in values],dtype='U')
        motions=np.asarray([r[2:] for r in values],dtype=float).reshape(-1,5)
        np.savez_compressed(output/f'private_matches_{method}_{index:03}.npz', external=ids, owner=keys, measures=motions)
    save_json(output/f'private_catalogue_shard_{index:03}.json',dict(snapshot=SNAPSHOT, stats=counts,
        magnitude_histogram=dict(histogram), source_pixel=dict(source_pixel), candidates=dict(cases), slurm_job=os.environ['SLURM_JOB_ID']))


@lru_cache(maxsize=128)
def cached_digest(path):
    return digest(path)


def compare_distribution(source):
    measurements=source[:,1:]
    sample=dict(n=len(source), panels={})
    for tag, limits in zip(('main','tail'), CONFIG['comparison_windows']):
        xmin,xmax,dmin,dmax=limits
        edges=np.linspace(xmin,xmax,CONFIG['comparison_bins']+1)
        delta_edges=np.linspace(dmin,dmax,CONFIG['comparison_bins']+1)
        profiles=np.linspace(xmin,xmax,26)
        panels=[]
        for axis in range(2):
            x=measurements[:,axis]
            y=measurements[:,axis+2]
            delta=y-x
            density=np.histogram2d(x,y,bins=[edges,edges])[0].astype(int)
            differences=np.histogram2d(x,delta,bins=[edges,delta_edges])[0].astype(int)
            lines=[]
            for a,b in zip(profiles[:-1],profiles[1:]):
                v=delta[(x>=a)&(x<b)]
                lines.append(dict(n=len(v), boundaries=[a,b], quantiles=np.quantile(v,[.16,.5,.84]) if len(v)>=20 else None))
            panels.append(dict(density=density, difference=differences, profile=lines,
                             outside_density=len(x)-int(density.sum()),outside_difference=len(x)-int(differences.sum()),
                             median=float(np.median(delta))))
        sample['panels'][tag]=dict(pm_edges=edges, delta_edges=delta_edges, axes=panels)
    return sample


def all_source_statistics(data, output, total):
    sources=Counter()
    stats=Counter()
    gauges=defaultdict(Counter)
    candidates=defaultdict(list)
    jobs=[]
    for index in range(total):
        ledger=read(output/f'private_catalogue_shard_{index:03}.json')
        assert ledger['snapshot']==SNAPSHOT
        jobs.append(ledger['slurm_job'])
        stats.update(ledger['stats'])
        sources.update({int(k):n for k,n in ledger['source_pixel'].items()})
        for name, bins in ledger['magnitude_histogram'].items():
            gauges[name].update({j:n for j,n in enumerate(bins)})
        for name, candidates_here in ledger['candidates'].items():
            candidates[name].extend(candidates_here)
    assert sources==data.pixel_counts and stats['cores']==2251 and stats['owned']==11862169
    assert stats['owned']==stats['magnitude_missing']+sum(v for k,v in stats.items() if k.startswith('magnitude_measured_'))
    out=dict(snapshot=SNAPSHOT, counts=dict(stats), source_counts=dict(sources),
             physical_r=sum(c['physical_r_fits'] for c in data.summary['cores'].values()),
             magnitude_histogram={k:[count[j] for j in range(len(CONFIG['magnitude_edges'])-1)] for k,count in gauges.items()},
             comparisons={}, shard_jobs=jobs, slurm_job=os.environ['SLURM_JOB_ID'])
    for method in ('Xiao','NSC'):
        parts=[]
        for index in range(total):
            with np.load(output/f'private_matches_{method}_{index:03}.npz') as archive:
                parts.append(tuple(archive[k] for k in ('external','owner','measures')))
        identity, owner, values=(np.concatenate(v) for v in zip(*parts))
        assert len(owner)==len(np.unique(owner))
        order=np.argsort(identity,kind='stable')
        values=values[order]
        identity=identity[order]
        unique=np.ones(len(identity),dtype=bool)
        single=identity[1:]!=identity[:-1]
        unique[1:] &= single
        unique[:-1] &= single
        lost=int(np.count_nonzero(~unique))
        for reason in ('multiple_member_ids','missing_member_match'):
            key=f'{method}_{reason}'
            out['counts'][key]=stats[key]
        assert stats[f'{method}_multiple_member_ids']+stats[f'{method}_missing_member_match']+len(identity)==stats['owned']
        selected=values[unique]
        if not len(selected):
            out['comparisons'][method]=dict(n=0)
            continue
        summary={key:compare_distribution(sub) for key,sub in
                 [('all',selected),('r21',selected[(selected[:,0]>=21)&(selected[:,0]<22)]),
                  ('r22',selected[(selected[:,0]>=22)&(selected[:,0]<23)])] if len(sub)}
        out['comparisons'][method]=dict(n=len(selected), global_conflict_excluded=lost, samples=summary)
    export(output/'catalogue_plot_summary.json',out)
    save_json(output/'private_example_candidates.json', {k:sorted(v) for k,v in candidates.items()})


def make_cases(data, output):
    from deep_pm.__main__ import years
    from deep_pm.astrometry import project
    candidates=read(output/'private_example_candidates.json')
    offsets={}
    points=[]
    selected=set()
    identities={}
    for tag in (*CASES,'spectral_qso'):
        available=[v for v in candidates.get(tag,[]) if tuple(v[2:]) not in selected]
        if not available:
            continue
        _, core, origin, slot=available[0]
        selected.add((origin,slot))
        part=data.parts_by_core[core]
        folder=part['folders'][origin]
        row=next(row for o,s,row in owned_records(Path(part['owned']),core) if o==origin and s==slot)
        src=(folder/'source_receipt.json').resolve().parent
        columns=['detection_id','band','survey','instrument','ra_deg','dec_deg','observation_time',
                 'sigma_east_mas','sigma_north_mas']
        verify_directory(src,('detections.csv',))
        frame=pd.concat([chunk[chunk.detection_id.isin(row['detection_ids'])] for chunk in
                         pd.read_csv(src/'detections.csv',usecols=columns,chunksize=200000)])
        assert len(frame)==len(row['detection_ids'])
        frame=frame.set_index('detection_id').loc[row['detection_ids']]
        graph=(folder/'reference_receipt.json').resolve().parent/'private_c0.npz'
        if cached_digest(graph)!=row['reference_model_sha256']:
            raise ValueError('The fit and C0 reference SHA differ')
        if graph not in offsets:
            offsets[graph]=(read_offsets(graph),reference_frame(graph))
        all_offsets, center=offsets[graph]
        raw=project(frame.ra_deg,frame.dec_deg,center)
        corrected=raw-np.asarray([all_offsets[g][0] for g in row['physical_groups']])
        state=np.asarray(row['tangent_state']).reshape(2,2)
        base=state[:,0]
        epoch=row['reference_epoch']
        times=years(frame.observation_time.to_numpy())
        config=folder/'config_snapshot.toml'
        if digest(config)!=read(folder/'run_manifest.json')['config_sha256']:
            raise ValueError('The source position error configuration changed')
        floor=tomllib.loads(config.read_text())['position_floor_mas']
        sigma=np.hypot(frame[['sigma_east_mas','sigma_north_mas']].to_numpy(),floor)
        assert len(row['robust_weight'])==len(times)
        prior_r=None
        if row['kind']=='r_existing':
            verify_directory(folder,('private_r_control.json.gz',))
            with gzip.open(folder/'private_r_control.json.gz','rt') as f:
                rank=[]
                rids=set(row['detection_ids'])
                for old in json.load(f):
                    old_ids=set(old['detection_ids'])
                    fraction=len(rids & old_ids)/len(old_ids)
                    if fraction>=.6:
                        rank.append((fraction,old))
                if len(rank)==1:
                    old=rank[0][1]
                    old_state=np.asarray(old['tangent_state']).reshape(2,2)
                    prior_r=dict(offset=old_state[:,0]-base,slope=old_state[:,1])
        cov=np.asarray(row['reference_tangent_covariance'])+np.asarray(row['independent_tangent_covariance'])
        alias=f'Example {len(points)+1:02}'
        identities[alias]=dict(origin=origin, slot=slot)
        points.append(dict(alias=alias,example_type=tag,magnitude=row['magnitude'],
            magnitude_source=row['magnitude_source'],bands=row['bands'],instruments=row['instruments'],
            reference_age=epoch, time_years=times, shift_mas=corrected-base, before_position_calibration_mas=raw-base if tag=='single_run' else None,
            w0_sigma_mas=sigma, weights=row['robust_weight'], published_icrs_motion=row['pm_icrs_masyr'],
            conditional_pm_error=row['pm_error_masyr'], display_slope=state[:,1], display_covariance=cov,
            facility=frame.survey.to_list(),passband=frame.band.to_list(),
            camera=frame.instrument.fillna('unknown').to_list(),legacy_r=prior_r,
            sampling={k:row['information'][k] for k in ('sdss_physical_runs','nsc_nights','nsc_seasons','joint_span_years','nsc_span_years')}))
    # The diagnostic trajectories are anonymized but still row-level in time.
    # Keep the numerical payload and the identity mapping under PScratch.
    save_json(output/'private_example_figures_data.json',points)
    save_json(output/'private_example_identity.json',identities)


def xiao_epoch_index(data, output, catalogue):
    from astropy.io import fits
    from astropy.time import Time
    source_stamp=catalogue.stat()
    ns=32
    owned=np.array(list(data.pixel_counts))
    occupied=np.unique(hp.ang2pix(ns,*hp.pix2ang(128,owned,lonlat=True),lonlat=True))
    nearby=np.unique(np.r_[occupied,hp.get_all_neighbours(ns,occupied).ravel()])
    support=np.zeros(hp.nside2npix(ns),bool)
    support[nearby[nearby>=0]]=True
    pieces=[]
    counts=Counter()
    with fits.open(catalogue,memmap=True,character_as_bytes=True) as hdus:
        table=hdus[1].data
        names={n.casefold():n for n in table.names}
        keys=('ra','dec','pm_source','pmra','pmdec','pmra_err','pmdec_err','mjd_desi','mjd_sdss','time')
        if not set(keys)<=names.keys():
            raise ValueError('Xiao archive lacks the published DESI mean epoch or motion fields')
        for start in range(0,len(table),1_000_000):
            stop=min(start+1_000_000,len(table))
            ra=table[names['ra']][start:stop]
            dec=table[names['dec']][start:stop]
            keep=np.isfinite(ra)&np.isfinite(dec)&(abs(dec)<=90)
            ij=np.flatnonzero(keep)
            ix=ij[support[hp.ang2pix(ns,ra[ij]%360,dec[ij],lonlat=True)]]
            ix=ix[np.char.strip(table[names['pm_source']][start:stop][ix])==b'DESI']
            counts['published_desi_in_spatial_halo']+=len(ix)
            ix=ix[np.all(np.column_stack([np.isfinite(table[names[k]][start:stop][ix])
                                         for k in ('pmra','pmdec','pmra_err','pmdec_err')]),axis=1)]
            vr=np.array([table[names[k]][start:stop][ix] for k in ('pmra','pmdec')]).T
            err=np.array([table[names[k]][start:stop][ix] for k in ('pmra_err','pmdec_err')]).T
            good=(np.abs(vr)<1e5).all(axis=1)&(err>0).all(axis=1)
            ix=ix[good]; vr=vr[good]
            stamp=np.asarray(table[names['mjd_desi']][start:stop][ix],float)
            # The historical FITS calls this MJD_DESI, but its actual values
            # are decimal years. Verify against the independent span column.
            decimal=np.isfinite(stamp)&(stamp>=2000)&(stamp<=2040)
            mjd=np.where(stamp>2_400_000,stamp-2_400_000.5,stamp)
            day=np.isfinite(mjd)&(mjd>=53000)&(mjd<=65000)
            counts['epoch_decimal_year']+=int(decimal.sum())
            counts['epoch_julian_date_or_mjd']+=int(day.sum())
            counts['valid_motion_missing_desi_epoch']+=int(np.count_nonzero(~(decimal|day)))
            epoch_year=stamp.copy()
            if day.any():
                epoch_year[day]=Time(mjd[day],format='mjd',scale='utc').tt.jyear
            actual=decimal|day
            ix=ix[actual];vr=vr[actual];epoch_year=epoch_year[actual]
            if not len(ix):
                continue
            sdss_mjd=table[names['mjd_sdss']][start:stop][ix]
            stated=table[names['time']][start:stop][ix]
            reconstructed=epoch_year-(2000+(sdss_mjd-51544.5)/365.25)
            counts['epoch_span_agrees_within_0p02yr']+=int(np.count_nonzero(np.abs(reconstructed-stated)<.02))
            counts['propagated_with_epoch']+=len(ix)
            pa,pd=np.deg2rad(ra[ix]),np.deg2rad(dec[ix])
            position=hp.ang2vec(ra[ix]%360,dec[ix],lonlat=True)
            east=np.column_stack((-np.sin(pa),np.cos(pa),np.zeros(len(ix))))
            north=np.column_stack((-np.cos(pa)*np.sin(pd),-np.sin(pa)*np.sin(pd),np.cos(pd)))
            change=np.deg2rad(1/3_600_000)*(2015.5-epoch_year)
            xyz=position+change[:,None]*(vr[:,:1]*east+vr[:,1:]*north)
            xyz/=np.linalg.norm(xyz,axis=1)[:,None]
            pixels=hp.vec2pix(ns,*xyz.T)
            pieces.append((pixels,ix.astype(np.int64)+start,xyz,vr))
    last=catalogue.stat()
    if (source_stamp.st_size,source_stamp.st_mtime_ns)!=(last.st_size,last.st_mtime_ns):
        raise ValueError('The historical Xiao FITS changed during the read')
    pixels, rows, xyz, pm=(np.concatenate(x) for x in zip(*pieces))
    order=np.argsort(pixels,kind='stable')
    np.savez(output/'private_xiao_epoch_index.npz',pix=pixels[order], external=rows[order], xyz=xyz[order], pm=pm[order])
    save_json(output/'xiao_epoch_index_receipt.json',dict(counts=counts,nside=ns, catalogue_bytes=last.st_size,
        output_sha256=digest(output/'private_xiao_epoch_index.npz'), slurm_job=os.environ['SLURM_JOB_ID']))


def xiao_positional_shard(data, output, index, total):
    from scipy.spatial import cKDTree
    receipt=read(output/'xiao_epoch_index_receipt.json')
    archive=output/'private_xiao_epoch_index.npz'
    if digest(archive)!=receipt['output_sha256']:
        raise ValueError('The propagated Xiao catalogue has changed')
    with np.load(archive) as stored:
        pix, ident, xyz, pm=(stored[k] for k in ('pix','external','xyz','pm'))
    radius=2*np.sin(np.deg2rad(1/3600)/2)
    ns=receipt['nside']
    tally=Counter()
    matches=[]
    for core in sorted(data.parts_by_core)[index::total]:
        part=data.parts_by_core[core]
        owned=list(owned_records(Path(part['owned']),core))
        tally['owned']+=len(owned)
        if not owned:
            continue
        sky=np.asarray([r['position_deg'] for _,_,r in owned])
        points=hp.ang2vec(*sky.T,lonlat=True)
        local=np.unique(hp.vec2pix(ns,*points.T))
        near=np.unique(np.r_[local,hp.get_all_neighbours(ns,local).ravel()])
        spans=[(a,b) for a,b in zip(np.searchsorted(pix,near,'left'),np.searchsorted(pix,near,'right')) if b>a]
        ids=np.concatenate([np.arange(a,b) for a,b in spans]) if spans else np.array([],dtype=int)
        targets=xyz[ids]
        # Every eligible unit-vector match must be within this Cartesian box.
        near_core=((targets>=points.min(axis=0)-radius)&(targets<=points.max(axis=0)+radius)).all(axis=1)
        ids=ids[near_core]
        candidates=cKDTree(targets[near_core]).query_ball_point(points,radius)
        # Include edges from ambiguous targets in global external uniqueness.
        # This rejects a selected edge if its counterpart has another candidate.
        for j, local_ids in enumerate(candidates):
            origin,slot,row=owned[j]
            m=row['magnitude']
            tally['no_position_candidate' if not local_ids else 'multiple_position_candidates' if len(local_ids)>1 else 'one_position_candidate']+=1
            for k in local_ids:
                external=ids[k]
                matches.append((ident[external], len(local_ids)==1, m, *pm[external], *row['pm_icrs_masyr']))
    values=np.asarray([x[2:] for x in matches],dtype=float).reshape(-1,5)
    np.savez_compressed(output/f'private_xiao_position_{index:03}.npz',
        external=np.array([x[0] for x in matches],dtype=np.int64),
        unique_from_owner=np.array([x[1] for x in matches],dtype=bool), measures=values)
    save_json(output/f'xiao_position_shard_{index:03}.json',tally)


def xiao_positional_summary(data, output, total):
    tally=Counter()
    parts=[]
    for index in range(total):
        tally.update(read(output/f'xiao_position_shard_{index:03}.json'))
        with np.load(output/f'private_xiao_position_{index:03}.npz') as payload:
            parts.append(tuple(payload[k] for k in ('external','unique_from_owner','measures')))
    ids,allowed,measures=(np.concatenate(v) for v in zip(*parts))
    order=np.argsort(ids,kind='stable')
    ids=ids[order]
    if len(ids)==0:
        raise ValueError('No positional matches can be verified at the common epoch')
    single=ids[1:]!=ids[:-1]
    unique=allowed[order].copy()
    unique[1:] &= single
    unique[:-1] &= single
    selected=measures[order][unique]
    assert sum(tally[k] for k in ('no_position_candidate','multiple_position_candidates','one_position_candidate'))==tally['owned']==data.summary['research_fits']
    summary={key:compare_distribution(sub) for key,sub in
             [('all',selected),('r21',selected[(selected[:,0]>=21)&(selected[:,0]<22)]),
              ('r22',selected[(selected[:,0]>=22)&(selected[:,0]<23)])] if len(sub)}
    export(output/'xiao_positional_summary.json',dict(n=len(selected), global_conflict_excluded=tally['one_position_candidate']-len(selected),
        radius_arcsec=1, comparison_epoch_jyear=2015.5, candidate_counts=tally,
        epoch_index=read(output/'xiao_epoch_index_receipt.json'),samples=summary,slurm_job=os.environ['SLURM_JOB_ID']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['qso','shard','catalogue','cases','xiao-index','xiao-shard','xiao-summary'])
    p.add_argument('frozen',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--xiao-fits',type=Path)
    args=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ:
        p.error('Use a NERSC Slurm compute node')
    args.output.mkdir(exist_ok=True,parents=True)
    data=FrozenCatalogue(args.frozen)
    action={'qso': lambda: qso_statistics(data,args.output),
            'shard': lambda: catalogue_shard(data,args.output,args.index,args.shards),
            'catalogue': lambda: all_source_statistics(data,args.output,args.shards),
            'cases': lambda: make_cases(data,args.output),
            'xiao-index': lambda: xiao_epoch_index(data,args.output,args.xiao_fits),
            'xiao-shard': lambda: xiao_positional_shard(data,args.output,args.index,args.shards),
            'xiao-summary': lambda: xiao_positional_summary(data,args.output,args.shards)}
    action[args.stage]()


if __name__=='__main__':
    main()
