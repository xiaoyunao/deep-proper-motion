"""Conditioned stellar attribution from saved fixed-source position diagnostics."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import numpy as np
from astropy.time import Time

from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate
from .usability_cluster import summarize


def group_statistics(pairs,metadata):
    output=dict(pairs=len(pairs),descriptive_pair_distribution=summarize([r['raw_difference'] for r in pairs]))
    nights=defaultdict(list);exposures=set()
    for r in pairs:
        m=metadata[r['physical_group']];night=Time(m['time'],format='jyear',scale='tt').utc.isot[:10]
        nights[(m['survey'],night)].append(r['raw_difference']);exposures.add(m['calibration_group'])
    means=np.asarray([np.median(v,axis=0) for v in nights.values()])
    output.update(independent_UTC_nights=len(nights),physical_calibration_groups=len(exposures),night_median_distribution=summarize(means))
    if len(means)>=3:
        rng=np.random.default_rng(210621);idx=rng.integers(0,len(means),(400,len(means)))
        output['equal_night_median_sensitivity95']=np.quantile(np.median(means[idx],axis=1),[.025,.975],axis=0).T
    return output


def run(input,output):
    p=read(input);measure=p['measurements'];pairs=p['pairs'];states=p['states'];meta={m['physical_group']:m for m in measure}
    stratified={}
    for category,fn in [('instrument_band',lambda m:m['instrument']+':'+m['band']),
                        ('UTC_year',lambda m:Time(m['time'],format='jyear',scale='tt').utc.isot[:4]),
                        ('airmass',lambda m:'unknown' if not np.isfinite(m['airmass']) else ('1_to1.3' if m['airmass']<1.3 else ('1.3_to1.7' if m['airmass']<1.7 else 'above1.7')))]:
        groups=defaultdict(list)
        for pair in pairs:groups[fn(meta[pair['physical_group']])].append(pair)
        stratified[category]={name:group_statistics(g,meta) for name,g in sorted(groups.items())}
    perstar=defaultdict(list)
    for m in measure:perstar[tuple(m['source_key'])].append(m)
    chosen=[r for r in states if 'c0_residual' in r and r.get('color') is not None and np.isfinite(r['rmag'])]
    target=[r for r in chosen if r['old_target']];field=[r for r in chosen if not r['old_target']]
    def feature(r):
        m=perstar[tuple(r['source_key'])]
        return (frozenset(v['calibration_group'] for v in m if v['survey']=='SDSS'),
                frozenset(v['band'] for v in m if v['survey']=='NSC'),
                float(np.ptp([v['time'] for v in m])),float(m[0]['crowding_mas']))
    features={tuple(r['source_key']):feature(r) for r in chosen};used=set();comparisons=[]
    for r in sorted(target,key=lambda r:r['source_key']):
        a=features[tuple(r['source_key'])];candidates=[]
        for j,s in enumerate(field):
            if j in used or abs(r['rmag']-s['rmag'])>.3 or abs(r['color']-s['color'])>.15:continue
            b=features[tuple(s['source_key'])]
            if a[0]!=b[0] or len(a[1]&b[1])/max(1,len(a[1]|b[1]))<.8 or abs(a[2]-b[2])>2:continue
            if min(a[3],b[3])<2000 or abs(np.log10(a[3]/b[3]))>.3:continue
            candidates.append((abs(r['rmag']-s['rmag'])/.3+abs(r['color']-s['color'])/.15,j))
        if not candidates:continue
        j=min(candidates)[1];used.add(j);s=field[j]
        comparisons.append(dict(target_minus_field=np.asarray(r['c0_residual'])-r['parallax_slope']-np.asarray(s['c0_residual'])+s['parallax_slope'],
            target_C0_minus_field_C0=np.asarray(r['c0_predicted_change'])-s['c0_predicted_change']))
    result=dict(input_sha256=digest(input),band_year_airmass=stratified,
        bright_optical_target_vs_field=dict(target_candidates=len(target),field_candidates=len(field),metadata_pairs=len(comparisons),
            Gaia_residual_difference_after_known_parallax_slope=summarize([r['target_minus_field'] for r in comparisons]),
            C0_response_difference=summarize([r['target_C0_minus_field_C0'] for r in comparisons]),
            scope='Fixed optical target identities and field Gaia controls; no PM selection. Same SDSS run/camcol set,>=0.8NSC band-set Jaccard, baseline within2yr, r within0.3mag, g-i within0.15mag, neighbour>=2arcsec and crowding ratio<=2. Individual Gaia PM removed, known annual-parallax slope removed. Bright overlap only: cannot certify dark21+ field or cluster membership.'),
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Position colour contrasts conditioned on actual physical group; UTC-night aggregate intervals are sensitivity to repeated measurements. No zenith direction supplied: airmass/colour dependence alone cannot identify residual DCR. No production correction is fitted.')
    check_aggregate(result);save_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
