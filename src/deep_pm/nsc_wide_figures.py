"""Make spatial quality maps from deidentified wide production statistics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


def science_value(core, method, key, axis=None):
    sample = core['common_r21']
    if sample['n'] < 5:
        return None
    value = sample['methods'][method][key]
    return value if axis is None else value[axis]


def gain(core):
    ref = science_value(core, 'Xiao', 'radial_abs68')
    current = science_value(core, 'all_nsc', 'radial_abs68')
    return 100 * (1-current/ref) if ref and current is not None else None


def save_core_map(cores, panels, title, output):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    areas = [r['core_bounds_deg'] for r in cores]
    lon = min(z['ra_bounds_deg'][0] for z in areas), max(z['ra_bounds_deg'][1] for z in areas)
    lat = min(z['dec_bounds_deg'][0] for z in areas), max(z['dec_bounds_deg'][1] for z in areas)
    for ax,(name,values,palette,normal) in zip(axes.flat,panels):
        cmap = plt.get_cmap(palette)
        for row, bounds in zip(cores, areas):
            value = values(row)
            x, y = bounds['ra_bounds_deg'], bounds['dec_bounds_deg']
            color = '#e2e2e2' if value is None or not np.isfinite(value) else cmap(normal(value))
            ax.add_patch(Rectangle((x[0], y[0]), np.ptp(x), np.ptp(y),
                                   facecolor=color, edgecolor='white', lw=.3))
        ax.set(xlim=lon, ylim=lat, xlabel='ICRS RA [deg]', ylabel='ICRS Dec [deg]', title=name)
        figure.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=normal), ax=ax, shrink=.78)
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def map_figure(data, output):
    cores = list(data['cores'].values())
    if not cores:
        raise ValueError('No complete processing cores in the science summary')
    density = lambda row: row['research_fits']/row['area_sqdeg']
    upper = max(1, np.quantile([density(r) for r in cores], .95))
    panels = (
        ('Research fits per deg²', density, 'viridis', Normalize(0, upper)),
        ('P68 improvement over Xiao2025 [%]', gain, 'RdBu', TwoSlopeNorm(0, -35, 35)),
        ('Median north motion [mas/yr]', lambda r: science_value(r,'all_nsc','median',1),
         'RdBu_r', TwoSlopeNorm(0, -4, 4)),
        ('North formal 1σ coverage [%]',
         lambda r: None if r['common_r21']['n']<5 else
         100*science_value(r,'all_nsc','fraction_within_1sigma',1),
         'PuBu', Normalize(0, 100)),
    )
    title = (f'Wide C0/W0 research catalogue · {len(cores)} processed cores\n'
             'QSO panels use common r=21–22 sources. Grey shows fewer than five objects.')
    save_core_map(cores, panels, title, output)


def support_figure(data, output):
    """Map sampling flags among the final owned full-band fits."""
    cores = list(data['cores'].values())
    if not cores:
        raise ValueError('No complete processing cores in the science summary')

    def share(row, column, key):
        if row['research_fits']:
            return 100 * row[column].get(key, 0) / row['research_fits']
        return None

    no_r = lambda row: share(row, 'source_classes', 'non_r_anchor')
    no_r_max = max((no_r(row) for row in cores if row['research_fits']), default=1)
    panels = (
        ('Well-sampled r-anchored fits [%]',
         lambda row: share(row, 'observing_support_counts', 'r_anchor_well_sampled'),
         'viridis', Normalize(0, 100)),
        ('One group supplies ≥80% of slope information [%]',
         lambda row: share(row, 'quality_flag_counts', 'one_group_supplies_at_least_80pct_leverage'),
         'magma', Normalize(0, 100)),
        ('Fewer than three NSC seasons [%]',
         lambda row: share(row, 'quality_flag_counts', 'fewer_than_three_nsc_seasons'),
         'magma', Normalize(0, 100)),
        ('Fits without an NSC r anchor [%]', no_r,
         'magma', PowerNorm(gamma=.35, vmin=0, vmax=max(1, np.ceil(no_r_max)))),
    )
    title = (f'Observing support · {data["research_fits"]:,} full-band fits\n'
             'Each panel shows a fraction of all owned fits in that core.')
    save_core_map(cores, panels, title, output)


def condition_figure(data, output):
    """Compare precision and calibration on identical spectral objects in each row."""
    cohorts = data['aggregate']['known_stationary']
    instruments = sorted({
        key.split('__instrument_', 1)[1]
        for key, value in cohorts.items()
        if '__instrument_' in key and value['strict_published_common']['n'] >= 100
    })
    observing_conditions = (
        ('all', 'All common QSOs'),
        ('one_sdss_run', 'One SDSS run'),
        ('multiple_sdss_runs', 'Multiple SDSS runs'),
        ('nsc_span_lt2', 'NSC span <2 yr'),
        ('nsc_span_ge2', 'NSC span ≥2 yr'),
        ('nsc_nights_le5', '≤5 NSC nights'),
        ('nsc_nights_gt5', '>5 NSC nights'),
        *((f'instrument_{name}', name) for name in instruments),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    for magnitude, (precision, zero, coverage) in zip((21, 22), axes):
        rows = []
        for condition, label in observing_conditions:
            cohort = cohorts.get(f'r{magnitude}__{condition}', {}).get('strict_published_common', {})
            if cohort.get('n', 0) >= 100:
                rows.append((label, cohort['n'], cohort['methods']))
        if not rows:
            raise ValueError(f'No supported observing conditions in r={magnitude}–{magnitude+1}')
        labels, counts, methods = zip(*rows)
        y = np.arange(len(rows))
        for key, label, marker in (('NSC','NSC DR2','x'), ('Xiao','Xiao2025','s'),
                                   ('r_physical_quality','Physical r','^'), ('all_nsc','Full NSC','o')):
            precision.plot([row[key]['radial_abs68'] for row in methods], y, marker,
                           label=label, markersize=5)
        for axis, name, marker in ((0,'East','o'), (1,'North','s')):
            zero.plot([r['all_nsc']['median'][axis] for r in methods], y, marker, label=name, markersize=5)
            coverage.plot([100*r['all_nsc']['fraction_within_1sigma'][axis] for r in methods],
                          y, marker, label=name, markersize=5)
        zero.axvline(0, color='0.5', lw=.8, ls='--')
        coverage.axvline(68.27, color='0.5', lw=.8, ls='--')
        coverage.set_xlim(0, 100)
        precision.set_yticks(y, [f'{label} ({n:,})' for label, n in zip(labels, counts)])
        for ax, xlabel in zip((precision, zero, coverage),
                             ('Absolute PM P68 [mas/yr]', 'Full NSC zero point [mas/yr]', 'Full NSC formal 1σ [%]')):
            ax.set(xlabel=xlabel, ylim=(len(rows)-.5, -.5))
            ax.grid(axis='x', alpha=.20)
            ax.legend(loc='lower center', bbox_to_anchor=(.5, 1), ncol=4 if ax is precision else 2,
                      frameon=False, fontsize=8)
        precision.set_ylabel(f'r = {magnitude}–{magnitude+1} mag')
        for ax in (zero, coverage):
            ax.set_yticks(y, [])
    figure.suptitle('Wide C0/W0 · strict common spectral QSOs\n'
                    'Parentheses give sample size. Conditions overlap. Instrument codes identify actual NSC combinations.')
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('science',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--conditions-output', type=Path)
    parser.add_argument('--support-output', type=Path)
    args=parser.parse_args()
    data = json.loads(args.science.read_text())
    map_figure(data,args.output)
    if args.conditions_output is not None:
        condition_figure(data,args.conditions_output)
    if args.support_output is not None:
        support_figure(data,args.support_output)


if __name__=='__main__':
    main()
