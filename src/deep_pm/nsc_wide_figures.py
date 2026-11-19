"""Make spatial quality maps from deidentified wide production statistics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
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


def map_figure(data, output):
    cores = list(data['cores'].values())
    if not cores:
        raise ValueError('No complete processing cores in the science summary')
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    areas = [r['core_bounds_deg'] for r in cores]
    lon = min(z['ra_bounds_deg'][0] for z in areas), max(z['ra_bounds_deg'][1] for z in areas)
    lat = min(z['dec_bounds_deg'][0] for z in areas), max(z['dec_bounds_deg'][1] for z in areas)
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
    figure.suptitle(f'Wide C0/W0 research catalogue · {len(cores)} processed cores\n'
                    'QSO panels use common r=21–22 sources. Grey shows fewer than five objects.')
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('science',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    map_figure(json.loads(args.science.read_text()),args.output)


if __name__=='__main__':
    main()
