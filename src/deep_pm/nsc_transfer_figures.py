"""Figures for the v1.6 fixed-member and independent-region trials."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ('#3674a6','#c15c3e')
ARMS = ('A_r','S_new_seasons','B_new_nights','C_all')


def read(directory, stem):
    return json.loads((directory/f'v16_{stem}_summary.json').read_text())


def intervals(ax, j, point, bounds, color, label=None):
    ax.vlines(j,*bounds,color=color,lw=1.6)
    ax.plot(j,point,'o',color=color,label=label)


def season_figure(season, output):
    fig, axes = plt.subplots(2,2,figsize=(11,7.2),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,(('repeat_sparse','C ordinary control'),('repeat_rich','C repeated control'),
                                        ('b_16_00','B control'),('north_one_00_00','North 1 control'))):
        record = season['summary']['qso/'+key]
        for low,color in zip((21,22),COLORS):
            sample = record['cohorts'][f'r{low}__all']['fixed_qso']
            ax.plot(range(4),[sample['methods'][model]['radial_abs68'] for model in ARMS],
                    'o-',color=color,label=f'{low}–{low+1}, n={sample["n"]}')
        ax.set(title=title,ylabel='QSO 2D PM P68 [mas/yr]',xticks=range(4),
               xticklabels=('r','+ new seasons','+ other new nights','+ old-night bands'))
        ax.grid(axis='y',alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed QSO and r members. Observing seasons use a common 90-day gap rule.')
    fig.savefig(output/'v16_season_summary.png',dpi=165)
    plt.close(fig)


def zero_figure(zero, output):
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True,sharey=True)
    for ax,low in zip(axes,(21,22)):
        row=zero['bins'][f'r{low}']
        for j,key in enumerate(('r_existing','extra_r_anchor')):
            stat=row[key]['fit_metrics']
            s=stat['methods']['all']
            intervals(ax,j,s['median'][1],s['median_bias95'][1],COLORS[j],f'n={stat["n"]}')
        for j,name in enumerate(('matched_color_unrestricted','matched_with_color'),1):
            stat=row[name]['comparison']
            for offset,(key,color) in enumerate(zip(('existing','extra'),COLORS)):
                if stat['n']<5:
                    continue
                s=stat['methods'][key]
                intervals(ax,1.7+j+.22*offset,s['median'][1],s['median_bias95'][1],color)
            ax.text(1.8+j,.96,f'{stat["n"]} pairs',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
        ax.axhline(0,color='.25',lw=.8)
        ax.grid(axis='y',alpha=.25)
        ax.set(title=f'{low} ≤ r < {low+1}',ylabel='North PM median [mas/yr]',xticks=(0,1,2.8,3.8),
               xticklabels=('r existing','extra r','matched','matched + color'))
        ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('C blind-catalogue QSO. Same-core matching excludes proper motion from selection.\nIntervals are conditional object bootstrap 95%.')
    fig.savefig(output/'v16_source_zero_summary.png',dpi=165)
    plt.close(fig)


def transfer_figure(data, output):
    fig,axes=plt.subplots(2,2,figsize=(10.7,7.5),constrained_layout=True)
    models=('NSC','Xiao','r_physical_quality','all_nsc')
    for i,low in enumerate((21,22)):
        for j,(tag,label) in enumerate((('one_sdss_run','One SDSS run'),('multiple_sdss_runs','Multiple SDSS runs'))):
            ax=axes[i,j]
            sample=data['known_stationary'][f'r{low}__{tag}']['strict_published_common']
            ax.set(title=f'{label}. {low}–{low+1}, n={sample["n"]}',ylabel='QSO 2D PM P68 [mas/yr]',
                   xticks=range(4),xticklabels=('NSC DR2','Xiao DESI','C0 r','C0 all NSC'))
            if sample['n']<5:
                ax.text(.5,.5,'Insufficient common QSO',ha='center',transform=ax.transAxes)
                continue
            points=[sample['methods'][m]['radial_abs68'] for m in models]
            ax.bar(range(4),points,color=['.6','#bf8c34',*COLORS])
            for k,y in enumerate(points):
                ax.text(k,y,f'{y:.2f}',ha='center',va='bottom',fontsize=8)
            ax.set_ylim(0,1.17*max(points))
            ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'North 2 transfer trial. {data["completed_nominal_core_area_sqdeg"]:.2f} square degrees\nStrictly common spectral objects on a shared physical C0 graph.')
    fig.savefig(output/'v16_transfer_precision_summary.png',dpi=165)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('figures',type=Path)
    p.add_argument('figure',choices=('season','zero','transfer'))
    a=p.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True)
    stem,func={'season':('season',season_figure),'zero':('source_zero',zero_figure),
               'transfer':('transfer_science',transfer_figure)}[a.figure]
    func(read(a.summary,stem),a.figures)


if __name__=='__main__':
    main()
