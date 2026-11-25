"""Plot completed production counts and the fixed-position QSO comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(data, output):
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
    fig, axes = plt.subplots(2,2,figsize=(10,7.6),layout='constrained')
    scopes = [('New area',data['new_spectral']),('Cumulative area',data['known_aggregate'])]
    methods = [('NSC','NSC DR2','#999999'),('Xiao','Xiao2025','#333333'),
               ('r_physical_quality','Physical r','#4682a9'),('all_nsc','All NSC bands','#b05d43')]
    for axis,(name,group) in zip(axes[0],scopes):
        ticks = []
        for low in (21,22):
            view=group['known_stationary'][f'r{low}__all']['strict_published_common']
            ticks.append(f'{low}–{low+1}\nN = {view["n"]:,}')
        for i,(key,label,color) in enumerate(methods):
            values=[group['known_stationary'][f'r{low}__all']['strict_published_common']['methods'][key]['radial_abs68'] for low in (21,22)]
            axis.bar(np.arange(2)+(i-1.5)*.19,values,width=.18,color=color,label=label)
        axis.set(xticks=[0,1],xticklabels=ticks,ylabel=r'$P_{68}(|\boldsymbol{\mu}|)$ (mas yr$^{-1}$)',title=name+' · common QSO')
        axis.set_ylim(0,max(patch.get_height() for patch in axis.patches)*1.25)
        axis.legend(frameon=False,ncol=2,fontsize=8,loc='upper left')
    gain, zero = axes[1]
    rows = []
    group=data['new_spectral']['known_stationary']
    for low in (21,22):
        for tag,label in [('all','All runs'),('one_sdss_run','Single run'),('multiple_sdss_runs','Multiple runs')]:
            view=group[f'r{low}__{tag}']['strict_published_common']
            pair=view['paired']['all_nsc_vs_Xiao']
            value=100*pair['fraction_improved'][2]
            lower,upper=np.asarray(pair['fraction95'][2])*100
            rows.append((f'{low}–{low+1} · {label}',value,lower,upper,view['n']))
    for i,(label,value,lower,upper,n) in enumerate(rows):
        gain.barh(i,value,color='#b05d43' if value>=0 else '#4682a9',height=.58)
        gain.errorbar(value,i,xerr=[[max(0,value-lower)],[max(0,upper-value)]],fmt='none',ecolor='black',capsize=2,lw=.9)
    gain.axvline(0,color='.5',lw=.8)
    gain.set(yticks=np.arange(len(rows)),yticklabels=[r[0] for r in rows],
             xlabel='P68 reduction relative to Xiao2025 (%)',title='New area · conditional object intervals')
    gain.invert_yaxis()
    for axis_index,(label,color) in enumerate([('East','#4682a9'),('North','#b05d43')]):
        for scope_index,(_,group) in enumerate(scopes):
            for j,low in enumerate((21,22)):
                row=group['known_stationary'][f'r{low}__all']['strict_published_common']['methods']['all_nsc']
                value=row['median'][axis_index]
                lower,upper=row['median_bias95'][axis_index]
                index=scope_index*2+j
                zero.errorbar(value,index+(axis_index-.5)*.15,xerr=[[value-lower],[upper-value]],
                              fmt='o',color=color,ms=4,capsize=2,label=label if index==0 else None)
    zero.axvline(0,color='.5',lw=.8)
    zero.set(yticks=np.arange(4),yticklabels=['New 21–22','New 22–23','All 21–22','All 22–23'],
             xlabel=r'Median QSO proper motion (mas yr$^{-1}$)',title='All bands · absolute QSO zero points')
    zero.invert_yaxis()
    zero.legend(frameon=False,fontsize=9)
    new,all_area=data['views']['new'],data['views']['cumulative']
    fig.suptitle(f'C0/W0 known-position controls\nNew {new["nominal_area_sqdeg"]:.3f} deg² · {new["research_fits"]:,} solutions     Cumulative {all_area["nominal_area_sqdeg"]:.3f} deg² · {all_area["research_fits"]:,} solutions',fontsize=12)
    fig.savefig(output,dpi=220)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    render(json.loads(args.summary.read_text()),args.output)
