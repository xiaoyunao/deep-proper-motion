"""Draw only measured aggregate final-year and independent-reference diagnostics."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .io import read_summary


def years(data,output):
    regions=data['regions'];names=list(regions)
    aliases=dict(a_09_00='A',b_16_00='B',north_one_00_00='N1',north_two_04_00='N2 test')
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for column,tag in enumerate(('20_to_21','21_to_22','22_to_23')):
        counts=[]
        for i,name in enumerate(names):
            region=regions[name];stats=region.get('by_r',{}).get(tag,{})
            label=aliases.get(name,name)
            counts.append(f"{label}\n{stats.get('matched_positions',0)} positions\n{stats.get('train_sources',0)} train" if stats else f'{label}\nnot testable')
            if 'radial_p68_mas' not in stats:continue
            axes[0,column].scatter(i,stats['radial_p68_mas'],color='#24689b')
            for axis,color,delta in ((0,'#24689b',-.07),(1,'#ce843d',.07)):
                axes[1,column].scatter(i+delta,stats['within_1sigma_fraction'][axis],color=color,
                    label=('Tangent x' if axis==0 else 'Tangent y') if i==0 else None)
        for row in range(2):
            axes[row,column].set(xticks=range(len(names)),xticklabels=counts)
            axes[row,column].tick_params(axis='x',labelsize=7)
            axes[row,column].grid(axis='y',alpha=.2)
        axes[0,column].set(title=tag.replace('_to_','–')+' NSC r',ylabel='Regional tangent radial P68 [mas]',ylim=(0,None))
        axes[1,column].axhline(.6827,color='.5',ls=':',label='Gaussian 1σ')
        axes[1,column].set(ylabel='Conditional fraction within 1σ',ylim=(0,1.03))
    axes[1,0].legend(handles=[Line2D([],[],marker='o',ls='',color=color,label=label)
        for label,color in [('Tangent x','#24689b'),('Tangent y','#ce843d')]]+
        [Line2D([],[],color='.5',ls=':',label='Gaussian 1σ')],fontsize=8)
    fig.suptitle('Final C0 graphs: training-only whole-year prediction; 1 arcsec gate; no absolute stellar PM truth',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)


def overlap(data,output):
    segments=data['segments'];x=np.arange(len(segments))
    labels=[f"{s['comparison_geometry']['dec_bounds_deg'][0]:g}–{s['comparison_geometry']['dec_bounds_deg'][1]:g}°" for s in segments]
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for axis in range(2):
        top,bottom=axes[:,axis]
        for label,key,color,shift in [('Same detections','identical_detection_subset_motion_difference_masyr','#24689b',-.07),
                                      ('Changed detections','changed_detection_subset_motion_difference_masyr','#ce843d',.07)]:
            for i,segment in enumerate(segments):
                stats=segment['stars'][key]
                if 'median' not in stats:continue
                top.errorbar(i+shift,stats['median'][axis],yerr=stats['nmad'][axis],fmt='o',color=color,
                    label=label if i==0 else None,capsize=2)
        for i,segment in enumerate(segments):
            stats=segment['held_galaxies']['held_position_difference_mas']
            if 'median' in stats:bottom.errorbar(i,stats['median'][axis],yerr=stats['nmad'][axis],fmt='o',color='#24689b',capsize=2)
        top.axhspan(-.5,.5,color='.8',alpha=.25,label='±0.5 target, no certification')
        for plot in (top,bottom):
            plot.axhline(0,color='.5',lw=1);plot.set(xticks=x,xticklabels=labels,xlabel='Registered declination strip')
            plot.grid(axis='y',alpha=.2)
        direction='East' if axis==0 else 'North'
        top.set(title=direction,ylabel='Star PM difference [mas/yr]')
        bottom.set(ylabel='Held-galaxy position difference [mas]')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Independent N1/N3 reference graphs: right − left; medians and NMAD (not confidence intervals)',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('years','overlap'));p.add_argument('summary',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();globals()[a.kind](read_summary(a.summary),a.output)


if __name__=='__main__':main()
