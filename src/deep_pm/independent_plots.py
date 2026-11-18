"""Plot inspected independent-region QSO comparisons; sealed region absent."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .io import read_summary


def render(data,output):
    cohorts=data['standard_icrs_comparison']
    boxes=[box for box in cohorts if box!='pooled']
    methods=[('NSC','#687581'),('Xiao','#ce843d'),('C0','#24689b')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for column,mag in enumerate(('r21_22','r22_23')):
        upper,lower=axes[:,column]
        counts=[]
        for i,box in enumerate(boxes):
            stats=cohorts[box][mag]['conditions']['all']['common']
            counts.append(stats['n'])
            for j,(method,color) in enumerate(methods):
                if 'methods' in stats:upper.scatter(i+(j-1)*.16,stats['methods'][method]['radial_abs68'],color=color,s=45)
        upper.set(xticks=range(len(boxes)),xticklabels=[f'{b}\n(n={n})' for b,n in zip(boxes,counts)],
                  ylabel='ICRS radial absolute P68 [mas/yr]',title=mag.replace('r','r = ').replace('_','–'),ylim=(0,None))
        labels=['all','one_sdss_run','multiple_sdss_runs','nsc_span_lt_2yr','nsc_span_ge_2yr']
        for i,label in enumerate(labels):
            stats=cohorts['pooled'][mag]['conditions'][label]
            paired=stats['common'].get('paired',{}).get('C0_vs_Xiao',{})
            if paired:
                point=100*paired['fraction_improved'][2]
                lo,hi=100*np.array(paired['fraction95'][2])
                lower.vlines(i,lo,hi,color='#24689b',lw=2)
                lower.scatter(i,point,color='#24689b',s=40)
                physical=stats['physical_group_sensitivity']['paired_radial_gain']['Xiao']['visit_weighted']['sdss_run'].get('relative_gain95')
                if physical is not None:lower.vlines(i+.1,* (100*np.array(physical)),color='#ce843d',lw=2)
        lower.axhline(0,color='.4',lw=1)
        lower.set(xticks=range(len(labels)),xticklabels=['All','1 SDSS run','2+ SDSS runs','NSC <2 yr','NSC ≥2 yr'],
                  ylabel='C0 relative P68 improvement vs Xiao [%]')
        lower.tick_params(axis='x',labelsize=9)
        for axis in (upper,lower):axis.grid(axis='y',alpha=.2)
    axes[0,0].legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=k) for k,c in methods])
    axes[1,0].legend(handles=[Line2D([],[],color=c,lw=2,label=k) for k,c in [('Object bootstrap 95%','#24689b'),('SDSS-run weight sensitivity','#ce843d')]],fontsize=8)
    fig.suptitle('Independent batch 01: common spectroscopic QSO; north_two remains sealed',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();render(read_summary(a.summary),a.output)


if __name__=='__main__':main()
