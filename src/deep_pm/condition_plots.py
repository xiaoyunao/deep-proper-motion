"""Render measured fixed-source ablations without changing their cohorts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .io import read_summary


def plot(summary,output):
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.linewidth':1.,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    colors={'NSC_only':'#777777','one_SDSS':'#e59c21','all_SDSS':'#2070b4','Xiao':'#9c3987','NSC':'#b54836'}
    labels={'NSC_only':'NSC detections only','one_SDSS':'+ earliest SDSS','all_SDSS':'+ all SDSS','Xiao':'Xiao2025 (DESI)','NSC':'NSC DR2 published'}
    x=np.arange(3)
    bins=[summary['cohorts'][f'ordinary__r{low}_{low+1}__common_releases'] for low in (20,21,22)]
    for name in colors:
        values=[b.get('methods',{}).get(name,{}).get('radial_abs68',np.nan) for b in bins]
        axes[0].plot(x,values,'o-',color=colors[name],label=labels[name],ms=4)
    axes[0].set(yscale='log',ylabel='Absolute 2D proper-motion P68 (mas/yr)',title='Ordinary-region common QSO')
    axes[0].legend(fontsize=8,frameon=False)
    for j,name in enumerate(('NSC_only','one_SDSS','all_SDSS')):
        for axis,marker in ((0,'o'),(1,'s')):
            values=[b.get('methods',{}).get(name,{}).get('fraction_within_1sigma',[np.nan,np.nan])[axis] for b in bins]
            axes[1].plot(x,values,marker+('-' if axis==0 else '--'),color=colors[name],label=labels[name]+(' E' if axis==0 else ' N'),ms=4)
    axes[1].axhline(.6827,c='k',lw=.8,ls=':')
    axes[1].set(ylabel='Fraction within formal +/- 1 sigma',ylim=(0,1),title='Same objects; C0 reference errors retained')
    axes[1].legend(fontsize=7,frameon=False,ncol=2)
    for name in ('NSC_only','one_SDSS','all_SDSS'):
        values=[np.mean(b.get('time_information',{}).get(name,{}).get('time_leverage_years2_per_mas2',[[np.nan]*2]*3)[1]) for b in bins]
        axes[2].plot(x,values,'o-',color=colors[name],label=labels[name],ms=4)
    axes[2].set(ylabel='Median time leverage (yr² / mas²)',yscale='log',title='Weighted temporal information')
    axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:
        ax.set_xticks(x,[f'{low}–{low+1}\nN={b["n"]}' for low,b in zip((20,21,22),bins)])
        ax.set_xlabel('Spectroscopic reference r magnitude')
        ax.tick_params(direction='in',top=True,right=True)
    fig.suptitle(f'Frozen C0/W0 detection ablation — {summary["completed_cores"]}/{summary["expected_cores"]} cores',y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args();plot(read_summary(a.summary),a.output)


if __name__=='__main__':
    main()
