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
    p.add_argument('--error-trial',type=Path,help='Second ordinary-field error summary')
    a=p.parse_args()
    if a.error_trial:
        plot_error_trials([read_summary(a.summary),read_summary(a.error_trial)],a.output)
    else:
        plot(read_summary(a.summary),a.output)


def plot_error_trials(data,output):
    plt.rcParams.update({'font.family':'serif','font.size':10,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    colors={'W2_r':'#c37b26','W0_multi':'#2275b5','W2_multi':'#873da8'}
    labels={'W2_r':'Trial errors, r','W0_multi':'W0 errors, multiband','W2_multi':'Trial errors, multiband'}
    for ax,s,title in zip(axes[:2],data,['Ordinary south','Ordinary north']):
        counts=[]
        for name in colors:
            values=[]
            for low in (20,21,22):
                sample=s['qso'][f'r{low}_{low+1}']
                counts.append(sample['n'])
                methods=sample.get('methods',{})
                values.append(methods[name]['radial_abs68']/methods['W0_r']['radial_abs68'] if name in methods else np.nan)
            ax.plot(range(3),values,'o-',color=colors[name],label=labels[name])
        ax.axhline(1,color='k',lw=.8,ls=':')
        ax.set(title=title,ylabel='QSO P68 / W0 r-only P68',xlabel='Reference r magnitude')
        ax.set_xticks(range(3),[f'{low}–{low+1}\nN={s["qso"][f"r{low}_{low+1}"]["n"]}' for low in (20,21,22)])
        ax.legend(fontsize=8,frameon=False)
    held=data[1]['fixed_source_held_year']['by_r']
    for j,name in enumerate(('W0_r','W2_r','W0_multi','W2_multi')):
        values=[held[f'{low}_to_{low+1}']['methods'][name]['radial_p68_mas'] for low in (20,21,22)]
        axes[2].plot(range(3),values,'o-',label=name,color='#666666' if name=='W0_r' else colors[name])
    axes[2].set_xticks(range(3),[f'{low}–{low+1}\npositions={held[f"{low}_to_{low+1}"]["matched_positions"]}' for low in (20,21,22)])
    axes[2].set(title='North: identical 2017 held r positions',ylabel='Held-position radial P68 (mas)',xlabel='NSC r magnitude')
    axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:
        ax.tick_params(direction='in',top=True,right=True)
    fig.suptitle('Ordinary-field error and multiband trials — no new model adopted',y=1.01)
    fig.tight_layout()
    fig.savefig(output,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
