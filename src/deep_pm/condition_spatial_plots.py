"""Plot measured condition-dependent zero and uncertainty diagnostics."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .io import read_summary


def render(data,output):
    plt.rcParams.update({'font.family':'serif','font.size':10,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    names=('ordinary','transition','repeat')
    colors=('#24689b','#ce843d')
    for j,low in enumerate((21,22)):
        ax=axes[j]
        labels=[]
        for i,name in enumerate(names):
            result=data['cohorts'][f'{name}__r{low}_{low+1}']['spatial_zero']
            labels.append(f'{name}\nN={result["n"]}')
            mean=np.array(result['weighted_mean_masyr'])
            bounds=np.array(result['model_mean95_masyr'])
            for axis in (0,1):
                x=i+(axis-.5)*.17
                ax.vlines(x,bounds[axis,0],bounds[axis,1],color=colors[axis],lw=2)
                ax.scatter(x,mean[axis],color=colors[axis],marker='o' if axis==0 else 's')
        ax.axhspan(-.5,.5,color='.9',zorder=0)
        ax.axhline(0,color='.3',lw=.7)
        ax.set(xticks=range(3),xticklabels=labels,title=f'r = {low}–{low+1}: C0 spectral QSO',ylabel='Weighted ICRS mean [mas/yr]')
        if j==0:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([],[],color=colors[k],marker=('o','s')[k],ls='',label=('East','North')[k]) for k in (0,1)],frameon=False)
    for low,style in ((21,'-'),(22,'--')):
        for axis in (0,1):
            scale=[data['cohorts'][f'{name}__r{low}_{low+1}']['source_error_calibration'].get('scale',[np.nan]*2)[axis] for name in names]
            axes[2].plot(range(3),scale,style,marker=('o','s')[axis],color=colors[axis],label=f'r {low}–{low+1}, '+('E','N')[axis])
    axes[2].axhline(1,c='.3',lw=.8)
    axes[2].set(xticks=range(3),xticklabels=names,title='Half-QSO uncertainty diagnostic',ylabel='Training normalized-residual NMAD')
    axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.tick_params(direction='in',top=True,right=True)
    fig.suptitle('99.873 deg² C area: synchronous C0 reference errors retained; model 95% intervals do not certify spatial systematics',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();render(read_summary(a.summary),a.output)


if __name__=='__main__':main()
