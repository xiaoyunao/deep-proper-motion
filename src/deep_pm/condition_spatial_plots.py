"""Plot measured condition-dependent zero and uncertainty diagnostics."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .io import read_summary


def render_map(data,output):
    import healpy as hp
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize,LogNorm
    from matplotlib.patches import Polygon,Patch
    plt.rcParams.update({'font.family':'serif','font.size':10,'savefig.dpi':180})
    fig,axes=plt.subplots(2,3,figsize=(13.5,8.5),layout='constrained')
    pixels=data['spatial_nside16'];minimum=data['spatial_zero_minimum_qso_count']
    counts=[cell['by_r'][f'r{low}_{low+1}']['spatial_zero']['n'] for cell in pixels.values() for low in (21,22)]
    zero=Normalize(-3,3);support=LogNorm(1,max(max(counts),2))
    for row,low in enumerate((21,22)):
        for pixel,cell in pixels.items():
            lon,lat=hp.vec2ang(hp.boundaries(16,int(pixel),step=20).T,lonlat=True)
            lon=(lon-350+180)%360-180+350
            polygon=np.column_stack([lon,lat])
            result=cell['by_r'][f'r{low}_{low+1}']['spatial_zero'];n=result['n']
            for column in range(3):
                if column==2:
                    face=plt.cm.viridis(support(n)) if n else '.92';hatch=None
                elif n>=minimum:
                    face=plt.cm.RdBu_r(zero(result['weighted_mean_masyr'][column]))
                    bounds=result['model_mean95_masyr'][column]
                    hatch='///' if bounds[0]<=0<=bounds[1] else None
                else:face='.92';hatch=None
                axes[row,column].add_patch(Polygon(polygon,facecolor=face,edgecolor='.5',lw=.4,hatch=hatch))
        for column in range(3):
            ax=axes[row,column]
            for y in (-3,-1,1,3):ax.axhline(y,c='.25',ls=':',lw=.5)
            ax.set(xlim=(345,355),ylim=(-5,5),xticks=[346,350,354],yticks=[-4,0,4],
                   xlabel='ICRS RA [deg]',ylabel='ICRS Dec [deg]',
                   title=f'r = {low}–{low+1}: '+('East mean','North mean','QSO support')[column])
            ax.set_aspect('equal');ax.tick_params(direction='in',top=True,right=True)
    fig.colorbar(ScalarMappable(norm=zero,cmap='RdBu_r'),ax=axes[:,:2].ravel().tolist(),
                 label='Weighted ICRS motion [mas/yr]',extend='both',shrink=.75)
    fig.colorbar(ScalarMappable(norm=support,cmap='viridis'),ax=axes[:,2].tolist(),label='QSO count',shrink=.75)
    axes[0,0].legend(handles=[Patch(facecolor='white',edgecolor='.5',hatch='///',label='Model 95% includes zero'),
                             Patch(facecolor='.92',label=f'N < {minimum}')],loc='lower left',fontsize=7,framealpha=.9)
    fig.suptitle('C0 spatial diagnostics: NSIDE 16 averages; no spatial-zero certification\n'
                 'Dotted lines mark scan-condition strata; coarse pixels can mix conditions. NSIDE 128 organization is unchanged.',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output);plt.close(fig)


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
    p.add_argument('--map',action='store_true')
    a=p.parse_args();(render_map if a.map else render)(read_summary(a.summary),a.output)


if __name__=='__main__':main()
