"""Standalone source-backed sky figures for the complete coordinate update."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,LogNorm
from matplotlib.ticker import PercentFormatter
from .blind_catalogue_quality import read
from .meeting_figures import sky_elements,for_sky

COLD=LinearSegmentedColormap.from_list('gain_hotcold',['#a51827','#e8977b','#faf8f2','#8cbed9','#153961'])


def panel(ax,values,nside,label,norm,cmap=COLD):
    artist=sky_elements(ax,nside,values,cmap,norm)
    for_sky(ax);ax.set_title(label,fontsize=11)
    return artist


def save(fig,folder,name):
    folder.mkdir(exist_ok=True,parents=True);fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def run(repo):
    folder=repo/'results/summary';quality=read(folder/'v22_global_QSO_quality_summary.json');spatial=quality['spatial_nside64'];output=repo/'figures'
    fig,axs=plt.subplots(2,2,figsize=(13,8),subplot_kw=dict(projection='aitoff'),constrained_layout=True)
    for row,low in enumerate((21,22)):
        for col,reference in enumerate(('Xiao','NSC')):
            values={};count=0
            for entry in spatial:
                comparison=entry['comparison']
                if entry['r_bin']!=low or 'methods' not in comparison:continue
                a=comparison['methods']['new_all']['radial_abs68'];b=comparison['methods'][reference]['radial_abs68']
                if b<=0:continue
                values[entry['pixel64']]=1-a/b;count+=comparison['n']
            bounds=(-.4,.6) if reference=='Xiao' else (-.3,.9)
            norm=TwoSlopeNorm(vmin=bounds[0],vcenter=0,vmax=bounds[1]);artist=panel(axs[row,col],values,64,f'{low} ≤ r < {low+1}; vs {reference}; N={count:,}',norm)
            bar=fig.colorbar(artist,ax=axs[row,col],orientation='horizontal',fraction=.055,pad=.08,extend='both');bar.ax.xaxis.set_major_formatter(PercentFormatter(1));bar.set_label('Improvement = 1 − P68(new)/P68(reference)',fontsize=9)
    fig.suptitle('Final blind QSO: strict common objects per HEALPix NSIDE=64',fontsize=14)
    save(fig,output,'v22_QSO_improvement_aitoff')
    counts_path=folder/'v22_catalogue_spatial_support_summary.json'
    if counts_path.exists():
        support=read(counts_path)['spatial'];fig,axs=plt.subplots(2,2,figsize=(13,8),subplot_kw=dict(projection='aitoff'),constrained_layout=True)
        specs=[('source_count','Retained source count / occupied pixel',LogNorm(10,100000),'magma'),('median_nsc_nights','Median actual NSC observing nights',plt.Normalize(0,40),'viridis'),('high_leverage_fraction','High physical-group leverage fraction',plt.Normalize(0,1),'inferno'),('median_SE_n','Median conditional North PM error',plt.Normalize(0,10),'cividis')]
        for ax,(key,label,norm,cmap) in zip(axs.flat,specs):
            a=panel(ax,{r['pixel128']:r[key] for r in support},128,label,norm,plt.get_cmap(cmap));fig.colorbar(a,ax=ax,orientation='horizontal',fraction=.055,pad=.08)
        fig.suptitle('HEALPix NSIDE=128: source support, not certified sky area',fontsize=14)
        save(fig,output,'v22_source_support_aitoff')
    for low in (21,22):
        fig,axs=plt.subplots(2,2,figsize=(13,8),subplot_kw=dict(projection='aitoff'),constrained_layout=True)
        for axis,label in enumerate(('East','North')):
            zero={};coverage={}
            for r in spatial:
                c=r['comparison']
                if r['r_bin']!=low or 'methods' not in c:continue
                s=c['methods']['new_all'];zero[r['pixel64']]=s['median'][axis];coverage[r['pixel64']]=s['fraction_within_1sigma'][axis]
            a=panel(axs[0,axis],zero,64,label+' median QSO zero',TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4),plt.get_cmap('RdBu_r'))
            fig.colorbar(a,ax=axs[0,axis],orientation='horizontal',fraction=.055,pad=.08,extend='both',label='mas / yr')
            a=panel(axs[1,axis],coverage,64,label+' raw 1σ coverage',plt.Normalize(0,1),plt.get_cmap('viridis'))
            bar=fig.colorbar(a,ax=axs[1,axis],orientation='horizontal',fraction=.055,pad=.08);bar.ax.xaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(f'{low} ≤ r < {low+1}; ≥5 strict-common QSO/pixel; conditional quality, not certified area',fontsize=13)
        save(fig,output,f'v22_QSO_zero_coverage_r{low}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repo',type=Path);a=p.parse_args();run(a.repo)
