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


def panel(ax,values,nside,label,norm,cmap=COLD,coverage=None):
    artist=sky_elements(ax,nside,values,cmap,norm,coverage=coverage)
    for_sky(ax);ax.set_title(label,fontsize=11)
    return artist


def save(fig,folder,name):
    folder.mkdir(exist_ok=True,parents=True);fig.savefig(folder/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def pixel_figure(repo):
    p=read(repo/'results/summary/v22_ngc_masked_pixel_opportunity_summary.json')
    names=['no_cached_SDSS','SDSS_quality_distance','positive_early_control'];labels=['No cached SDSS\nN=20','Quality / distance\nexit; N=10','Early-position\ncontrol; N=10']
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for offset,key,color,label in [(-.12,'per_exposure_SNR_proxy_p16_p50_p84','#b94949','Single exposure'),(.12,'one_per_run_joint_SNR_proxy_p16_p50_p84','#21557f','One field per independent run, combined')]:
        values=np.array([p['categories'][n][key] for n in names]);x=np.arange(3)+offset
        axs[0].errorbar(x,values[:,1],yerr=[values[:,1]-values[:,0],values[:,2]-values[:,1]],fmt='o',capsize=4,label=label,color=color)
    axs[0].set_xticks(range(3),labels);axs[0].set_ylabel('Aperture SNR proxy; median and 16–84% range',fontsize=10);axs[0].legend(fontsize=8);axs[0].axhline(5,color='0.6',ls=':',lw=1)
    x=np.arange(3);axs[1].bar(x-.18,[p['categories'][n]['targets'] for n in names],.36,label='Frozen targets',color='#8cbed9');axs[1].bar(x+.18,[p['categories'][n]['multiple_conditionally_usable_runs'] for n in names],.36,label='≥2 conditionally usable runs',color='#173e66');axs[1].set_xticks(range(3),labels,fontsize=10);axs[1].set_ylabel('Target count');axs[1].legend(fontsize=8)
    fig.suptitle('Frozen forty: real SDSS masks and early information, not certified centroids',fontsize=12)
    save(fig,repo/'figures','v22_NGC_early_pixel_information')


def angle_figures(repo,quality):
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for row,low in enumerate((21,22)):
        bins=quality['fixed_old_QSO'][str(low)]['by_chart_angle']
        for axis in range(2):
            ax=axs[row,axis]
            for offset,key,color,label in [(-.5,'B1_over_old','#295b92','Covariance only / old'),(.5,'new_over_old','#d18c2e','Full refit / old')]:
                x=[];y=[];errors=[]
                for lower,entry in bins.items():
                    if 'error_ratio' not in entry:continue
                    r=entry['error_ratio'][key];x.append(float(lower)+5+offset);y.append(r['median'][axis]);errors.append(r['median_source_sampling95'][axis])
                    if key=='new_over_old':ax.annotate(f"{entry['computed']}/{entry['frozen_denominator']}",(x[-1],y[-1]),xytext=(0,7),textcoords='offset points',fontsize=6,ha='center')
                if x:
                    y=np.array(y);bound=np.array(errors);ax.errorbar(x,y,yerr=[np.maximum(y-bound[:,0],0),np.maximum(bound[:,1]-y,0)],fmt='o-',capsize=3,color=color,label=label)
            ax.axhline(1,color='.5',ls=':',lw=1);ax.set(xlim=(0,70),xlabel='Actual reference-chart angle (deg)',ylabel='Median reported PM error ratio',title=f'{low} ≤ r < {low+1}; '+['East','North'][axis]);ax.grid(alpha=.2)
            if row==0 and axis==0:ax.legend(fontsize=8)
    fig.suptitle('Fixed old QSO identities; 95% conditional source-sampling intervals; N computed/original',fontsize=11)
    save(fig,repo/'figures','v22_fixed_QSO_error_vs_chart_angle')
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,low in zip(axs,(21,22)):
        x=[];y=[];bounds=[]
        for lower,entry in quality['fixed_old_QSO'][str(low)]['by_chart_angle'].items():
            m=entry.get('same_identity_quality',{});p=m.get('paired',{}).get('new_vs_old')
            if p is None:continue
            x.append(float(lower)+5);y.append(p['fraction_improved'][2]);bounds.append(p['fraction95'][2])
            ax.annotate(str(entry['computed']),(x[-1],y[-1]),xytext=(0,7),textcoords='offset points',fontsize=7,ha='center')
        if x:
            y=np.array(y);b=np.array(bounds);ax.errorbar(x,y,yerr=[np.maximum(y-b[:,0],0),np.maximum(b[:,1]-y,0)],fmt='o-',capsize=3,color='#173e66')
        ax.axhline(0,color='.5',ls=':',lw=1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(xlim=(0,70),xlabel='Actual reference-chart angle (deg)',ylabel='1 − P68(new)/P68(old)',title=f'{low} ≤ r < {low+1}');ax.grid(alpha=.2)
    fig.suptitle('Point-estimate change on the same computable identities; new quality failures remain',fontsize=12)
    save(fig,repo/'figures','v22_fixed_QSO_point_gain_vs_chart_angle')


def run(repo):
    folder=repo/'results/summary';quality=read(folder/'v22_global_QSO_quality_summary.json');spatial=quality['spatial_nside64'];output=repo/'figures'
    counts_path=folder/'v22_catalogue_spatial_support_summary.json';source_map=read(counts_path) if counts_path.exists() else {}
    covered={r['pixel64']:1 for r in source_map.get('plot_spatial_nside64',[])}
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
            norm=TwoSlopeNorm(vmin=bounds[0],vcenter=0,vmax=bounds[1]);artist=panel(axs[row,col],values,64,f'{low} ≤ r < {low+1}; vs {reference}; N plotted={count:,}',norm,coverage=covered)
            bar=fig.colorbar(artist,ax=axs[row,col],orientation='horizontal',fraction=.055,pad=.08,extend='both');bar.ax.xaxis.set_major_formatter(PercentFormatter(1));bar.set_label('Improvement = 1 − P68(new)/P68(reference)',fontsize=9)
    fig.suptitle('Final blind QSO; NSIDE=64; gray: occupied sources but <5 strict QSO',fontsize=14)
    save(fig,output,'v22_QSO_improvement_aitoff')
    angle_figures(repo,quality)
    counts_path=folder/'v22_catalogue_spatial_support_summary.json'
    if counts_path.exists():
        support=source_map['plot_spatial_nside64'];fig,axs=plt.subplots(2,2,figsize=(13,8),subplot_kw=dict(projection='aitoff'),constrained_layout=True)
        specs=[('source_count','Retained source count / occupied pixel',LogNorm(10,100000),'magma'),('median_nsc_nights','Median actual NSC observing nights',plt.Normalize(0,40),'viridis'),('high_leverage_fraction','High physical-group leverage fraction',plt.Normalize(0,1),'inferno'),('median_SE_n','Median conditional North PM error',plt.Normalize(0,10),'cividis')]
        for ax,(key,label,norm,cmap) in zip(axs.flat,specs):
            a=panel(ax,{r['pixel64']:r[key] for r in support},64,label,norm,plt.get_cmap(cmap));fig.colorbar(a,ax=ax,orientation='horizontal',fraction=.055,pad=.08)
        fig.suptitle('HEALPix NSIDE=64: source support, not certified sky area',fontsize=14)
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repo',type=Path);p.add_argument('--pixels-only',action='store_true');a=p.parse_args();pixel_figure(a.repo) if a.pixels_only else run(a.repo)
