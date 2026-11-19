"""Plot final continuous research results using deidentified aggregates only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

METHODS=('NSC','Xiao','r_physical_quality','all_nsc')
LABELS=('NSC DR2','Xiao2025 DESI','C0 r physical','C0 all NSC')
COLORS=('#999999','#bd8e35','#3674a6','#cf6545')
KINDS=('r_existing','extra_r_anchor','non_r_anchor')


def run(summary,output):
    data=json.loads(summary.read_text());output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.5),constrained_layout=True)
    for row,low in enumerate((21,22)):
        for col,(tag,label) in enumerate((('all','All scans'),('one_sdss_run','One SDSS run'),('multiple_sdss_runs','Multiple SDSS runs'))):
            part=data['known_stationary'][f'r{low}__{tag}']['strict_published_common'];ax=axes[row,col]
            for j,(method,tint) in enumerate(zip(METHODS,COLORS)):
                if method in part.get('methods',{}):
                    value=part['methods'][method]['radial_abs68'];ax.bar(j,value,color=tint);ax.text(j,value,f'{value:.2f}',ha='center',va='bottom',fontsize=8)
            ax.set(title=f'{label}; {low} <= r < {low+1}, n={part["n"]}',xticks=range(4),xticklabels=['NSC','Xiao','r','all'])
            ax.set_ylim(0,ax.get_ylim()[1]*1.17);ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylabel('QSO 2D absolute PM P68 [mas/yr]')
    fig.legend([plt.Rectangle((0,0),1,1,color=c) for c in COLORS],LABELS,loc='lower center',bbox_to_anchor=(.5,-.045),ncol=4,frameon=False)
    fig.suptitle('Continuous C: strict common spectral QSO; frozen C0/W0, J2015.5\nKnown-position members; precision comparison does not certify blind identities',fontsize=12)
    fig.savefig(output/'v15_continuous_precision_summary.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12.5,8.3),constrained_layout=True)
    for k,(kind,label,tint) in enumerate(zip(KINDS,('r-existing','extra r-anchor','non-r anchor'),('#3674a6','#cf6545','#79a58b'))):
        counts=[data['by_class'].get(f'{kind}__r{low}',{}).get('research_fits',0) for low in (21,22)]
        axes[0,0].bar(np.arange(2)+(k-1)*.25,counts,.25,label=label,color=tint)
        for x,n in zip(np.arange(2)+(k-1)*.25,counts):axes[0,0].text(x,n,f'{n:,}',ha='center',va='bottom',fontsize=8)
    axes[0,0].set(xticks=[0,1],xticklabels=['21–22','22–23'],ylabel='Full-period research fits',title='Full-data classes, not held-year training classes');axes[0,0].legend(fontsize=8);axes[0,0].margins(y=.2)
    for j,low in enumerate((21,22)):
        part=data['known_stationary'][f'r{low}__all']['strict_published_common'];ax=axes[0,1]
        for k,method in enumerate(('r_physical_quality','all_nsc')):
            values=part['methods'][method]['fraction_within_1sigma']
            ax.plot(np.arange(2)+j*3,np.array(values),'o-' if k else 's--',color=COLORS[3 if k else 2],label=('all NSC' if k else 'r physical') if j==0 else None)
        for k,kind in enumerate(KINDS):
            cohort=data['blind_spectral'][f'{kind}__r{low}']['all_spectral'];m=cohort.get('methods',{}).get('blind_all');ax=axes[1,j]
            if m is None:
                ax.text(k,.18,f'n={cohort["n"]}\nnot scored',transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='.4');continue
            median=m['median'];bounds=m['median_bias95']
            for axis,color in enumerate(('#3674a6','#cf6545')):
                x=k+(-.12 if axis==0 else .12);ax.vlines(x,*bounds[axis],color=color);ax.plot(x,median[axis],'o',color=color,label=('east','north')[axis] if k==0 else None)
            ax.text(k,max(b[1] for b in bounds)+.15,f'n={cohort["n"]}',ha='center',fontsize=8)
        ax=axes[1,j];ax.axhline(0,color='.3',lw=.8);ax.axhspan(-.5,.5,color='.9',zorder=-2);ax.set(xticks=range(3),xticklabels=['r-existing','extra r','non-r'],ylabel='QSO median PM [mas/yr]',title=f'Blind catalogue spectral matches, {low}–{low+1}');ax.legend(fontsize=8);ax.margins(y=.25);ax.set_xlim(-.4,2.4)
    axes[0,1].axhline(.6827,color='.4',lw=.8,label='Gaussian 68.3%');axes[0,1].set(xticks=[0,1,3,4],xticklabels=['21 east','21 north','22 east','22 north'],ylabel='Fraction within formal 1 sigma',ylim=(0,1.02),title='Same common QSO; W0 + reference bootstrap');axes[0,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Continuous C: source yield and remaining uncertainty\nZero intervals: object bootstrap 95%; grey +/-0.5 is a goal, not certification',fontsize=12)
    fig.savefig(output/'v15_continuous_yield_zero_summary.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8.4),constrained_layout=True)
    entries=list(data['spatial'].values());maps=[]
    maps.append(('All / legacy-r research fits (owned)',lambda r:r['owned_all_nsc']/r['owned_r_control'] if r['owned_r_control'] else None,'viridis',None))
    maps.append(('Extra-r + non-r fits, r=21–23 / deg2',lambda r:sum(r['by_class'].get(f'{k}__r{low}',{}).get('research_fits',0) for k in KINDS[1:] for low in (21,22))/box_area(r['core_bounds_deg']),'viridis',None))
    maps.append(('r-existing fits: median NSC nights, r=21–22',lambda r:r['by_class']['r_existing__r21']['information'].get('nsc_nights_p16_p50_p84',[None,None,None])[1],'viridis',None))
    for axis in (0,1):
        maps.append((f'Common QSO r=21–22 {("east","north")[axis]} median [mas/yr]',lambda r,a=axis:qso_value(r,'median',a),'coolwarm',(-5,5)))
    maps.append(('Common QSO r=21–22 all vs r P68 gain [%]',lambda r:qso_gain(r),'coolwarm',(-30,30)))
    for ax,(title,value,cmap,limits) in zip(axes.flat,maps):
        finite=[value(r) for r in entries];finite=[v for v in finite if v is not None and np.isfinite(v)]
        if not finite:ax.set_title(title+'\nNo supported cells');continue
        vmin,vmax=limits or (min(finite),max(finite));vmax=max(vmax,vmin+1e-8)
        norm=TwoSlopeNorm(0,vmin,vmax) if limits else plt.Normalize(vmin,vmax)
        cm=plt.get_cmap(cmap)
        for r in entries:
            box=r['core_bounds_deg'];v=value(r);valid=v is not None and np.isfinite(v)
            ax.add_patch(Rectangle((box['ra_bounds_deg'][0],box['dec_bounds_deg'][0]),np.ptp(box['ra_bounds_deg']),np.ptp(box['dec_bounds_deg']),facecolor=cm(norm(v)) if valid else '#eeeeee',edgecolor='white',lw=.35))
        ax.set(xlim=(345,355),ylim=(-5,5),xlabel='RA [deg]',ylabel='Dec [deg]',title=title);ax.set_aspect('equal');fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),ax=ax,shrink=.8,extend='both' if limits else 'neither')
    fig.suptitle('Continuous C: actual core distributions under one frozen physical reference graph\nQSO maps require n>=5 per cell; grey = insufficient data, not zero or certified sky',fontsize=12)
    fig.savefig(output/'v15_continuous_spatial_summary.png',dpi=180);plt.close(fig)
    extra=summary.parent/'v15_C_information_summary.json'
    if extra.exists():c_ablation_figure(json.loads(extra.read_text()),output)


def c_ablation_figure(data,output):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    tints=('#3674a6','#cf6545');names=('A_r','B1_one_per_new_night','B_new_nights','C_all')
    for col,(core,label) in enumerate((('repeat_sparse','Ordinary C control'),('repeat_rich','Repeated C control'))):
        trial=data['trials'][core]
        for low,tint in zip((21,22),tints):
            part=trial['time']['cohorts'][f'r{low}__all']['fixed_qso']
            if part['n']>=5:axes[0,col].plot(range(4),[part['methods'][name]['radial_abs68'] for name in names],'o-',color=tint,label=f'r={low}–{low+1}, n={part["n"]}')
        axes[0,col].set(title=label,ylabel='QSO 2D absolute PM P68 [mas/yr]',xticks=range(4),xticklabels=['r only','+ one/new night','+ all/new night','+ all bands']);axes[0,col].tick_params(axis='x',labelsize=8);axes[0,col].legend(fontsize=8)
        bands=sorted({key[5:] for low in (21,22) for key in trial['bands']['cohorts']['r'+str(low)] if key.startswith('band_')})
        for low,tint,shift in zip((21,22),tints,(-.10,.10)):
            for j,band in enumerate(bands):
                sample=trial['bands']['cohorts']['r'+str(low)].get('band_'+band,{}).get('same_band_covered_objects',{})
                if sample.get('n',0)<5:continue
                result=sample['paired']['all_nsc_sdss_all_vs_leave_'+band];point=100*result['fraction_improved'][2];interval=100*np.array(result['fraction95'][2]);x=j+shift
                axes[1,col].vlines(x,*interval,color=tint);axes[1,col].plot(x,point,'o',color=tint);axes[1,col].annotate(f'n={sample["n"]}',(x,point),xytext=(3,5 if low==21 else -12),textcoords='offset points',fontsize=7,color=tint)
        axes[1,col].axhline(0,color='.4',lw=.8);axes[1,col].set(title='Add one band to all other eligible bands',ylabel='Paired P68 gain vs band deletion [%]',xticks=range(len(bands)),xticklabels=bands);axes[1,col].margins(x=.2,y=.2)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed C control objects: new-night and individual-band information\nEach band uses its same covered subset; object 95% intervals; n<5 not scored. Explanatory, not new blind validation.',fontsize=11)
    fig.savefig(output/'v15_C_time_band_summary.png',dpi=180);plt.close(fig)


def box_area(box):
    return float(np.ptp(box['ra_bounds_deg'])*np.degrees(np.ptp(np.sin(np.deg2rad(box['dec_bounds_deg'])))))


def qso_value(row,key,axis):
    sample=row['known_stationary']['r21__all']['strict_published_common']
    return sample['methods']['all_nsc'][key][axis] if sample['n']>=5 else None


def qso_gain(row):
    sample=row['known_stationary']['r21__all']['strict_published_common']
    if sample['n']<5:return None
    before=sample['methods']['r_physical_quality']['radial_abs68']
    return 100*(1-sample['methods']['all_nsc']['radial_abs68']/before) if before else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('output_directory',type=Path);a=p.parse_args();run(a.summary,a.output_directory)


if __name__=='__main__':main()
