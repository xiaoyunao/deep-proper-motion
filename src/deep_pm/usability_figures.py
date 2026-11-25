"""Scientific v2 plots from reviewed aggregate files only."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
import numpy as np

from .meeting_figures import sky_elements


def load(path):return json.loads(path.read_text())


def save(fig,dest):
    fig.savefig(dest,dpi=250,facecolor='white',bbox_inches='tight');plt.close(fig)


def blind_figure(quality,replay,output):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    scopes=[('New area',lambda low:replay['samples']['new'][str(low)]['all']['strict_published']),
            ('Cumulative area',lambda low:quality['cohorts'][str(low)]['all']['common_published'])]
    for axis,(name,rows) in zip(axes[0],scopes):
        for j,(key,label,color) in enumerate([('NSC','NSC DR2','#999999'),('Xiao','Xiao2025','#333333'),('blind_all','Blind all-band catalogue','#b05d43')]):
            axis.bar(np.arange(2)+(j-1)*.24,[rows(low)['methods'][key]['radial_abs68'] for low in (21,22)],width=.23,label=label,color=color)
        axis.set(xticks=[0,1],xticklabels=[f'{low}–{low+1}\nN = {rows(low)["n"]:,}' for low in (21,22)],
            ylabel=r'$P_{68}(|\boldsymbol{\mu}|)$ (mas yr$^{-1}$)',title=name+' · strictly common blind QSO')
        axis.set_ylim(0,max(rows(low)['methods']['NSC']['radial_abs68'] for _,rows in scopes for low in (21,22))*1.25)
        axis.legend(frameon=False,fontsize=8)
    axis=axes[1,0];labels=[];gain=[];ns=[]
    for low in (21,22):
        for tag,name in [('single_sdss_run','Single run'),('multiple_sdss_runs','Multiple runs')]:
            row=replay['samples']['new'][str(low)][tag]['strict_published']
            labels.append(f'{low}–{low+1} · {name}');gain.append(row['relative_p68_gain']['Xiao']*100);ns.append(row['n'])
    axis.barh(range(4),gain,color=['#b05d43' if v>=0 else '#4682a9' for v in gain])
    for j,(g,n) in enumerate(zip(gain,ns)):axis.text(33,j,f'{g:+.1f}% · N={n:,}',va='center',ha='right',fontsize=8)
    axis.axvline(0,color='.5',lw=.8);axis.set(yticks=range(4),yticklabels=labels,xlim=(-10,34),xlabel='P68 reduction relative to Xiao2025 (%)',title='New blind catalogue · fixed common objects');axis.invert_yaxis()
    axis=axes[1,1]
    for j,(component,label,color) in enumerate([(0,'East','#4682a9'),(1,'North','#b05d43')]):
        points=[rows(low)['methods']['blind_all']['median'][component] for _,rows in scopes for low in (21,22)]
        axis.plot(points,np.arange(4)+(j-.5)*.13,'o',color=color,label=label)
    axis.axvline(0,color='.5',lw=.8);axis.set(yticks=range(4),yticklabels=['New 21–22','New 22–23','All 21–22','All 22–23'],xlabel='Median QSO proper motion (mas yr⁻¹)',title='Absolute zero point · no fitted offset removed');axis.invert_yaxis();axis.legend(frameon=False)
    fig.suptitle('Frozen C0/W0 · actual blind research catalogue',fontsize=15)
    save(fig,output/'v20_blind_science_summary.png')


def funnel_figure(data,output):
    totals={}
    for low in ('21','22'):
        value=Counter()
        for core in data['cores']:value.update(core['qso'][low]['geometric_opportunity']['primary_exit'])
        totals[low]=value
    reasons=[('no_nsc_detection','No NSC detection'),('no_sdss_detection','No SDSS detection'),('nsc_detection_physical_quality','NSC quality'),
        ('sdss_detection_quality_or_position','SDSS quality / position'),('no_eligible_anchor_or_track','Anchor / association'),
        ('pointlike_morphology','Morphology < 0.8'),('time_baseline','Time support'),('m3_support','Reference support'),
        ('quality','Fit quality'),('final_spectral_identity_or_owner','Identity / owner'),('recovered','Recovered')]
    reasons=[r for r in reasons if any(totals[low][r[0]] for low in totals)]
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    for axis,(low,values) in zip(axes,totals.items()):
        names=[name for _,name in reasons];v=[values[key] for key,_ in reasons]
        axis.barh(np.arange(len(v)),v,color=['#b05d43' if key=='recovered' else '#4682a9' if key=='pointlike_morphology' else '#879ba8' for key,_ in reasons])
        for j,count in enumerate(v):axis.text(count+max(v)*.02,j,f'{count:,} ({count/sum(v):.1%})',va='center',fontsize=9)
        axis.set(yticks=np.arange(len(v)),yticklabels=names,xlim=(0,max(v)*1.42),xlabel='Spectroscopic QSO count',title=f'{low}–{int(low)+1} · geometric opportunities N={sum(v):,}')
        axis.invert_yaxis()
    n=sum(r['morphology']['physical_r']['anchors'] for r in data['cores']);unknown=sum(r['morphology']['physical_r']['legacy_pass_valid_only_fail'] for r in data['cores'])
    fig.suptitle(f'16 metadata-selected cores · mutually exclusive exits\nPhysical r-anchor morphology: {n:,} anchors; missing/invalid bypass impact = {unknown:,}',fontsize=13)
    save(fig,output/'v20_selection_funnel_summary.png')


def spatial_figure(data,output):
    fig,axes=plt.subplots(2,4,figsize=(17,7),subplot_kw=dict(projection='aitoff'),layout='constrained')
    columns=[('median',0,'East zero point','RdBu_r',TwoSlopeNorm(0,vmin=-3,vmax=3)),
        ('median',1,'North zero point','RdBu_r',TwoSlopeNorm(0,vmin=-3,vmax=3)),
        ('raw_1sigma',0,'East 1σ coverage','cividis',Normalize(.30,.85)),
        ('raw_1sigma',1,'North 1σ coverage','cividis',Normalize(.30,.85))]
    for row,low in enumerate((21,22)):
        for col,(key,axis,title,cmap,norm) in enumerate(columns):
            values={str(v['pixel']):v[key][axis] for k,v in data['spatial'].items() if k.startswith(str(low)+'__')}
            artist=sky_elements(axes[row,col],32,values,cmap,norm)
            axes[row,col].set_title(f'{low}–{low+1} · {title}',fontsize=10,pad=12)
            cb=fig.colorbar(artist,ax=axes[row,col],orientation='horizontal',fraction=.055,pad=.12)
            cb.set_label('mas yr⁻¹' if key=='median' else 'Observed probability',fontsize=9)
            cb.ax.tick_params(labelsize=8)
    fig.suptitle('Blind spectroscopic QSO · HEALPix NSIDE32, N≥30 per displayed cell\nRaw medians and reported-error coverage; display cells do not certify their full sky area',fontsize=14)
    save(fig,output/'v20_spatial_quality_summary.png')


def cluster_deep_figure(data,output):
    photo=data['deep_photometry'];reg=photo['selection'];areas=reg['effective_area_sqdeg']
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    for axis,target,bg,title in [(axes[0,0],'train','training_background','Independent optical training'),
        (axes[0,1],'target','evaluation_background','Frozen optical evaluation')]:
        value=np.asarray(photo['optical_histograms'][target])-areas[target]/areas[bg]*np.asarray(photo['optical_histograms'][bg])
        vmax=max(10,np.quantile(abs(value),.98))
        artist=axis.pcolormesh(photo['color_edges'],photo['r0_edges'],value,cmap='RdBu_r',norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax),rasterized=True)
        sequence=[r for r in reg['template'] if r['center_color'] is not None]
        y=np.asarray([(r['lower_r0']+r['upper_r0'])/2 for r in sequence]);x=np.asarray([r['center_color'] for r in sequence])
        axis.plot(x,y,'k.-',lw=1);axis.plot(x-.105,y,'k--',lw=.7);axis.plot(x+.105,y,'k--',lw=.7)
        axis.set(xlabel='Extinction-corrected g − r',ylabel=r'$r_0$ (mag)',title=title,ylim=(23,18))
        fig.colorbar(artist,ax=axis,label='Area-scaled target − background count')
    axis=axes[1,0];xx=np.arange(21,23,.5)
    target=[photo['cohorts']['target'][str(v)]['cmd_candidates'] for v in xx]
    motion=[data['cohorts'][f'deep__target__{v}']['motion']['n'] for v in xx]
    axis.bar(xx+.11,target,width=.2,color='#879ba8',label='Independent CMD candidates')
    axis.bar(xx-.11,motion,width=.2,color='#b05d43',label='Catalogue motion available')
    for x,n,d in zip(xx,motion,target):axis.text(x,max(n,d)+9,f'{n}/{d}',ha='center',fontsize=9)
    axis.set(xticks=xx,xticklabels=['21–21.5','21.5–22','22–22.5','22.5–23'],ylabel='Target-annulus count',title='Optical availability and catalogue selection');axis.legend(frameon=False,fontsize=8)
    axis.set_ylim(0,max(target)*1.30)
    axis=axes[1,1]
    for component,color,label in [(0,'#4682a9','East'),(1,'#b05d43','North')]:
        means=[];errors=[];bins=[]
        for lower in xx:
            r=data['deep_background_mixture'][str(lower)]['background_mixture_motion']
            if 'weighted_mean' not in r:continue
            bins.append(lower+.25);means.append(r['offset_from_bright_reference'][component])
            errors.append(np.sqrt(r['independent_covariance'][component][component]+r['synchronous_reference_covariance'][component][component]))
        axis.errorbar(bins,means,yerr=errors,fmt='o-',label=label,color=color,capsize=3)
    axis.axhline(0,lw=.8,color='.5');axis.set(xlabel=r'$r_0$ bin center (mag)',ylabel='Offset from bright reference (mas yr⁻¹)',title='Background-subtracted observed CMD mixture');axis.legend(frameon=False)
    fig.suptitle('NGC 5466 · independent deep photometric selection\nDepth/mask-aware areas; candidate mixture is not certified cluster membership',fontsize=14)
    save(fig,output/'v20_ngc_deep_summary.png')


def cluster_bias_figure(data,output):
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    names=['Old 197 raw','Subtract all local QSO','Response-matched raw','Response-matched translated']
    response=data['local_candidate_cluster_validation'];ref=np.asarray(data['bright_reference_masyr'])
    values=[data['original_faint']['weighted_mean'],data['original_faint_qso_subtraction'],response['raw']['weighted_mean'],ref+response['translated_offset']]
    axis=axes[0,0]
    for j,(label,color) in enumerate([('East','#4682a9'),('North','#b05d43')]):
        axis.plot(np.asarray(values)[:,j]-ref[j],np.arange(4)+(j-.5)*.15,'o',label=label,color=color)
    axis.axvline(0,lw=.8,color='.5');axis.set(yticks=range(4),yticklabels=names,xlabel='Offset from bright reference (mas yr⁻¹)',title='Fixed stellar cohorts · diagnostic shifts only');axis.invert_yaxis();axis.legend(frameon=False)
    axis=axes[0,1]
    categories=[('QSO validation',data['local_qso']['validation']['identity_validation']),('QSO confirmation',data['local_qso']['validation']['identity_confirmation']),('Bright stars',data['local_candidate_bright_validation'])]
    for j,(label,color) in enumerate([('East','#4682a9'),('North','#b05d43')]):
        for i,(name,row) in enumerate(categories):
            points=[row[t]['median'][j] for t in ('before','after')]
            axis.plot([i-.13+j*.26]*2,points,'-',color=color,alpha=.6)
            axis.scatter(i-.13+j*.26,points[0],marker='o',color=color,label=label+' before' if i==0 else None)
            axis.scatter(i-.13+j*.26,points[1],marker='x',color=color,label=label+' after' if i==0 else None)
    axis.axhline(0,lw=.8,color='.5');axis.set(xticks=range(3),xticklabels=[f'{name}\nN={row["n"]:,}' for name,row in categories],ylabel='Median residual (mas yr⁻¹)',title='Training-QSO shift does not transfer to stars');axis.legend(frameon=False,fontsize=8,ncol=2)
    axis=axes[1,0]
    for j,(label,color) in enumerate([('East','#4682a9'),('North','#b05d43')]):
        bins=list(data['bright_gaia']['color_bins'].values());x=[(b['lower_gi']+b['upper_gi'])/2 for b in bins]
        y=[b['residual']['median'][j] for b in bins];bounds=np.asarray([b['residual']['median_object_interval95'][j] for b in bins])
        axis.errorbar(x,y,yerr=[np.asarray(y)-bounds[:,0],bounds[:,1]-y],fmt='o-',color=color,label=label,capsize=3)
    axis.axhline(0,lw=.8,color='.5');axis.set(xlabel='SDSS g − i (mag)',ylabel='Median C0 − bright Gaia (mas yr⁻¹)',title='Bright-only color dependence · conditional object intervals');axis.legend(frameon=False)
    axis=axes[1,1];labels=['Old bright','Bright + old faint'];values=[data['old_bright_mean'],data['old_combined_mean']]
    for j,(label,color) in enumerate([('East','#4682a9'),('North','#b05d43')]):
        axis.plot(range(2),[v['independent_standard_error'][j] for v in values],'o-',color=color,label=label+' independent')
        axis.plot(range(2),[v['reference_common_standard_error'][j] for v in values],'s--',color=color,label=label+' shared reference')
    axis.set(xticks=range(2),xticklabels=[f'{label}\nN={v["n"]:,}' for label,v in zip(labels,values)],ylabel='Conditional mean error (mas yr⁻¹)',title='Shared reference draws remain synchronized');axis.legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle('NGC 5466 · two-axis systematics and a closed local-correction experiment',fontsize=14)
    save(fig,output/'v20_ngc_stellar_bias_summary.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    root=args.root;out=root/'figures';out.mkdir(exist_ok=True)
    p=root/'results/summary'
    blind_figure(load(p/'v19_blind_quality_summary.json'),load(p/'v19_production_replay_quality_summary.json'),out)
    funnel_figure(load(p/'v20_selection_summary.json'),out)
    spatial_figure(load(p/'v20_quality_calibration_summary.json'),out)
    cluster=load(p/'v20_ngc_motion_summary.json')
    cluster_deep_figure(cluster,out)
    cluster_bias_figure(cluster,out)


if __name__=='__main__':main()
