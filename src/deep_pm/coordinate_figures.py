"""Four scientific figures from reviewed public aggregate summaries only."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OLD='#777777';B1='#2879a5';B2='#c15c35';SKY='#376d55'


def load(p):return json.loads(p.read_text())
def save(fig,path):
    fig.savefig(path,dpi=250,bbox_inches='tight',facecolor='white');plt.close(fig)


def scale_figure(science,out):
    fig,axes=plt.subplots(2,2,figsize=(11.2,8),layout='constrained')
    for i,low in enumerate(('21','22')):
        bins=science['scientific'][low]['angle_bins'];n=science['scientific'][low]['fixed_members']['n']
        for axis,component in zip(axes[i],range(2)):
            for method,label,color,marker in [('old','Archived geometry',OLD,'o'),('B1','B1: same PM, correct covariance',B1,'s'),('B2','B2: fixed members, full 2D fit',B2,'^')]:
                values=[]
                for name,b in bins.items():
                    if b['n']<5:continue
                    a,z=map(float,name.split('_'));values.append(((a+z)/2,b['methods'][method]['standardized_nmad'][component],b['n']))
                if values:
                    x,y,counts=zip(*values);axis.plot(x,y,marker+'-',color=color,label=label,lw=1.4,ms=5)
                    if method=='old':
                        for a,b,c in values:axis.annotate(f'N={c}',(a,b),xytext=(0,9),textcoords='offset points',ha='center',fontsize=7,color='.35')
            axis.axhline(1,color='.35',lw=.8,ls='--');axis.set(xlim=(0,70),ylim=(0,2.8),xlabel='Angular distance to actual graph centre (deg)',
                ylabel='NMAD(PM / reported error)',title=f'{low}–{int(low)+1} · '+('East' if component==0 else 'North')+f' · fixed QSO N={n}')
            axis.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Native → graph → ICRS: correct covariance removes strong far-centre error compression\nSame identities and detections; no zero-point or empirical error multiplier fitted',fontsize=13)
    save(fig,out/'v21_coordinate_scale_summary.png')


def association_figure(science,out):
    cores=science['cores'];theta=np.array([r['theta_deg'] for r in cores]);order=np.argsort(theta)
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.8),layout='constrained')
    old=np.array([r['association']['members']['old']['sources'] for r in cores]);b2=np.array([r['association']['members']['B2_selected']['sources'] for r in cores]);sky=np.array([r['association']['members']['B3_spherical']['sources'] for r in cores])
    ax=axes[0,0]
    ax.scatter(theta,100*(b2/old-1),c=B2,marker='^',label='2D/ICRS quality change, old radius')
    ax.scatter(theta,100*(sky/b2-1),c=SKY,marker='s',label='Additional change from sky radius')
    ax.axhline(0,color='.5',lw=.8);ax.set(xlabel='Distance to graph centre (deg)',ylabel='Source-count change (%)',title=f'16 registered cores · {old.sum():,} → {b2.sum():,} → {sky.sum():,}');ax.legend(frameon=False,fontsize=8)
    ax=axes[0,1];t=np.linspace(0,65,150);ax.plot(t,1000*np.cos(np.deg2rad(t))**2,color=OLD,label='Old radial sky radius');ax.plot(t,1000*np.cos(np.deg2rad(t)),color=OLD,ls='--',label='Old transverse sky radius');ax.axhline(1000,color=SKY,label='B3: physical sky radius')
    ax.set(xlabel='Distance to graph centre (deg)',ylabel='True sky radius (mas)',title='Meaning of a nominal 1000 mas gate');ax.legend(frameon=False,fontsize=8)
    ax=axes[1,0]
    for low,color,marker in [('21',B1,'o'),('22',B2,'^')]:
        a=[];b=[]
        for r in cores:
            total=lambda name:sum(r['numerical'][name]['magnitude_bins'].get(low,{}).values())
            a.append(total('B2_selected'));b.append(total('B3_spherical'))
        a,b=np.array(a),np.array(b);ax.scatter(theta,b-a,c=color,marker=marker,label=f'{low}–{int(low)+1}: {a.sum():,} → {b.sum():,}')
    ax.axhline(0,color='.5',lw=.8);ax.set(xlabel='Distance to graph centre (deg)',ylabel='B3 minus B2 source count',title='Dark solutions · same original 0.8 morphology');ax.legend(frameon=False,fontsize=8)
    ax=axes[1,1]
    added=[r['association']['reference_selection']['spherical']['reference_sources']-r['association']['reference_selection']['old_planar']['reference_sources'] for r in cores]
    ax.scatter(theta,added,c=SKY);ax.axhline(0,color='.5',lw=.8);ax.set(xlabel='Distance to graph centre (deg)',ylabel='Change in local reference identities',title='Reference entry changes require a new joint graph')
    fig.suptitle('Physical angular gates restore candidates at large projection distance\nConditional on archived C0; these counts are research replays, not certified new stars or new survey area',fontsize=12)
    save(fig,out/'v21_spherical_selection_summary.png')


def stellar_figure(stellar,group,out):
    fig,axes=plt.subplots(2,2,figsize=(11.8,8),layout='constrained')
    bands=group['band_year_airmass']['instrument_band'];keys=sorted(bands)
    for ax,component,name in [(axes[0,0],1,'North'),(axes[0,1],0,'East')]:
        for j,key in enumerate(keys):
            d=bands[key];stats=d['night_median_distribution'];point=(stats.get('median') or d['descriptive_pair_distribution'].get('median'))
            if point is None:continue
            bounds=d.get('equal_night_median_sensitivity95');color=B1 if key.startswith('nan') else B2
            if bounds is not None:
                lo,hi=np.array(bounds)[component];ax.vlines(j,lo,hi,color=color,lw=2)
            ax.plot(j,point[component],'o',color=color)
        ax.axhline(0,color='.5',lw=.8);ax.set(xticks=range(len(keys)),xticklabels=[('SDSS:r' if k=='nan:r' else k)+f'\n{bands[k]["pairs"]} pairs / {bands[k]["independent_UTC_nights"]} nights' for k in keys],
            ylabel='Red − blue position residual (mas)',title=f'{name} · equal-night medians; 95% night sensitivity')
        ax.tick_params(axis='x',labelsize=7)
    ax=axes[1,0];data=stellar['bright_motion']['19_21'];x=np.arange(2)
    for name,offset,color in [('raw',-.10,OLD),('c0',.10,B2)]:
        y=[data[c][name]['median'][1] for c in ('blue','red')];bounds=np.array([data[c][name]['median_object_interval95'][1] for c in ('blue','red')]);ax.vlines(x+offset,bounds[:,0],bounds[:,1],color=color);ax.plot(x+offset,y,'o-',color=color,label='Published positions' if name=='raw' else 'After frozen C0')
    ax.axhline(0,color='.5',lw=.8);ax.set(xticks=x,xticklabels=[f'Blue g−i<0.5\nN={data["blue"]["raw"]["n"]}',f'Red g−i≥2\nN={data["red"]["raw"]["n"]}'],
        ylabel='PM − bright Gaia, North (mas yr⁻¹)',title='Fixed bright controls · SDSS r=19–21');ax.legend(frameon=False,fontsize=8)
    ax=axes[1,1];held=stellar['held_identity_partial_K_prediction']
    for j,(name,d) in enumerate(held.items()):
        a=d['before']['absolute_radial_p68'];b=d['diagnostic_residual_after_partial_prediction']['absolute_radial_p68'];ax.plot([j-.1,j+.1],[a,b],color=B2,lw=1.7);ax.plot(j-.1,a,'o',color=OLD,label='Before' if j==0 else None);ax.plot(j+.1,b,'s',color=B2,label='After partial K prediction' if j==0 else None)
    ax.set(xticks=range(len(held)),xticklabels=[f'{k}\nN={v["n"]}' for k,v in held.items()],ylabel='Radial PM residual P68 (mas yr⁻¹)',title='Training-position colour signal → held-identity PM');ax.legend(frameon=False,fontsize=8);ax.text(.03,.04,'≥50% K-amplitude support only; residual zero remains\nDiagnostic only · no production colour/DCR model',transform=ax.transAxes,fontsize=8)
    fig.suptitle('A residual colour signal exists in published NSC positions\nSame physical group, brightness, position and crowding support; common C0 cancels from the colour contrast',fontsize=12)
    save(fig,out/'v21_stellar_attribution_summary.png')


def selection_figure(ngc,candidate,identity,out):
    fig,axes=plt.subplots(2,2,figsize=(12.1,8.2),layout='constrained')
    names={'recovered':'Recovered','pointlike_morphology':'Morphology','no_published_SDSS_detection':'No published SDSS','SDSS_quality_or_distance':'SDSS quality / distance','quality':'Fit quality','time_baseline':'Time support','no_published_NSC_detection':'No published NSC','NSC_physical_quality_or_distance':'NSC quality / distance'}
    colors={'recovered':SKY,'pointlike_morphology':B2,'no_published_SDSS_detection':OLD,'SDSS_quality_or_distance':'#9ab3c3','quality':'#d3aa83','time_baseline':'#c5c5b0','no_published_NSC_detection':'#555555','NSC_physical_quality_or_distance':'#bdcdd8'}
    handles={}
    for i,low in enumerate(('22.0','22.5')):
        ax=axes[i,0];d=ngc['bins'][low]
        for j,method in enumerate(('old','geometry')):
            left=0
            for reason in names:
                count=d['models'][method].get(reason,0)
                if not count:continue
                bars=ax.barh(j,count,left=left,color=colors.get(reason,'.7'),height=.55,label=names.get(reason,reason));handles[names.get(reason,reason)]=bars[0]
                if count>=12:ax.text(left+count/2,j,str(count),ha='center',va='center',color='white' if reason in ('recovered','pointlike_morphology','no_published_SDSS_detection') else '.15',fontsize=9)
                left+=count
        ax.set(yticks=[0,1],yticklabels=['Archived','Geometry'],xlabel='Frozen optical candidate count',xlim=(0,d['optical_candidates']),title=f'NGC r₀={low}–{float(low)+.5:g} · N={d["optical_candidates"]}');ax.invert_yaxis()
    axes[1,0].legend(handles.values(),handles.keys(),frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,-.23),ncol=2)
    ax=axes[0,1];rows=candidate['cores'];den=[sum(r['training']['added']['training_sources'] for r in rows),sum(r['training']['added']['opportunities']['geometric']['candidates'] for r in rows),sum(r['training']['added']['opportunities']['catalogue_10sigma']['candidates'] for r in rows)];hit=[sum(r['training']['added']['unique_reproductions'] for r in rows),sum(r['training']['added']['opportunities']['geometric']['unique_reproductions'] for r in rows),sum(r['training']['added']['opportunities']['catalogue_10sigma']['unique_reproductions'] for r in rows)]
    fraction=np.array(hit)/den;ax.bar(range(3),fraction,color=[OLD,B1,SKY]);
    for j,(h,n) in enumerate(zip(hit,den)):ax.text(j,fraction[j]+.02,f'{h:,}/{n:,}\n{fraction[j]:.1%}',ha='center',fontsize=9)
    ax.set(ylim=(0,1.13),xticks=range(3),xticklabels=['All training\nadded candidates','Covered by\nheld-year CCD','Covered + catalogue\n10σ depth'],ylabel='Unique held-year position reproduction',title='Sole 0.6 candidate · full-source competition');ax.text(.02,.02,'CCD internal effective pixels unknown; no star-purity claim',transform=ax.transAxes,fontsize=8)
    ax=axes[1,1]
    for j,(name,label,color) in enumerate([('original_hash_control','Original hash control',B1),('added','Added candidates',B2)]):
        counts=Counter();n=0
        for r in identity['cores']:
            v=r['classes'][name]['all'];n+=v['n'];counts.update(v['shape_counts'])
        values=[counts['PSF'],counts['REX'],sum(counts[k] for k in ('EXP','DEV','SER'))];values.append(n-sum(values))
        ax.bar(np.arange(4)+(j-.5)*.34,np.array(values)/n,width=.32,color=color,label=f'{label} N={n:,}')
    ax.set(xticks=range(4),xticklabels=['PSF','REX','EXP/DEV/SER','Other / no unique\nexternal label'],ylim=(0,1),ylabel='Fraction in LS independent-processing morphology',title='External shape evidence · some imaging shared');ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed dark optical denominator and the completed sole morphology trial\nNGC deepest recovery remains limited by early positions and morphology; position recovery ≠ verified stellar correctness',fontsize=12)
    save(fig,out/'v21_dark_selection_summary.png')


def run(root):
    p=root/'results/summary';out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'medium'})
    science=load(p/'v21_scientific_comparison_summary.json')
    scale_figure(science,out);association_figure(science,out)
    stellar_figure(load(p/'v21_stellar_attribution_summary.json'),load(p/'v21_stellar_group_summary.json'),out)
    selection_figure(load(p/'v21_ngc415_trace_summary.json'),load(p/'v21_candidate_completion_summary.json'),load(p/'v21_candidate_identity_summary.json'),out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
