"""Render standalone group-meeting figures from the frozen and derived aggregates."""
import argparse
import json
from pathlib import Path

import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import ListedColormap, LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.special import erf

if __package__:
    from .meeting_stats import CONFIG, METHODS, SNAPSHOT
else:
    from meeting_stats import CONFIG, METHODS, SNAPSHOT

COLORS = ('#777777', '#D55E00', '#009E73', '#0072B2')
LABELS = ('NSC DR2', 'Xiao et al. (2025)', 'SDSS+NSC r only', 'SDSS+NSC all bands')
MARKERS = ('x','s','^','o')
LINESTYLES = (':','--','-.','-')
AXES = (r'$\mu_{\alpha *}$', r'$\mu_\delta$')
BANDS={21:'21 ≤ r < 22',22:'22 ≤ r < 23'}
UNIT='mas yr⁻¹'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#183040','axes.titlecolor':'#142c3c',
                     'axes.edgecolor':'#58717e','text.color':'#142c3c',
                     'xtick.color':'#294858','ytick.color':'#294858','pdf.fonttype':42})


def read(path):
    return json.loads(path.read_text())


def save(fig, directory, name):
    directory.mkdir(exist_ok=True, parents=True)
    fig.savefig(directory/f'{name}.pdf',dpi=320,facecolor='white')
    fig.savefig(directory/f'{name}.png',dpi=320,facecolor='white')
    plt.close(fig)


def title(fig, name, detail):
    fig.suptitle(name, fontweight='bold', fontsize=19, y=.985)
    fig.text(.5,.929,detail,ha='center',va='center',fontsize=10,color='#49616f')


def grid(ax, direction='y'):
    ax.grid(axis=direction,alpha=.23,color='#9aaeb9')
    ax.set_axisbelow(True)


def for_sky(ax):
    ax.set_xticks(np.deg2rad(np.arange(-150,180,30)))
    ax.set_xticklabels([f'{(180-x)%360}°' for x in range(-150,180,30)])
    ax.set_yticks(np.deg2rad([-60,-30,0,30,60]))
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.35,linewidth=.55,color='#829aac')
    ax.spines['geo'].set_edgecolor('#94a5b0')


def sky_elements(ax, nside, values, cmap, norm, coverage=None, local=None):
    vals=np.full(hp.nside2npix(nside),np.nan)
    for key,val in values.items():
        if val is not None:
            vals[int(key)]=val
    if local is None:
        x=np.linspace(-180,180,1441)
        y=np.linspace(-90,90,721)
        xx,yy=np.meshgrid((x[1:]+x[:-1])/2, (y[1:]+y[:-1])/2)
        angle = x,y
        pix=hp.ang2pix(nside,(180-xx)%360,yy,lonlat=True)
        draw_x,draw_y=(np.deg2rad(z) for z in angle)
    else:
        left,right,bottom,top=local
        x=np.linspace(left,right,701)
        y=np.linspace(bottom,top,301)
        xx,yy=np.meshgrid((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2)
        pix=hp.ang2pix(nside,xx%360,yy,lonlat=True)
        draw_x,draw_y=x,y
        ax.set(xlim=(right,left),ylim=(bottom,top),xlabel='ICRS RA',ylabel='Declination')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda angle, _: f'{angle % 360:g}°'))
        ax.grid(alpha=.20,linewidth=.55)
    if coverage:
        status=np.zeros(len(vals))
        for key,val in coverage.items():
            status[int(key)]=val
        ax.pcolormesh(draw_x,draw_y,np.ma.masked_where(status[pix]==0,status[pix]),
                     cmap=ListedColormap(['#e3e9ed','#aebac3']), vmin=1,vmax=2,
                     shading='flat',rasterized=True,zorder=1)
    image=ax.pcolormesh(draw_x,draw_y,np.ma.masked_invalid(vals[pix]),cmap=cmap,norm=norm,
                       shading='flat',rasterized=True,zorder=2)
    if local is None:
        for_sky(ax)
    return image


def footprint(science, counts, output):
    pix_counts={int(p):n for p,n in counts.items()}
    assert sum(pix_counts.values())==science['research_fits']==11862169
    scale=Normalize(2.3,3.45)
    values={p:np.log10(n) for p,n in pix_counts.items()}
    fig=plt.figure(figsize=(14,8))
    fig.subplots_adjust(top=.86,bottom=.18,left=.07,right=.95,hspace=.38,wspace=.34)
    g=fig.add_gridspec(2,3,height_ratios=[1.45,1])
    ax=fig.add_subplot(g[0,:],projection='aitoff')
    im=sky_elements(ax,128,values,'viridis',scale)
    for i,(name,bounds) in enumerate([('Around the RA seam',(-34,32,-11,12)),
                                     ('Western and northern windows',(-43,34,11,32)),
                                     ('Eastern and northern windows',(115,225,-12,41))]):
        ax=fig.add_subplot(g[1,i])
        sky_elements(ax,128,values,'viridis',scale,local=bounds)
        ax.set_title(name,fontsize=10)
    bar=fig.add_axes([.29,.070,.42,.018])
    fig.colorbar(im,cax=bar,orientation='horizontal',extend='both',label=r'$\log_{10}$(sources per HEALPix pixel)')
    title(fig,'Where the catalogue has measurements',
          '2,154.184 deg² registered   ·   2,132.865 deg² joint opportunity   ·   11,862,169 fits   ·   J2015.5')
    save(fig,output,'01_footprint_density')
    fig,ax=plt.subplots(figsize=(12,6.6),subplot_kw={'projection':'aitoff'})
    fig.subplots_adjust(top=.82,bottom=.12,left=.04,right=.96)
    for_sky(ax)
    values={p:1 for p in pix_counts}
    sky_elements(ax,128,values,ListedColormap(['#0072B2']),Normalize(0,1))
    title(fig,'SDSS + NSC research footprint',
          'ICRS Aitoff   ·   occupied NSIDE 128 / RING pixels   ·   5.22% of the sky registered')
    save(fig,output,'01_footprint_density_outline')


def spatial_figure(qso, output, which, nside=32, local=None):
    choice={
        'gain':('P68 gain on strictly common QSOs','P68 reduction [%]',80,'RdBu','02_qso_improvement_aitoff'),
        'count':('Spatial support of the common QSO sample','Common QSOs per pixel',None,'viridis','02_qso_improvement_qso_counts'),
        'median':('Spatial proper-motion zero point',f'QSO median [{UNIT}]',3,'RdBu_r','03_qso_precision_and_zero_aitoff'),
        'zero_change':('Zero-point change after adding bands',f'Median(all) − median(r only) [{UNIT}]',3,'RdBu_r','03_qso_precision_and_zero_all_minus_r'),
    }[which]
    name, units, limit, cmap, filename=choice
    if local:
        filename += '_local_nside64'
    ncols=1 if which=='count' else 2
    fig, axs=plt.subplots(2,ncols,squeeze=False,figsize=(13 if ncols==2 else 10,8.2),
                           subplot_kw={} if local else {'projection':'aitoff'})
    fig.subplots_adjust(top=.847,bottom=.19,left=.055,right=.95,hspace=.47,wspace=.14)
    norm=TwoSlopeNorm(0,-limit,limit) if limit is not None else LogNorm(1, max(
        v['n'] for b in qso['bands'].values() for v in b['spatial'][str(nside)].values()))
    for band, axes in zip(BANDS,axs):
        d=qso['bands'][str(band)]
        p=d['spatial'][str(nside)]
        support={k:2 if c['n'] else 1 for k,c in p.items()}
        for col, ax in enumerate(axes):
            if which=='count':
                vals={k:c['n'] for k,c in p.items() if c['n']}
                subtitle=BANDS[band]
            else:
                vals={k:(c[which][-1][col] if which=='median' else c[which][col]) for k,c in p.items() if c['display']}
                subtitle=(LABELS[1-col] if which=='gain' else AXES[col])+'   ·   '+BANDS[band]
            im=sky_elements(ax,nside,vals,cmap,norm,support,local)
            ax.set_title(subtitle,pad=13,fontsize=11,fontweight='bold')
            if which!='count':
                n=sum(c['n'] for c in p.values() if c['display'])
                scope='full spatial grid' if local else 'displayed grid'
                ax.text(.5,-.095 if not local else .025,
                        f'N above threshold in {scope} {n:,} / {d["n"]:,}',
                        transform=ax.transAxes,ha='center',fontsize=8)
    box=fig.add_axes([.30,.112,.40,.019])
    fig.colorbar(im,cax=box,orientation='horizontal',label=units,extend='both' if limit else 'neither')
    swatches=[Patch(facecolor='#e3e9ed',label='Catalogue support, zero common QSOs')]
    if which!='count':
        swatches.insert(0,Patch(facecolor='#aebac3',label='1–29 common QSOs'))
    fig.legend(handles=swatches,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,fontsize=9)
    subtitle=f'ICRS Aitoff   ·   HEALPix NSIDE {nside} / RING   ·   threshold N ≥ 30 for science panels'
    if local:
        subtitle='ICRS local view   ·   HEALPix NSIDE 64 / RING   ·   threshold N ≥ 30'
    title(fig,name,subtitle)
    save(fig,output,filename)


def method_legend(fig):
    lines=[Line2D([0],[0],color=c,marker=m,linestyle=l,label=name)
           for c,m,l,name in zip(COLORS,MARKERS,LINESTYLES,LABELS)]
    fig.legend(handles=lines,loc='upper center',bbox_to_anchor=(.5,.883),ncol=4,frameon=False,fontsize=10)


def precision_page(qso, output, pointlike=False):
    fig, axs=plt.subplots(3,2,figsize=(12,8.4),sharex=True)
    fig.subplots_adjust(top=.812,bottom=.10,left=.10,right=.96,hspace=.23,wspace=.25)
    coarse=[dict(qso['bands'][str(b)][('pointlike' if pointlike else 'all')],
                 n=qso['bands'][str(b)]['pointlike']['n'] if pointlike else qso['bands'][str(b)]['n'], mid=b+.5) for b in BANDS]
    samples=coarse if pointlike else [dict(mid=f['lower']+.25,**f) for b in BANDS for f in qso['bands'][str(b)]['halves']]
    x=[v['mid'] for v in samples]
    for j, (color,mark,line) in enumerate(zip(COLORS,MARKERS,LINESTYLES)):
        for i,(quantity,col) in enumerate([('p68',None),('nmad',0),('median',0)]):
            for axis,ax in enumerate(axs[i, :] if col is not None else axs[i, :1]):
                y=[v[quantity][j] if col is None else v[quantity][j][axis] for v in samples]
                ax.plot(x,y,color=color,marker=mark,ls=line,lw=1.5,ms=5)
                if not pointlike:
                    val=[v[quantity][j] if col is None else v[quantity][j][axis] for v in coarse]
                    ax.scatter([v['mid'] for v in coarse],val,facecolors='none',edgecolors=color,marker='o',s=46,zorder=5)
    for row in axs:
        for ax in row:
            grid(ax)
    axs[0,0].set(ylabel=f'2D P68 [{UNIT}]')
    axs[0,1].axis('off')
    values='\n'.join(f'{BANDS[b]}   N = {v["n"]:,}\n   NSC {v["p68"][0]:.2f}    Xiao {v["p68"][1]:.2f}    r {v["p68"][2]:.2f}    all {v["p68"][3]:.2f}'
                     for b,v in zip(BANDS,coarse))
    axs[0,1].text(.02,.92,values,transform=axs[0,1].transAxes,va='top',fontsize=11,linespacing=1.42)
    for axis in range(2):
        axs[1,axis].set_ylabel(f'{AXES[axis]} NMAD [{UNIT}]')
        axs[2,axis].set(xlabel='Spectroscopic reference r [mag]',ylabel=f'{AXES[axis]} median [{UNIT}]')
        axs[2,axis].axhline(0,c='#728490',ls=':',lw=.9)
    axs[2,0].set_xticks(x,[f'{v["mid"]:.2f}\nN={v["n"]:,}' for v in samples])
    title(fig,'Four methods on the same spectroscopic quasars',
          'Pointlike r morphology (median CLASS_STAR ≥ 0.8)' if pointlike else
          'Filled marks give 0.5 mag bins. Open circles give the original 1 mag bins.')
    method_legend(fig)
    save(fig,output,'03_qso_precision_and_zero'+('_pointlike' if pointlike else ''))


def condition_numbers(science, derived, band, tag, j):
    if tag.startswith('joint_'):
        source=derived['bands'][str(band)]['conditions'][tag]
        error=source['sensitivity'] if j==1 else {}
        return source['n'],100*(1-source['p68'][-1]/source['p68'][j]),error
    source=science['aggregate']['known_stationary'][f'r{band}__{tag}']
    common=source['strict_published_common']
    if common['n']<100:
        return None
    allpm=common['methods']['all_nsc']['radial_abs68']
    base=common['methods'][METHODS[j]]['radial_abs68']
    shared=source['shared_group_sensitivity'].get('paired_radial_gain',{}).get(METHODS[j],{}).get('visit_weighted',{})
    if j==2:
        pair=common['paired']['all_nsc_vs_r_physical_quality']['fraction95'][-1]
        shared={'objects':{'relative_gain95':pair}}
    return common['n'],100*(1-allpm/base),shared


def conditions(science, derived, output, j, instrument=False):
    groups=[('all','All common QSOs'),('one_sdss_run','One SDSS run'),('multiple_sdss_runs','Multiple SDSS runs'),
            ('nsc_span_lt2','NSC span <2 yr'),('nsc_span_ge2','NSC span ≥2 yr'),
            ('nsc_nights_le5','≤5 NSC nights'),('nsc_nights_gt5','>5 NSC nights'),
            ('joint_lt12','Joint span <12 yr'),('joint_ge12','Joint span ≥12 yr')]
    if instrument:
        camera={'c4d':'DECam','k4m':'Mosaic3','ksb':'90Prime'}
        instrument_tags=sorted({k.split('__')[-1] for k,v in science['aggregate']['known_stationary'].items()
                                if '__instrument_' in k and v['strict_published_common']['n']>=100})
        groups=[('all','All common QSOs'),*((k,' + '.join(camera.get(c,c) for c in k[11:].split('+'))) for k in instrument_tags)]
    fig,axs=plt.subplots(1,2,figsize=(13.5,6.7),sharex=True)
    fig.subplots_adjust(top=.83,bottom=.15,left=.22,right=.97,wspace=.64)
    observed=[]
    for band,ax in zip(BANDS,axs):
        labels=[]
        for tag,label in groups:
            measurement=condition_numbers(science,derived,band,tag,j)
            if measurement is None:
                continue
            n,gain,errors=measurement
            labels.append(f'{label}  ({n:,})')
            y=len(labels)-1
            observed.append(gain)
            for key,delta,c in [('sdss_run',-.12,'#6659a2'),('nsc_night',.12,'#4e9aaa'),('objects',0,'#799298')]:
                interval=errors.get(key,{}).get('relative_gain95')
                if interval is not None:
                    observed.extend(100*np.asarray(interval))
                    ax.hlines(y+delta,*[100*v for v in interval],color=c,lw=2)
            ax.plot(gain,y,'o',c='#0072B2',ms=6)
        ax.axvline(0,color='#4c5e66',ls=':',lw=1)
        ax.set(yticks=range(len(labels)),yticklabels=labels,
               ylim=(len(labels)-.5,-.5),xlabel='2D P68 reduction [%]',title=BANDS[band])
        ax.tick_params(labelsize=9)
        grid(ax,'x')
    low=min(-5,min(observed)-3); hi=max(observed)+4
    axs[0].set_xlim(low,hi)
    items=[Line2D([0],[0],color='#0072B2',marker='o',ls='none',label='Matched-object gain')]
    if j in (0,1):
        items.extend([Line2D([0],[0],color='#6659a2',label='SDSS-run reweighting 95% sensitivity'),
                      Line2D([0],[0],color='#4e9aaa',label='NSC-night reweighting 95% sensitivity')])
    else:
        items.append(Line2D([0],[0],color='#799298',label='Object-paired 95% interval where archived'))
    fig.legend(handles=items,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=9)
    title(fig, f'P68 gain over {LABELS[j]}',
          'Strictly common spectroscopic QSOs   ·   observing categories overlap')
    suffix='_instruments' if instrument else ('_vs_r_only' if j==2 else '_vs_nsc' if j==0 else '')
    save(fig,output,'06_gain_by_observing_conditions'+suffix)


def comparison_plot(setup, method, output, scope, width):
    sample=setup['samples'][scope]
    displayed=sample['panels'][width]
    fig,axs=plt.subplots(2,2,figsize=(10.5,8.6),sharex=True)
    fig.subplots_adjust(top=.84,left=.10,right=.86,bottom=.11,wspace=.30,hspace=.24)
    label=LABELS[METHODS.index(method)]
    pm=displayed['pm_edges']
    delta=displayed['delta_edges']
    color_max=max(a[k][i][j] for a in displayed['axes'] for k in ('density','difference')
                  for i in range(len(a[k])) for j in range(len(a[k][i])))
    norm=LogNorm(1,max(2,color_max))
    for axis,entry in enumerate(displayed['axes']):
        ax, res=axs[:,axis]
        style=dict(shading='auto',norm=norm,cmap='magma_r',rasterized=True)
        mesh=ax.pcolormesh(pm,pm,np.ma.masked_equal(np.array(entry['density']).T,0),**style)
        res.pcolormesh(pm,delta,np.ma.masked_equal(np.array(entry['difference']).T,0),**style)
        outline=[patheffects.withStroke(linewidth=2.6,foreground='white',alpha=.86)]
        ax.plot([pm[0],pm[-1]],[pm[0],pm[-1]],c='#526878',ls='--',lw=.9,path_effects=outline)
        res.axhline(0,c='#526878',ls='--',lw=1)
        xx=np.array([np.mean(x['boundaries']) for x in entry['profile']])
        q=np.array([x['quantiles'] if x['quantiles'] is not None else [np.nan]*3 for x in entry['profile']])
        res.plot(xx,q[:,1],c='#0072B2',lw=1.25,label='Bin median',path_effects=outline)
        res.fill_between(xx,q[:,0],q[:,2],color='#0072B2',alpha=.22,label='Object differences, 16–84%')
        ax.set_aspect('equal',adjustable='box')
        ax.set(xlim=(pm[0],pm[-1]),ylim=(pm[0],pm[-1]),title=AXES[axis],
               ylabel=f'SDSS+NSC all bands [{UNIT}]')
        res.set(xlabel=f'{label} [{UNIT}]',ylabel=f'All bands − reference [{UNIT}]',
                ylim=(delta[0],delta[-1]))
        for panel,key in [(ax,'outside_density'),(res,'outside_difference')]:
            count=entry[key]
            panel.text(.96,.04,f'Outside axes {count:,} ({100*count/sample["n"]:.2f}%)',transform=panel.transAxes,
                       ha='right',va='bottom',fontsize=8,bbox=dict(fc='white',ec='none',alpha=.85,pad=1.5))
    fig.colorbar(mesh,cax=fig.add_axes([.90,.23,.025,.50]),label='Sources per bin (log color scale)')
    axs[1,0].legend(loc='upper left',fontsize=8,frameon=True,facecolor='white')
    label_scope='All eligible matches' if scope=='all' else BANDS[int(scope[1:])]
    route='positional matches at J2015.5' if 'radius_arcsec' in setup else 'physical member matches'
    title(fig,f'Catalogue agreement with {label}',
          f'{label_scope}   ·   N = {sample["n"]:,} one-to-one {route}' +
          ('   ·   extended axes' if width=='tail' else ''))
    base='04_all_sources_comparison_'+method.lower()
    suffix=('_'+scope if scope!='all' else '')+('_tail' if width=='tail' else '')
    save(fig,output,base+suffix)


def error_coverage(qso,output):
    fig,axs=plt.subplots(2,3,figsize=(13.8,7.6))
    fig.subplots_adjust(top=.785,bottom=.12,left=.075,right=.97,hspace=.51,wspace=.32)
    for band,row in zip(BANDS,axs):
        all_data=qso['bands'][str(band)]['coverage']['all']
        x=np.asarray(all_data['density_edges'])
        mid=(x[1:]+x[:-1])/2
        for axis,(c,lab) in enumerate(zip(('#0072B2','#D55E00'),AXES)):
            for mode,ls in [('original','-'),('median_removed','--')]:
                values=np.array(all_data[mode]['density_counts'][axis])
                row[0].plot(mid,values/(all_data['n']*np.diff(x)),c=c,ls=ls,lw=1.25,label=lab)
            for panel,key,name in [(row[1],'joint_lt12','Joint span <12 yr'),(row[2],'joint_ge12','Joint span ≥12 yr')]:
                d=qso['bands'][str(band)]['coverage'][key]
                for mode,ls in [('original','-'),('median_removed','--')]:
                    panel.plot(d['thresholds'],np.array(d[mode]['cdf'][axis])*100,c=c,ls=ls,lw=1.25)
                panel.set(xlim=(0,3),ylim=(0,101),title=f'{name}   ·   N={d["n"]:,}',
                          xlabel=r'Threshold $k$',ylabel=r'Objects with $|z| \leq k$ [%]')
        row[0].plot(mid,np.exp(-mid**2/2)/np.sqrt(2*np.pi),':',c='#555555')
        row[0].set(xlabel=r'Normalized motion $z$',ylabel='Density',xlim=(-5,5),
                   title=f'{BANDS[band]}   ·   N={all_data["n"]:,}')
        for ax in row[1:]:
            u=np.asarray(all_data['thresholds'])
            ax.plot(u,100*erf(u/np.sqrt(2)),':',c='#555555')
            for k,f in [(1,68.27),(2,95.45)]:
                ax.plot(k,f,'+',ms=7,c='#555555')
        for ax in row:
            grid(ax)
    fig.legend(handles=[Line2D([0],[0],color=c,label=l) for c,l in
                        zip(('#0072B2','#D55E00'),AXES)]+[
                        Line2D([0],[0],color='#536674',label='Relative to zero'),
                        Line2D([0],[0],color='#536674',ls='--',label='After subtracting motion median'),
                        Line2D([0],[0],color='#536674',ls=':',label='Normal reference')],
               loc='upper center',bbox_to_anchor=(.5,.89),ncol=5,frameon=False,fontsize=9)
    title(fig,'How often the formal uncertainties cover zero',
          'All-band strictly common QSOs   ·   fixed published uncertainties')
    save(fig,output,'07_error_coverage')


def yield_support(science, catalog, output):
    n=science['research_fits']
    r=catalog['physical_r']
    fig,axs=plt.subplots(2,2,figsize=(13.5,8))
    fig.subplots_adjust(top=.83,bottom=.12,left=.15,right=.96,hspace=.55,wspace=.48)
    ax=axs[0,0]
    ax.barh([0,1],np.array([r,n])/1e6,color=[COLORS[2],COLORS[3]],height=.58)
    ax.set(yticks=[0,1],yticklabels=['Physical r only','All accepted bands'],xlabel='Research fits [million]',
           title=f'{n-r:,} more fits   ·   +{100*(n/r-1):.1f}%',xlim=(0,14.4))
    for k,v in enumerate([r,n]):
        ax.text(v/1e6+.17,k,f'{v:,}',va='center',fontsize=10)
    for ax,key,labs in [(axs[1,0],'source_classes',['Existing r','Additional r anchor','No NSC r anchor']),
                        (axs[1,1],'observing_support_counts',['Well sampled r anchor','Limited r anchor','No NSC r anchor'])]:
        tags=('r_existing','extra_r_anchor','non_r_anchor') if key=='source_classes' else (
            'r_anchor_well_sampled','r_anchor_limited_sampling','non_r_anchor')
        v=[science[key][tag] for tag in tags]
        ax.barh(range(3),np.array(v)/1e6,color=['#0072B2','#4b9eab','#c87c46'],height=.5)
        for k,size in enumerate(v):
            ax.text(size/1e6+.15,k,f'{size:,}',va='center',fontsize=10)
        ax.set(yticks=range(3),yticklabels=labs,ylim=(2.55,-.55),xlim=(0,11.4),
               xlabel='Full-band research fits [million]',
               title='By source classification' if key=='source_classes' else 'By observing support')
    ax=axs[0,1]
    edges=CONFIG['magnitude_edges']
    for name in ('NSC_r','SDSS_r'):
        bins=catalog['magnitude_histogram'].get(name)
        if bins is not None:
            ax.stairs(np.array(bins)/1e3,edges,color=COLORS[3 if name=='NSC_r' else 1],
                      label=name.replace('_',' '),lw=1.6)
    ax.set(xlabel='Saved r magnitude [mag]',ylabel='Fits per 0.25 mag [thousand]',title='Photometric origin of r')
    ax.legend(loc='upper left',frameon=False,fontsize=9)
    for ax in axs.flat:
        grid(ax,'x' if ax!=axs[0,1] else 'y')
    title(fig,'Where the additional research fits come from',
          'Catalogue counts, member classes and observing support use different selection definitions')
    save(fig,output,'08_yield_and_support')
    support={}
    for k,v in [('group','one_group_supplies_at_least_80pct_leverage'),('season','fewer_than_three_nsc_seasons')]:
        numer=np.zeros(hp.nside2npix(128));denom=numer.copy()
        for core in science['cores'].values():
            flags=core['quality_flag_counts']
            if not core['research_fits']:
                continue
            fraction=flags.get(v,0)/core['research_fits']
            for pixel,num in core['nside128_counts'].items():
                numer[int(pixel)]+=fraction*num
                denom[int(pixel)]+=num
        support[k]={i:100*numer[i]/denom[i] for i in np.flatnonzero(denom)}
    fig,axs=plt.subplots(1,2,figsize=(13,5.2),subplot_kw={'projection':'aitoff'})
    fig.subplots_adjust(top=.82,bottom=.21,left=.04,right=.96,wspace=.16)
    for (k,val),ax in zip(support.items(),axs):
        im=sky_elements(ax,128,val,'magma',Normalize(0,100))
        ax.set_title('One group carries ≥80% of slope weight' if k=='group' else 'Fewer than three NSC seasons',pad=12)
    fig.colorbar(im,cax=fig.add_axes([.30,.09,.40,.028]),orientation='horizontal',label='Fits flagged within core [%]')
    title(fig,'Observing support across the footprint',
          'Core-level fractions displayed on occupied pixels, weighted by the number of fits in each core and pixel')
    save(fig,output,'08_yield_and_support_spatial')


def history(summaries, output):
    info=read(summaries/'v15_C_information_summary.json')['trials']
    fig,axs=plt.subplots(1,2,figsize=(12,5.8))
    fig.subplots_adjust(top=.79,bottom=.19,left=.105,right=.96,wspace=.28)
    route=('A_r','B1_one_per_new_night','B_new_nights','C_all')
    steps=['r only','One per\nnew night','All in\nnew nights','Add same-\nnight bands']
    for sample,lab,ax in zip(('repeat_rich','repeat_sparse'),('Repeat-rich C','Ordinary C control'),axs):
        for band,color,mark in zip(BANDS,COLORS[-2:],('o','s')):
            d=info[sample]['time']['cohorts'][f'r{band}__all']['fixed_qso']
            ax.plot(range(4),[d['methods'][k]['radial_abs68'] for k in route],marker=mark,color=color,
                    label=f'{BANDS[band]}  (N={d["n"]})')
        ax.set(title=lab,xticks=range(4),xticklabels=steps,ylabel=f'2D P68 [{UNIT}]')
        ax.tick_params(axis='x',labelsize=9)
        ax.legend(frameon=False,fontsize=9)
        grid(ax)
    title(fig,'Where the extra measurements add information',
          'v1.5 C controls   ·   fixed QSO identities and nested saved detection sets')
    save(fig,output,'09_historical_information_tests')
    old=read(summaries/'v16_held_year_summary.json')['cohorts']
    fig,axs=plt.subplots(1,2,figsize=(11,5.7))
    fig.subplots_adjust(top=.79,bottom=.20,left=.12,right=.95,wspace=.38)
    r=old['repeat_rich']['year']['by_class']
    yy=np.arange(2)
    match=[r[f'non_r_anchor__r{k}'] for k in BANDS]
    for y,val in zip(yy,match):
        m=val['matched_sources']
        low,high=m['wilson95']
        axs[0].errorbar(m['value']*100,y,xerr=[[100*(m['value']-low)],[100*(high-m['value'])]],
                       marker='o',color=COLORS[-1],capsize=3)
        k=val['shifted_position_matches'][0]
        axs[0].hlines(y+.16,k['wilson95'][0]*100,k['wilson95'][1]*100,color='#9a6b52',lw=2)
        axs[0].scatter(k['value']*100,y+.16,color='#9a6b52',marker='s',clip_on=False)
        axs[0].text(m['value']*100,y-.15,f'{m["n"]}/{m["denominator"]}',ha='center',fontsize=9)
    axs[0].set(xlim=(-3,103),ylim=(1.45,-.45),yticks=yy,yticklabels=list(BANDS.values()),xlabel='Unique held-out position [%]')
    axs[1].barh(yy,np.array([v['radial_p68_mas'] for v in match]),color=COLORS[-1],height=.45)
    axs[1].set(ylim=(1.45,-.45),yticks=yy,yticklabels=list(BANDS.values()),xlabel='Held-out radial position P68 [mas]')
    axs[1].set_xlim(0,740)
    for y,val in zip(yy,match):
        axs[1].text(val['radial_p68_mas']+10,y,f'{val["radial_p68_mas"]:.0f} mas',va='center',fontsize=9)
    for ax in axs:
        grid(ax,'x')
    fig.legend(handles=[Line2D([0],[0],c=COLORS[-1],marker='o',label='Real positions · Wilson 95%'),
                        Line2D([0],[0],c='#9a6b52',marker='s',label='First shifted negative control · Wilson 95%')],
               loc='lower center',bbox_to_anchor=(.5,.02),ncol=2,frameon=False,fontsize=9)
    title(fig,'Checking positions in an unseen year',
          'v1.6 repeated C, non-r training anchors   ·   all bands from 2016 excluded before association')
    save(fig,output,'09_historical_information_tests_held_year')


def example_figures(source, output):
    band_colors={'u':'#8885bd','g':'#009E73','r':'#D55E00','i':'#0072B2',
                 'z':'#aa527e','Y':'#b38330','VR':'#4e8490'}
    camera_marks={'SDSS':'s','c4d':'o','k4m':'^','ksb':'d','unknown':'P'}
    camera_names={'SDSS':'SDSS','c4d':'DECam','k4m':'Mosaic3','ksb':'90Prime','unknown':'NSC'}
    names={'single_run':'One SDSS run, several NSC seasons','multiple_runs':'Repeated SDSS, long time baseline',
           'extra_band':'A qualified solution from additional bands','downweighted':'A measurement with low robust weight',
           'high_leverage':'A high-leverage or short-NSC example','spectral_qso':'A spectroscopic stationary control'}
    for i,record in enumerate(source,1):
        fig, axs=plt.subplots(2,2,figsize=(12,6.9),sharex=True,
                              gridspec_kw={'height_ratios':[2.4,1]})
        fig.subplots_adjust(left=.09,right=.97,bottom=.13,top=.784,hspace=.12,wspace=.26)
        t=np.asarray(record['time_years'])
        x=np.asarray(record['shift_mas'])
        sigma=np.asarray(record['w0_sigma_mas'])
        weight=np.asarray(record['weights'])
        epoch=record['reference_age']
        line_t=np.linspace(min(t)-.3,max(t)+.3,200)
        state=np.asarray(record['display_slope'])
        cov=np.asarray(record['display_covariance'])
        camera=[s if s=='SDSS' else (c if c in camera_marks else 'unknown') for s,c in zip(record['facility'],record['camera'])]
        colors=[band_colors.get(b,'#616c74') for b in record['passband']]
        for axis in range(2):
            ax,res=axs[:,axis]
            dt=line_t-epoch
            design=np.stack([np.ones_like(dt),dt],axis=-1)
            block=cov[2*axis:2*axis+2,2*axis:2*axis+2]
            pred=dt*state[axis]
            uncertainty=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',design,block,design)))
            ax.fill_between(line_t,pred-uncertainty,pred+uncertainty,color='#7fb8d1',alpha=.24)
            ax.plot(line_t,pred,color=COLORS[-1],lw=1.45)
            res.axhline(0,color='#607587',ls=':',lw=1)
            residual=x[:,axis]-(t-epoch)*state[axis]
            old=record['legacy_r']
            if old is not None:
                ax.plot(line_t,np.asarray(old['offset'])[axis]+dt*np.asarray(old['slope'])[axis],
                        color=COLORS[-2],lw=1.2,ls='--')
            before=record['before_position_calibration_mas']
            if before is not None:
                ax.scatter(t,np.asarray(before)[:,axis],marker='+',s=28,c='#b8aeb0',alpha=.7,zorder=2)
            for j in np.argsort(weight):
                c=colors[j]
                mark=camera_marks[camera[j]]
                alpha=.40+.58*weight[j]
                fill=c if weight[j]>=.5 else 'white'
                for panel, values in ((ax,x[:,axis]),(res,residual)):
                    panel.errorbar(t[j],values[j],yerr=sigma[j,axis],color=c,marker=mark,ls='none',
                        markerfacecolor=fill,markeredgecolor=c,alpha=alpha,ms=4.5,lw=.8,zorder=4)
            coord='x' if axis==0 else 'y'
            ax.set(ylabel=f'C0 gnomonic Δ{coord} [mas]',
                   title=f'{AXES[axis]} (ICRS) = {record["published_icrs_motion"][axis]:+.2f} ± {record["conditional_pm_error"][axis]:.2f} {UNIT}')
            res.set(xlabel='Observation Julian year (TT)',ylabel=f'{coord} residual [mas]')
            res.set_xlim(min(t)-.5,max(t)+.5)
            grid(ax)
            grid(res)
        labels=[Line2D([0],[0],color=c,lw=2,label=b) for b,c in band_colors.items() if b in record['passband']]
        labels+= [Line2D([0],[0],color='#344b5c',marker=camera_marks[c],ls='none',label=camera_names[c]) for c in sorted(set(camera))]
        labels.append(Line2D([0],[0],c=COLORS[-1],label='Saved all-band fit'))
        if record['legacy_r'] is not None:
            labels.append(Line2D([0],[0],c=COLORS[-2],ls='--',label='Legacy-quality r fit'))
        if record['before_position_calibration_mas'] is not None:
            labels.append(Line2D([0],[0],c='#b8aeb0',marker='+',ls='none',label='Before C0 correction'))
        if np.any(weight<.5):
            labels.append(Line2D([0],[0],c='#607587',marker='o',mfc='white',ls='none',label='Robust factor <0.5'))
        fig.legend(handles=labels,loc='upper center',bbox_to_anchor=(.5,.876),ncol=min(7,len(labels)),frameon=False,fontsize=8.2)
        s=record['sampling']
        title(fig,names[record['example_type']],f'{record["alias"]}   ·   r = {record["magnitude"]:.2f} ({record["magnitude_source"].replace("_"," ")})   ·   '
              f'{s["sdss_physical_runs"]} SDSS runs   ·   {s["nsc_nights"]} NSC nights / {s["nsc_seasons"]} seasons   ·   {s["joint_span_years"]:.1f} yr joint span')
        save(fig,output,f'05_astrometric_time_series_{i:02d}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',type=Path,help='Combined, deidentified meeting plotting summary')
    p.add_argument('--qso',type=Path)
    p.add_argument('--catalogue',type=Path)
    p.add_argument('--private-examples',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--summaries',type=Path,default=Path('results/summary'))
    p.add_argument('--set',choices=['all','historical','catalogue','qso'],default='all')
    a=p.parse_args()
    if a.private_examples:
        example_figures(read(a.private_examples),a.output)
        return
    science=read(a.summaries/'v17_2154_science_summary.json')
    combined=read(a.summary) if a.summary else {}
    qso=read(a.qso) if a.qso else combined.get('qso')
    catalogue=read(a.catalogue) if a.catalogue else combined.get('catalogue')
    if a.set in ('all','historical'):
        history(a.summaries,a.output)
    if qso and a.set in ('all','qso'):
        assert qso['snapshot']==SNAPSHOT
        footprint(science,qso['source_counts'],a.output)
        for kind in ('gain','count','median','zero_change'):
            spatial_figure(qso,a.output,kind)
        spatial_figure(qso,a.output,'gain',64,local=(137,215,-10,32))
        precision_page(qso,a.output)
        precision_page(qso,a.output,pointlike=True)
        for method in (0,1,2):
            conditions(science,qso,a.output,method)
        conditions(science,qso,a.output,1,instrument=True)
        error_coverage(qso,a.output)
    if catalogue and a.set in ('all','catalogue'):
        assert catalogue['snapshot']==SNAPSHOT
        for label,source in catalogue['comparisons'].items():
            if source['n']:
                for scope in source['samples']:
                    comparison_plot(source,label,a.output,scope,'main')
                comparison_plot(source,label,a.output,'all','tail')
        yield_support(science,catalogue,a.output)


if __name__=='__main__':
    main()
