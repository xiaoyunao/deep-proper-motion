"""Four source-backed figures and non-identifying v2.3 scientific readout."""
import argparse,os
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
from .blind_catalogue_quality import read
from .early_pixel_model_v23 import sky_at
from .io import digest,save_json
from .public_summary import check_aggregate

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':11,'figure.facecolor':'white','savefig.facecolor':'white'})
COLORS=['#294f78','#d87542','#8b969f']


def save(fig,root,name):
    fig.savefig(root/'figures'/name,dpi=190,bbox_inches='tight');plt.close(fig)


def run(base,root):
    (root/'figures').mkdir(exist_ok=True);pre=read(root/'preflight/v23_preflight_summary.json')
    if digest(base/'v2.2/registration/private_full_coordinate_plan.json')!=pre['plan_sha256'] or digest(base/'v2.2/v22_final_stage_binding_summary.json')!=pre['final_binding_sha256']:raise ValueError('Frozen full science binding changed')
    hst=read(root/'hst/v23_HST_frozen_overlap_summary.json');opp=read(root/'hst/v23_HST_opportunity_summary.json');ties=read(root/'pixelprep/v23_frame_registration_summary.json');pixel=read(root/'pixels/v23_pixel_validation_summary.json');bud=read(root/'budget/v23_frozen_error_budget_summary.json');late=read(root/'pixels/private_frozen_forty_late.json.gz');background=read(root/'pixels/private_background_registration.json');results=[read(f)['result'] for f in sorted((root/'pixels/validation').glob('private_*.json'))];records=[r for c in bud['cores'] for r in read(root/'budget'/c['core']/'private_covariance_budget.json.gz')]
    summary=dict(baseline_science=pre['baseline_science'],baseline_plan_sha256=pre['plan_sha256'],baseline_final_binding_sha256=pre['final_binding_sha256'],baseline_solutions_unchanged=pre['all_v22_solutions_frozen'],HST_primary_matches=sum(f['quality_unique_v22'] for f in hst['fields']),HST_position_matches=sum(f['all_position_unique_v22'] for f in hst['fields']),HST_opportunity=opp['fields'],reference_fields_passed=ties['passed_physical_fields'],reference_fields_tested=ties['registered_physical_fields'],budget_sources=len(records),budget_cores=len(bud['cores']),maximum_saved_covariance_error=max(c['maximum_covariance_reconstruction_error'] for c in bud['cores']),variance_by_ground_r={},pixel_by_depth={},run_holdout={},slurm_job=os.environ['SLURM_JOB_ID'])
    # Exact source-level pooling on NERSC: never average per-core medians.
    for lo in [20,21,22]:
        part=[r for r in records if lo<=r['r']<lo+1];item=dict(n=len(part))
        if part:
            fractions=np.array([[np.diag(r['independent_terms']['SDSS'])/np.diag(r['total']),np.diag(r['independent_terms']['NSC'])/np.diag(r['total']),np.diag(r['reference'])/np.diag(r['total'])] for r in part]);item.update(mean_axis_variance_fractions=fractions.mean(axis=0),fraction_p16_p50_p84=np.quantile(fractions,[.16,.5,.84],axis=0))
        summary['variance_by_ground_r'][str(lo)]=item
    for lo in [22,22.5]:
        part=[r for r in results if r['experiment_mode']=='control' and r['category'].startswith('injection') and lo<=r['r0']<lo+.5];summary['pixel_by_depth'][str(lo)]=dict(injected=len(part),computed=sum(r['status']=='computed' for r in part),information_pass=sum(r.get('information_gate_pass',False) for r in part),adoption_pass=sum(r.get('adoption_gate_pass',False) for r in part))
    real=[r for r in results if r['experiment_mode']=='real'];holds=[h for r in real for h in r.get('held_run_predictions',[])];supported=[h for h in holds if 'conditional_position_residual_mas' in h];summary['run_holdout']=dict(whole_runs=len(holds),independent_SNR5_positions=len(supported),conditional_radial_P68_mas=float(np.quantile([np.linalg.norm(h['conditional_position_residual_mas']) for h in supported],.68)) if supported else None,ellipse68_coverage=float(np.mean([h['conditional_mahalanobis2']<=2.278868566 for h in supported])) if supported else None,ellipse95_coverage=float(np.mean([h['conditional_mahalanobis2']<=5.991464547 for h in supported])) if supported else None)
    summary['real_forty']=dict(initial=len(real),late_unique=sum(r['late_status']=='conditional_late_association' for r in real),computed=sum(r['status']=='computed' for r in real),information_pass=sum(r.get('information_gate_pass',False) for r in real),adoption_pass=sum(r.get('adoption_gate_pass',False) for r in real));summary['415_expansion']=dict(original_parent=415,new_pixels_fitted=0,decision='Not expanded: no independently qualified frame registration. Information-only results and conditional controls do not satisfy adoption, irrespective of selected conditional precision.')
    summary['scope']='Frozen production unchanged. HST primary overlap insufficient: no measured dark-star residual precision. Covariance budget is the saved conditional estimator, not all systematic error. Pixel information gate differs from adoption; injections test same-depth early information on actual backgrounds with conditional late sampling, not new-star correctness or absolute calibration. Profile coverage is conditional; linearized common reference/registration terms are separate. The missing reference tie cannot be cured by object counts.'
    check_aggregate(summary);save_json(root/'v23_key_science_summary.json',summary)
    # 1. Central-field opportunity, clearly separate unlike photometric axes.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');x=np.arange(3);names=[r['field'].upper().replace('NGC','NGC ') for r in hst['fields']]
    for shift,key,label,color in [(-.27,'HST_parent','HST PM parent',COLORS[0]),(0,'combined_quality','Frozen strict quality',COLORS[1]),(.27,'all_position_unique_v22','Unique ground overlap (all quality)',COLORS[2])]:
        values=[r[key] for r in hst['fields']];axes[0].bar(x+shift,np.maximum(values,.5),.25,label=label,color=color)
        for i,v in enumerate(values):axes[0].text(i+shift,max(v,.5)*1.2,str(v),ha='center',fontsize=8)
    axes[0].set(yscale='log',ylim=(.4,3e5),xticks=x,xticklabels=names,ylabel='Actual objects (log scale)',title='Central HST coverage and selection');axes[0].legend(fontsize=8,loc='upper left')
    for i,row in enumerate(opp['fields']):
        values=[row['HST_F606W_parent_bins'].get(str(k),0) for k in [20,21,22]];axes[1].plot([20.5,21.5,22.5],values,'o-',label=names[i],color=COLORS[i])
    axes[1].set(xlabel='Native HST F606W (VEGA mag)',ylabel='HST quality except correction flag',title='HST faint parent is not ground r');axes[1].legend();axes[1].text(.04,.06,'Strict common HST–ground sample: N = 0\nNo dark-star precision or coverage claim',transform=axes[1].transAxes,bbox=dict(facecolor='#f4ede7',edgecolor='none',pad=7));fig.suptitle('Independent HST validation: measured opportunity limits',fontsize=14);save(fig,root,'v23_HST_summary.png')
    # 2. Preserved covariance, east/north fractions with exact source pooling.
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for axis,ax in enumerate(axes[:2]):
        bottom=np.zeros(3)
        for term,name,color in zip(range(3),['SDSS independent','NSC independent','Common reference'],COLORS):
            values=np.array([summary['variance_by_ground_r'][str(k)]['mean_axis_variance_fractions'][term][axis] for k in [20,21,22]]);ax.bar(x,values,bottom=bottom,label=name,color=color);bottom+=values
        ax.set(ylim=(0,1.02),xticks=x,xticklabels=['20–21','21–22','22–23'],xlabel='Actual ground r (mag)',ylabel='Mean fraction of conditional PM variance',title=['East','North'][axis]);ax.text(.04,.95,'N = '+', '.join(str(summary['variance_by_ground_r'][str(k)]['n']) for k in [20,21,22]),transform=ax.transAxes,va='top',fontsize=8)
    axes[0].legend(loc='center left',fontsize=8);blocks=[b for c in bud['cores'] for b in c['bright_position_blocks'] if b['mode']=='survey_band' and 'median_native_position_residual_mas' in b];labels=sorted({b['observational_block'] for b in blocks})
    for j,label in enumerate(labels):
        values=np.array([b['median_native_position_residual_mas'] for b in blocks if b['observational_block']==label]);axes[2].scatter(np.full(len(values),j)-.08,values[:,0],s=20,color=COLORS[0],label='East' if j==0 else None);axes[2].scatter(np.full(len(values),j)+.08,values[:,1],s=20,color=COLORS[1],label='North' if j==0 else None)
    axes[2].axhline(0,color='.5',lw=.8);axes[2].set(xticks=range(len(labels)),xticklabels=labels,ylabel='Bright-star median position residual (mas)',title='Actual survey/band × core; ≥5 stars');axes[2].tick_params(axis='x',rotation=70,labelsize=7);axes[2].legend(fontsize=8);fig.suptitle(f"Frozen estimator: {len(records):,} sources, {len(bud['cores'])} cores; shared terms retained",fontsize=13);save(fig,root,'v23_error_budget_summary.png')
    # 3. Both successes and non-adoptions; denominator cannot disappear.
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');ax=axes[0,0];categories=['no_cached_SDSS','SDSS_quality_distance','positive_early_control'];pos=np.arange(3)
    for d,key,label,color in [(-.26,'initial','Frozen parent',COLORS[2]),(0,'information_pass','Early information gate',COLORS[0]),(.26,'adoption_pass','Qualified new PM',COLORS[1])]:ax.bar(pos+d,[pixel['groups'][c][key] for c in categories],.24,label=label,color=color)
    ax.set(xticks=pos,xticklabels=['No cached SDSS','Quality/distance exit','Positive controls'],ylabel='Actual objects',title='Fixed forty: localization ≠ adoption');ax.legend(fontsize=8)
    ax=axes[0,1];depth=summary['pixel_by_depth'];ax.bar([0,1],[depth[str(k)]['information_pass']/max(1,depth[str(k)]['injected']) for k in [22,22.5]],color=COLORS[:2]);ax.set(xticks=[0,1],xticklabels=['22–22.5','22.5–23'],ylim=(0,1),xlabel='Injected SDSS r (mag)',ylabel='Conditional information recovery fraction',title='Moving sources in independent real backgrounds')
    for j,k in enumerate([22,22.5]):ax.text(j,.95,f"{depth[str(k)]['information_pass']}/{depth[str(k)]['injected']}",ha='center',va='top')
    ax=axes[1,0]
    for j,k in enumerate(['injection_isolated','injection_neighbor']):
        stat=pixel['groups'][k]['all_computed']
        if 'mean_PM_bias_masyr' in stat:
            v=np.array(stat['mean_PM_bias_masyr']);ci=np.array(stat['mean_bias_interval95']);ax.errorbar(np.array([0,1])+(j-.5)*.1,v,yerr=np.maximum(0,np.stack([v-ci[:,0],ci[:,1]-v])),fmt='o',capsize=4,label=k.replace('injection_',''),color=COLORS[j])
    ax.axhspan(-.5,.5,color='#e3ebdf');ax.axhline(0,color='.5',lw=.7);ax.set(xticks=[0,1],xticklabels=['East','North'],ylabel='Injected mean PM bias (mas/yr)',title='All computed injections: conditional 95% intervals');ax.legend(fontsize=8)
    ax=axes[1,1];ax.axis('off');b=pixel['groups']['blank'];lines=[f"Empty backgrounds: {b['information_pass']}/{b['initial']} information accepts",f"One-sided 95% upper bound: {100*b['information_false_accept_one_sided95_upper']:.2f}% (target ≤1%)",f"Independent frame ties passed: {ties['passed_physical_fields']}/{ties['registered_physical_fields']}",f"Full-run holdouts: {summary['run_holdout']['whole_runs']}; localized: {summary['run_holdout']['independent_SNR5_positions']}",'415 expansion: not adopted; calibration gate failed','Intervals/Monte Carlo are conditional on real late schedules.','No certified new-star count or absolute PM precision.'];ax.text(.02,.95,'\n\n'.join(lines),va='top',fontsize=10);fig.suptitle('Early SDSS pixel test: depth, bias and empty-position controls',fontsize=14);save(fig,root,'v23_pixel_validation_summary.png')
    # 4. Two metadata/frozen-index selected examples, no successful-case picking.
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');selected=[]
    for category in ['positive_early_control','no_cached_SDSS']:
        candidates=sorted([r for r in real if r['category']==category and r['status']=='computed'],key=lambda r:r['frozen_index']);selected.append(candidates[0] if candidates else None)
    for col,result in enumerate(selected):
        if result is None:
            axes[0,col].text(.1,.5,'No computable fixed example');continue
        row=next(r for r in late if r['frozen_index']==result['frozen_index']);tt=np.asarray(row['late_time']);xy=np.asarray(row['late_xy']);cov=np.asarray(row['late_covariance']);earlytime=result['early_only_epoch'];grid=np.linspace(min(tt.min(),earlytime)-.5,tt.max()+.5,200)
        for axis in [0,1]:
            ax=axes[axis,col];ax.errorbar(tt,xy[:,axis],yerr=np.sqrt(cov[:,axis,axis]),fmt='.',ms=4,alpha=.55,color=COLORS[2],label='NSC positions (once)');ax.plot(grid,sky_at(row['late_only_state'],grid,2015.5)[:,axis],color=COLORS[0],ls='--',label='NSC-only');ax.plot(grid,sky_at(result['joint_state_local'],grid,2015.5)[:,axis],color=COLORS[1],label='Free joint pixel + NSC')
            interval=result.get('early_position_profiles',{}).get(str(2*axis),{}).get('1.0');value=result['early_only_state'][2*axis]
            if interval is not None and all(v is not None for v in interval):ax.errorbar([earlytime],[value],yerr=[[max(0,value-interval[0])],[max(0,interval[1]-value)]],fmt='s',color='#52764d',label='Early-only position profile')
            else:ax.axvline(earlytime,color='#52764d',alpha=.6,lw=1,label='Early epoch; weak localization')
            ax.set(xlabel='Actual observation year (TT)',ylabel=['East offset (mas)','North offset (mas)'][axis]);ax.grid(alpha=.15)
        axes[0,col].set_title(result['category'].replace('_',' ')+f"; r={row['r0']:.2f}\nInformation={result.get('information_gate_pass')}; adoption=False",fontsize=11)
    axes[0,0].legend(fontsize=7);fig.suptitle('Two frozen-index examples: trajectories and independent early information',fontsize=14);save(fig,root,'v23_trajectories_summary.png')
    save_json(root/'v23_delivery_summary.json',dict(public_figures={f.name:digest(f) for f in sorted((root/'figures').glob('v23_*.png'))},key_science_sha256=digest(root/'v23_key_science_summary.json'),four_main_figures=True,baseline_binding_unchanged=True,scope='Only four new figures and aggregate readout. Real per-source, control positions, HST and pixel arrays stay on personal PScratch.'))
    print('V23_CLOSEOUT_COMPLETE',summary['real_forty'],summary['variance_by_ground_r'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.base,a.root)
