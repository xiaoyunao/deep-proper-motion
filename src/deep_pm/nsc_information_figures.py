"""v1.5 scientific figures generated exclusively from reviewed public aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(root,output):
    info=json.loads((root/'v15_information_exposure_summary.json').read_text())['trials']['qso']
    controls=json.loads((root/'v15_dcr_error_pilot_summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8.8),constrained_layout=True)
    tints=['#3674a6','#cf6545'];labels=['21–22','22–23'];names=['A_r','B1_one_per_new_night','B_new_nights','C_all']
    for low,tint,label in zip((21,22),tints,labels):
        sample=info['b_16_00']['cohorts'][f'r{low}__all'];metrics=sample['fixed_qso'];n=metrics['n']
        axes[0,0].plot(range(4),[metrics['methods'][k]['radial_abs68'] for k in names],'o-',color=tint,label=f'r={label}, n={n}')
        axes[0,1].plot(range(4),[sample['information'][k]['nsc_nights_p16_p50_p84'][1] for k in names],'o-',color=tint,label=f'r={label}')
    axes[0,0].set(ylabel='QSO 2D absolute PM P68 [mas/yr]',title='B: same QSO and SDSS r measurements')
    axes[0,1].set(ylabel='Median actual NSC nights',title='More observed nights need not lower absolute PM error')
    for ax in axes[0]:ax.set_xticks(range(4));ax.set_xticklabels(['r only','+ one/new night','+ all/new night','+ all bands']);ax.tick_params(axis='x',labelsize=8)
    for j,(core,short) in enumerate([('b_16_00','B'),('north_one_00_00','N1')]):
        control=controls['dcr_qso_'+core+'_summary']['cohorts']
        for low,tint in zip((21,22),tints):
            part=control['r'+str(low)]['fixed'];keys=('D0_local_C0_all','D1_geometric_DCR_all')
            if not set(keys)<=part.get('methods',{}).keys():
                keys=tuple(k for k in part.get('methods',{}) if k.startswith(('D0_','D1_')) and k.endswith('_all'))
            if len(keys)!=2:continue
            x=j+(-.1 if low==21 else .1);gain=100*(1-part['methods'][keys[1]]['radial_abs68']/part['methods'][keys[0]]['radial_abs68'])
            bounds=100*np.asarray(part['paired'][keys[1]+'_vs_'+keys[0]]['fraction95'][2]);axes[1,0].vlines(x,*bounds,color=tint)
            axes[1,0].plot(x,gain,'o',color=tint);axes[1,0].annotate(f"n={part['n']}",(x,gain),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1,0].axhline(0,color='.4',lw=.8);axes[1,0].set(xticks=[0,1],xticklabels=['B','N1'],ylabel='DCR relative QSO P68 gain [%]',title='One physical DCR coefficient: no uniform adoption gain')
    pilot=controls['pilot_summary']['trials']
    groups=[('b_16_00','r_existing__r21','B existing'),('b_16_00','extra_r_anchor__r21','B extra r'),('north_one_00_00','r_existing__r21','N1 existing'),('north_one_00_00','extra_r_anchor__r21','N1 extra r')]
    for j,(core,cohort,label) in enumerate(groups):
        result=pilot['errors/'+core]['cohorts'][cohort]['E_within_band_vs_W0'];value=100*result['relative_source_p68_gain'];bounds=100*np.array(result['object_bootstrap_gain95'])
        axes[1,1].plot(j,value,'o',color=tints[0]);axes[1,1].vlines(j,*bounds,color=tints[0]);axes[1,1].annotate(f"n={result['source_count']}",(j,value),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1,1].axhline(0,color='.4',lw=.8);axes[1,1].set(xticks=range(4),xticklabels=[r[2] for r in groups],ylabel='Held-year source P68 gain [%]',title='Galaxy-learned night errors: independent stellar test')
    axes[1,0].set_xlim(-.35,1.4);axes[1,1].set_xlim(-.3,3.55)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    fig.suptitle('v1.5 real-data diagnostics: time information, DCR and error calibration\nFixed objects; bottom panels: object 95% intervals, not spatial certification',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary_directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.summary_directory,a.output)


if __name__=='__main__':main()
