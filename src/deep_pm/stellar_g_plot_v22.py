"""Public aggregate figures; position, relative coefficient and PM kept distinct."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .blind_catalogue_quality import read


def run(repo):
    summary=repo/'results/summary';coefficient=read(summary/'v22_stellar_g_coefficient_summary.json');conditions=read(summary/'v22_stellar_g_conditions_summary.json');east=read(summary/'v22_ngc102_east_attribution_summary.json')
    plt.rcParams.update({'font.family':'serif','font.size':11,'axes.spines.top':True,'axes.spines.right':True,'xtick.direction':'in','ytick.direction':'in','savefig.dpi':180})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    labels=['Raw','Frozen -23.39','Frozen -28.32','One stellar\ncandidate'];tags=['uncorrected','frozen_minus23p39','frozen_minus28p32','one_stellar_candidate']
    validation=coefficient['validation']['independent_identity_and_night']['models']
    raw=[validation[k]['raw_C0_difference_native_mas']['absolute_radial_p68'] for k in tags]
    relative=[validation[k]['relative_after_nuisance_intercept_native_mas']['absolute_radial_p68'] for k in tags]
    x=np.arange(4);ax=axes[0,0];ax.bar(x-.18,raw,.36,label='Raw C0 difference',color='#c44e52');ax.bar(x+.18,relative,.36,label='Pair-intercept diagnostic',color='#295b92');ax.set_xticks(x,labels);ax.set_ylabel('Repeat-position radial P68 (mas)');ax.legend(fontsize=9);ax.set_title('Held stars AND nights: 6,303 pairs / 1,309 stars / 17 nights')
    ax=axes[0,1];beta=coefficient['coefficient_mas_per_mag_tanz'];lo,hi=coefficient['crossed_star_night_and_shared_colour_sensitivity95']
    ax.axvspan(lo,hi,color='#295b92',alpha=.2,label='Crossed star/night + colour sensitivity 95%');ax.axvline(beta,color='#173e66',lw=2,label=f'Observed repeats: {beta:.2f}');ax.axvline(-28.32,color='#c44e52',ls='--',label='Frozen galaxy coefficient -28.32');ax.axvline(-23.39,color='#df9b31',ls=':',label='Frozen galaxy coefficient -23.39');ax.set_ylim(0,1);ax.set_yticks([]);ax.set_xlabel(r'$k$ [mas / (mag tan $z$)]');ax.set_title('Relative colour slope; absolute zero unconstrained');ax.legend(fontsize=8,loc='upper left')
    ax=axes[1,0];groups=[('all','All'),('same_SDSS_fields','Same SDSS\nfields'),('different_SDSS_fields','Different\nSDSS fields'),('NSC_exposure_jaccard_ge0p8','Exposure\nJaccard >=0.8'),('NSC_exposure_jaccard_lt0p8','Exposure\nJaccard <0.8')];center=[];bounds=[];n=[]
    for key,label in groups:
        value=east['strata'][key]['residual'];center.append(value['median'][0]);bounds.append(value.get('pair_resampling_sensitivity95',value.get('median_object_interval95'))[0]);n.append(value['n'])
    center=np.array(center);bounds=np.array(bounds);ax.errorbar(range(5),center,yerr=[center-bounds[:,0],bounds[:,1]-center],fmt='o',color='#173e66',capsize=3);ax.axhline(0,color='0.4',lw=1);ax.set_xticks(range(5),[label for key,label in groups]);ax.set_ylabel(r'Target - field $\Delta\mu_{\alpha *}$ (mas yr$^{-1}$)');ax.set_title('Frozen 102 bright pairs: conditions remain confounded')
    for i,(v,count) in enumerate(zip(center,n)):ax.annotate(f'n={count}',(i,v),xytext=(5,7),textcoords='offset points',fontsize=8)
    ax=axes[1,1];matched=conditions['matched_actual_g_colour'];models=matched['frozen_models'];v=[matched['observed_common_basis_red_minus_blue_mas']['median'][1],models['-28.32']['median'][1],models['-23.39']['median'][1]];ax.bar(range(3),v,color=['#c44e52','#295b92','#df9b31']);ax.axhline(0,color='0.4',lw=1);ax.set_xticks(range(3),['Raw','Frozen -28.32','Frozen -23.39']);ax.set_ylabel('Red - blue North position difference (mas)');ax.set_title('388 actual-g / SNR / neighbour matched pairs')
    for i,a in enumerate(axes.flat):a.text(-.08,1.04,'abcd'[i]+')',transform=a.transAxes,fontweight='bold');a.tick_params(which='both',top=True,right=True)
    folder=repo/'figures';folder.mkdir(exist_ok=True);fig.savefig(folder/'v22_stellar_g_mechanism_summary.png');fig.savefig(folder/'v22_stellar_g_mechanism_summary.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repo',type=Path);a=p.parse_args();run(a.repo)
