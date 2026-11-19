"""Public aggregate-only figures for the actual all-band research phase."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(summary,output):
    data=json.loads(summary.read_text())['trials']['allband_qso']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    colors={'b_16_00':'#3674a6','north_one_00_00':'#cf6545'}
    for name,region in data.items():
        x=[];y=[];err=[];counts=[]
        for low in (21,22):
            cohort=region['cohorts'][f'r{low}_{low+1}__all']['joint_r_quality_r_all_nsc_w0']
            before=cohort['methods']['quality_r']['radial_abs68'];after=cohort['methods']['all_nsc_w0']['radial_abs68']
            axes[0].plot([low+.5-.07,low+.5+.07],[before,after],color=colors[name],marker='o',lw=1.5,
                         label=name.replace('_00','') if low==21 else None)
            axes[0].text(low+.5+.08,after,f"n={cohort['n']}",fontsize=8,color=colors[name])
            paired=cohort['paired']['all_nsc_w0_vs_quality_r'];value=100*paired['fraction_improved'][2]
            bounds=100*np.asarray(paired['fraction95'][2]);x.append(low+.5+(-.05 if name.startswith('b') else .05));y.append(value);err.append([value-bounds[0],bounds[1]-value])
        axes[1].errorbar(x,y,yerr=np.asarray(err).T,fmt='o',capsize=3,color=colors[name],label=name.replace('_00',''))
    axes[0].set(xlabel='Spectroscopic reference r magnitude',ylabel='2D absolute proper motion P68 [mas/yr]',xticks=[21.5,22.5],xticklabels=['21–22','22–23'])
    axes[0].set_title('Fixed objects: quality r → all candidate NSC bands',fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0,color='.4',lw=.8);axes[1].set(xlabel='Spectroscopic reference r magnitude',ylabel='Relative P68 gain [%]; positive is smaller',xticks=[21.5,22.5],xticklabels=['21–22','22–23'])
    axes[1].set_title('Paired object bootstrap 95%; not spatial certification',fontsize=10)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('v1.4 preliminary real-data trial — W0, no adopted multiband correction',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


def reference_figure(summary,overlap,output):
    data=json.loads(summary.read_text())['bootstrap_modes'];strips=json.loads(overlap.read_text())['strips']
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.3),constrained_layout=True)
    for axis,color,label,shift in [(0,'#3674a6','East',-.06),(1,'#cf6545','North',.06)]:
        quantiles=np.stack([data['marginal_se_p16_p50_p84_mas'],data['relative_to_cluster_translation_se_p16_p50_p84_mas']])[:,:,axis]
        axes[0].errorbar(np.arange(2)+shift,quantiles[:,1],yerr=np.stack([quantiles[:,1]-quantiles[:,0],quantiles[:,2]-quantiles[:,1]]),fmt='o',capsize=4,label=label,color=color)
        old=[];new=[]
        for name in sorted(strips):
            old.append(strips[name]['original_independent_graphs']['identical_detection_subset_motion_difference_masyr']['median'][axis])
            new.append(strips[name]['joint_graph_fixed_members']['identical_detection_subset_motion_difference_masyr']['median'][axis])
        axes[1].plot(np.arange(28.5,34),old,'o-',color=color,label='Original graphs: '+label)
    axes[0].set_yscale('log');axes[0].set(xticks=[0,1],xticklabels=['Marginal group SE','Relative to cluster translation'],ylabel='Reference position bootstrap SE [mas]')
    axes[0].set_title('A: 212 rejected groups; median and group 16–84%',fontsize=10);axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(28.5,34),new,'ks--',ms=4,label='Common graph, same detections')
    axes[1].set(xlabel='Overlap strip declination [deg]',ylabel='Median ICRS motion difference [mas/yr]')
    axes[1].set_title('1332 identical-member pairs: consistency only',fontsize=10);axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Measured reference-frame diagnostics — no absolute-zero certification',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--reference-overlap',type=Path)
    a=p.parse_args()
    if a.reference_overlap:reference_figure(a.summary,a.reference_overlap,a.output)
    else:run(a.summary,a.output)


if __name__=='__main__':main()
