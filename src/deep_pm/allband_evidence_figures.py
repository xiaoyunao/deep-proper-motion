"""Scientific figures from reviewed v1.4 public aggregates only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(root,name):
    return json.loads((root/name).read_text())['trials']


def run(root,output):
    paired=load(root,'v14_paired_sdss_summary.json')['paired']
    color=load(root,'v14_color_resume_summary.json')
    support=load(root,'v14_allband_source_support_summary.json')['auxiliary']
    qso=load(root,'v14_joint_color_validation_summary.json')['qso']
    fig,axes=plt.subplots(2,2,figsize=(12,9.3),constrained_layout=True)
    colors=['#3674a6','#cf6545']
    for low,tint in zip((21,22),colors):
        for j,(name,label) in enumerate([('a_09_00','A (support trial)'),('b_16_00','B'),('north_one_00_00','N1')]):
            block=paired[name]['cohorts'][f'r{low}__all_held_bands'];value=100*block['relative_source_p68_gain'];bounds=100*np.array(block['object_bootstrap_gain95'])
            x=j+(-.1 if low==21 else .1)
            axes[0,0].plot(x,value,'o',color=tint,label=f'r={low}–{low+1}' if j==0 else None)
            axes[0,0].vlines(x,*bounds,color=tint)
            axes[0,0].annotate(f"n={block['source_count']}",(x,value),xytext=(4,5 if low==21 else -13),textcoords='offset points',fontsize=8,color=tint)
    axes[0,0].set(xticks=range(3),xticklabels=['A support trial','B','N1'],ylabel='Held-year position P68 gain [%]',title='Same stars, r → all NSC bands')
    for j,(name,label) in enumerate([('a_01_01','Y-one-night field'),('b_21_01','Y-multiyear field'),('b_01_10','VR-multiyear field'),('north_one_00_01','90Prime field')]):
        for low,tint in zip((21,22),colors):
            block=support[name]['cohorts'][f'r{low}_{low+1}__all']['joint_r_quality_r_all_nsc_w0']
            if 'methods' not in block:continue
            contrast=block['paired']['all_nsc_w0_vs_quality_r'];value=100*contrast['fraction_improved'][2];bounds=100*np.array(contrast['fraction95'][2]);x=j+(-.1 if low==21 else .1)
            axes[0,1].plot(x,value,'o',color=tint);axes[0,1].vlines(x,*bounds,color=tint)
            axes[0,1].annotate(f"n={block['n']}",(x,value),xytext=(4,4),textcoords='offset points',fontsize=8,color=tint)
    axes[0,1].set(xticks=range(4),xticklabels=['Y one night','Y multiyear','VR multiyear','90Prime'],ylabel='QSO proper-motion P68 gain [%]',title='Metadata-selected fields; combined bands')
    for j,(name,label) in enumerate([('north_one_00_00','N1'),('north_one_00_01','90Prime'),('north_three_00_00','N3')]):
        for low,tint in zip((21,22),colors):
            block=qso[name]['cohorts'][f'r{low}__all']
            if block['n']<5:continue
            contrast=block['paired']['color_all_vs_C0_all'];value=100*contrast['fraction_improved'][2];bounds=100*np.array(contrast['fraction95'][2]);x=j+(-.1 if low==21 else .1)
            axes[1,0].plot(x,value,'o',color=tint);axes[1,0].vlines(x,*bounds,color=tint)
            axes[1,0].annotate(f"n={block['n']}",(x,value),xytext=(4,5),textcoords='offset points',fontsize=8,color=tint)
    axes[1,0].set(xticks=range(3),xticklabels=['N1','90Prime','N3 (n<5)'],ylabel='QSO proper-motion P68 gain [%]',title='All-band C0 → joint g color term')
    groups=[('year/north_one_00_00','r_existing','N1 r-existing'),('year/north_one_00_00','extra_r_anchor','N1 extra r'),('year/north_three_00_00','extra_r_anchor','N3 extra r'),('year/north_three_00_00','non_r_anchor','N3 non-r')]
    for j,(name,kind,label) in enumerate(groups):
        for low,tint in zip((21,22),colors):
            block=color[name]['cohorts'][f'{kind}__r{low}']
            if block['source_count']<5:continue
            value=100*block['relative_source_p68_gain'];bounds=100*np.array(block['object_bootstrap_gain95']);x=j+(-.1 if low==21 else .1)
            axes[1,1].plot(x,value,'o',color=tint);axes[1,1].vlines(x,*bounds,color=tint)
            axes[1,1].annotate(f"n={block['source_count']}",(x,value),xytext=(4,4),textcoords='offset points',fontsize=8,color=tint)
    axes[1,1].set(xticks=range(4),xticklabels=[g[2] for g in groups],ylabel='Held-year position P68 gain [%]',title='Same held detections: C0 → joint g color term')
    axes[0,0].legend(fontsize=8)
    for ax in axes.flat:
        ax.axhline(0,color='.35',lw=.8);ax.grid(axis='y',alpha=.2);ax.tick_params(labelsize=8)
    fig.suptitle('v1.4 real-data evidence: paired object 95% intervals\nPosition prediction and absolute proper motion are separate tests; no production adoption',fontsize=12)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary_directory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.summary_directory,a.output)


if __name__=='__main__':main()
