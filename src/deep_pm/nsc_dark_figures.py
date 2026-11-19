"""Actual all-object held-year and spectral-zero evidence, aggregate inputs only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def run(summary,output):
    trials=json.loads(summary.read_text())['trials'];fig,axes=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
    regions=[('b_16_00','B'),('north_one_00_00','N1'),('repeat_sparse','Ordinary C')];tints=['#3674a6','#cf6545']
    for panel,low in enumerate((21,22)):
        for j,(core,label) in enumerate(regions):
            for kind,tint,delta in [('extra_r_anchor',tints[0],-.08),('non_r_anchor',tints[1],.08)]:
                row=trials['year/'+core]['by_class'][kind+'__r'+str(low)];ratio=row['matched_sources'];value=ratio['value'];bounds=ratio['wilson95'];x=j+delta
                axes[panel].plot(x,value,'o',color=tint,label=kind.replace('_',' ') if j==0 else None);axes[panel].vlines(x,*bounds,color=tint)
                axes[panel].annotate(f"{ratio['n']}/{ratio['denominator']}",(x,value),xytext=(2,-13 if delta<0 else 6),textcoords='offset points',fontsize=7,color=tint)
        axes[panel].set(title=f'Whole-year prediction: r={low}–{low+1}',xticks=range(3),xticklabels=[r[1] for r in regions],ylabel='Unique held-position recovery',ylim=(.75,1.055),xlim=(-.4,2.55));axes[panel].grid(axis='y',alpha=.2)
    selected=[('b_16_00','extra_r_anchor__r22','B extra r, 22–23'),('north_one_00_00','extra_r_anchor__r21','N1 extra r, 21–22'),('repeat_sparse','extra_r_anchor__r21','C extra r, 21–22')]
    for j,(core,key,label) in enumerate(selected):
        part=trials['members/'+core]['by_class'][key]['all_spectral_zero'];model=part['methods']['multi']
        for axis,tint,delta in [(0,tints[0],-.08),(1,tints[1],.08)]:
            x=j+delta;axes[2].plot(x,model['median'][axis],'o',color=tint,label=('East' if axis==0 else 'North') if j==0 else None);axes[2].vlines(x,*model['median_bias95'][axis],color=tint)
        axes[2].annotate('n='+str(part['n']),(j,max(model['median'])),xytext=(4,8),textcoords='offset points',fontsize=8)
    axes[2].axhspan(-.5,.5,color='.88',label='±0.5 target');axes[2].axhline(0,color='.4',lw=.7);axes[2].set(xticks=range(3),xticklabels=['B: 22–23','N1: 21–22','C: 21–22'],ylabel='Spectral QSO median PM [mas/yr]',title='High recurrence does not certify PM zero',xlim=(-.4,2.55));axes[2].grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='lower left');axes[2].legend(fontsize=8,loc='lower left')
    fig.suptitle('v1.5 complete object-index inputs: new-source evidence\nTraining-year classes; Wilson 95% recovery and object 95% zero intervals; unknown stellar identities unverified',fontsize=11)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.summary,a.output)


if __name__=='__main__':main()
