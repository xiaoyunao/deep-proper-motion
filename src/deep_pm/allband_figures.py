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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(**vars(a))


if __name__=='__main__':main()
