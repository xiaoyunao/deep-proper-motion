"""Plot frozen blind QSO quality from its public aggregate."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(data, output):
    cohorts = data['cohorts']
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelsize':11,'font.size':10})
    fig, (precision, zero) = plt.subplots(1,2,figsize=(11.2,4.65),
                                         gridspec_kw={'width_ratios':[1,1.35]},layout='constrained')
    methods = ('blind_all','Xiao','NSC')
    labels = ('Blind all-band','Xiao 2025','NSC DR2')
    colors = ('#254a6d','#bd6846','#799683')
    for i, (name,label,color) in enumerate(zip(methods,labels,colors)):
        matched = [cohorts[mag]['all']['common_published'] for mag in ('21','22')]
        bars = precision.bar(np.arange(2)+(i-1)*.23,
                             [x['methods'][name]['radial_abs68'] for x in matched],
                             width=.21,color=color,label=label)
        precision.bar_label(bars,fmt='%.1f',padding=2,fontsize=8)
    precision.set_xticks([0,1],[f"${lo}\\leq r<{lo+1}$\nN = {x['n']:,}" for lo,x in zip((21,22),matched)])
    precision.set_ylabel(r'2D absolute proper-motion P68 (mas yr$^{-1}$)')
    precision.set_title('Strict common optical spectra')
    precision.set_ylim(0, precision.get_ylim()[1]*1.14)
    precision.legend(frameon=False,loc='upper left',fontsize=8)

    points = {}
    for key, sample in cohorts['21'].items():
        if not key.startswith('window_') or sample['all_blind']['n']<30:
            continue
        name = key[len('window_'):]
        points[name] = sample['all_blind']['methods']['blind_all']['median'][1]
    names = sorted(points,key=points.get)
    for mag, marker, color in (('21','o',colors[0]),('22','D',colors[1])):
        yy, xx = [], []
        for index,name in enumerate(names):
            group = cohorts[mag].get('window_'+name,{}).get('all_blind',{})
            if group.get('n',0)>=30:
                yy.append(index)
                xx.append(group['methods']['blind_all']['median'][1])
        zero.scatter(xx,yy,marker=marker,s=23,c=color,label=f'{mag}–{int(mag)+1}')
    zero.axvspan(-.5,.5,color='#e8ecf0',zorder=0)
    zero.axvline(0,color='#526474',lw=.7)
    zero.set_yticks(range(len(names)),names,fontsize=7)
    zero.invert_yaxis()
    zero.set_xlabel(r'Northward QSO median (mas yr$^{-1}$)')
    zero.set_title('Blind output by window, N ≥ 30 per bin')
    zero.legend(title='Spectral r',frameon=False,fontsize=8,title_fontsize=8)
    fig.savefig(output,dpi=250,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    render(json.loads(args.summary.read_text()),args.output)
