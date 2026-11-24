"""Plot the NGC 5466 frozen-control result from its released aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np


def render(data, output):
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10})
    fig, axes=plt.subplots(2,2,figsize=(9.2,8.2),layout='constrained')
    spatial,cmd,bright,faint=axes.ravel()
    density=data['spatial_cmd_counts']
    edges=np.asarray(density['edges_deg'])
    z=np.asarray(density['tile_counts'])
    image=spatial.pcolormesh(edges,edges,np.ma.masked_less(z.T,1),cmap='Blues',norm=LogNorm(vmin=2,vmax=max(3,z.max())))
    fig.colorbar(image,ax=spatial,shrink=.67,label='Objects per cell')
    rules=data['selection']
    for name,color in [('target_annulus_deg','#1d4b6a'),('background_annulus_deg','#b15b43')]:
        for radius in rules[name]:
            spatial.add_patch(Circle((0,0),radius,facecolor='none',edgecolor=color,lw=1.2))
    spatial.set(xlim=(.5,-.5),ylim=(-.5,.5),aspect='equal',xlabel='East offset (deg)',ylabel='North offset (deg)',
                title='CMD-selected photometric source density')
    m=data['optical_cmd']
    excess=np.asarray(m['histograms']['target'])-np.asarray(m['histograms']['background'])
    cmd.pcolormesh(m['color_edges'],m['r_edges'],np.ma.masked_less(excess,1),
                   cmap='Blues',norm=LogNorm(vmin=1,vmax=max(3,excess.max())))
    for row in rules['independently_fitted_cmd']:
        center=row['center_color']
        if center is not None:
            lo,hi=(row['lower_r0'],row['upper_r0'])
            cmd.plot([center-.105,center+.105],[(lo+hi)/2]*2,color='#b15b43',lw=2.2)
    cmd.set(xlim=(-.15,1.25),ylim=(23,18),xlabel=r'$(g-r)_0$',ylabel=r'$r_0$',
            title='Target minus equal-area background')
    reference=data.get('independent_bright_cluster_reference_masyr',[-5.336,-.809])
    for ax,label,mag in [(bright,'bright','18–21'), (faint,'faint','21–21.5')]:
        centers=np.arange(-19.,20.,2.)
        t=data['cohorts']['target'][label]
        b=data['cohorts']['background'][label]
        th=np.asarray(t['observed_motion_hist'])
        bh=np.asarray(b['observed_motion_hist'])
        mesh=np.ma.masked_less((th-bh).T,2)
        ax.pcolormesh(np.arange(-20.,20.01,2.),np.arange(-20.,20.01,2.),mesh, cmap='Blues',
                      norm=LogNorm(vmin=1,vmax=max(3,np.max(th-bh))))
        if bh.max()>=3:
            ax.contour(centers,centers,bh.T,levels=[2.5],colors=['#b15b43'],linewidths=1.2)
        ax.plot(*reference,linestyle='none',marker='*',mec='#b15b43',mfc='white',ms=12,label='Bright Gaia reference')
        if 'weighted_icrs_mean' in t['candidate_motion']:
            ax.plot(*t['candidate_motion']['weighted_icrs_mean'],linestyle='none',marker='+',mec='k',ms=10,label='Target mean')
        ax.axhline(0,color='#a5a5a5',lw=.5)
        ax.axvline(0,color='#a5a5a5',lw=.5)
        ax.set(xlim=(-20,12),ylim=(-18,14),aspect='equal',
               xlabel=r'$\mu_{\alpha*}$ (mas yr$^{-1}$)',ylabel=r'$\mu_{\delta}$ (mas yr$^{-1}$)',
               title=f"{mag} mag · target {t['matched_frozen_motion']:,}, background {b['matched_frozen_motion']:,}")
    bright.legend(frameon=False,loc='upper right',fontsize=8)
    fig.suptitle('NGC 5466 outskirts · photometric selection and frozen SDSS+NSC motions',fontsize=12)
    fig.savefig(output,dpi=230)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    render(json.loads(args.summary.read_text()),args.output)
