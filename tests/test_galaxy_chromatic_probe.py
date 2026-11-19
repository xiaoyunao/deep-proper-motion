import numpy as np
from deep_pm.galaxy_chromatic_probe import assess,correlation


def test_held_physical_galaxies_reveal_color_signal_beyond_constant():
    rng=np.random.default_rng(55)
    def observations(n):
        color=rng.uniform(0,3,n);mag=rng.uniform(18,22,n);noise=rng.normal(0,10,(n,2))
        value=noise+np.column_stack([60*(color-1.2),-40*(color-1.2)])
        return [dict(color=c,r=m,residual=v,error=np.array([20.,20.]),seeing=1.2,airmass=1.3,epoch=2016.) for c,m,v in zip(color,mag,value)]
    result=assess(observations(300),observations(200))
    assert result['status']=='diagnostic_only'
    assert result['paired_gain_vs_constant']['color']['object_bootstrap95'][0]>.5
    assert correlation(np.ones(30),np.arange(30)) is None
