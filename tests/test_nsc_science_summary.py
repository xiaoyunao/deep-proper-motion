import numpy as np
from deep_pm.nsc_science_summary import known_cohorts


def test_pool_uses_identical_sources_and_individual_vectors_for_p68():
    rows=[]
    for j in range(6):
        values={k:([float(value),0.],[1.,1.]) for k,value in [('r_legacy_quality',2+j),('r_physical_quality',3+j),('all_nsc',1+j),('NSC',4+j),('Xiao',5+j)]}
        if j==0:del values['NSC']
        rows.append(dict(magnitude=21.5,shape_class='pointlike' if j%2==0 else 'extended_or_mixed',sdss_runs=1,nsc_span=3.,values=values,exposures=[f'SDSS:{j}',f'NSC:{j}'],nsc_epoch_days=[j]))
    cohorts=known_cohorts(rows)
    assert cohorts['r21__one_run_pointlike']['same_objects']['n']==3
    assert cohorts['r21__one_run_extended_or_mixed']['same_objects']['n']==3
    output=cohorts['r21__all']
    assert output['same_objects']['n']==6
    common=output['strict_published_common'];assert common['n']==5
    assert common['methods']['all_nsc']['radial_abs68']==np.quantile(np.arange(2,7),.68)
    assert common['methods']['r_legacy_quality']['radial_abs68']==np.quantile(np.arange(3,8),.68)
    assert output['observational_blocks']['physical']['independent_blocks']==5
