import pytest

from deep_pm.full_coordinate_production_readout import aggregate


def test_receipt_denominators_keep_component_applications_separate():
    proof=dict(research_solutions=7,r_control_solutions=5,physical_r_control_solutions=4,
        by_class=dict(r_existing=5,extra_r_anchor=2),funnel={'0':{'legacy_r':{'21_to_22':{'initial':10,'final':5}}}},
        support={'0':dict(core_group_present=100,core_local12mas_supported=80)},elapsed_seconds=2.,peak_rss_mib=300.)
    other=dict(proof,support={'1':dict(core_group_present=200,core_local12mas_supported=190)},peak_rss_mib=500.)
    result=aggregate([proof,other])
    assert result['origin_counts']['research_solutions']==14
    assert result['selection_funnel']['legacy_r/21_to_22/initial']==20
    assert result['component_detection_applications']=={'0':dict(core_group_present=100,core_local12mas_supported=80),'1':dict(core_group_present=200,core_local12mas_supported=190)}
    assert result['maximum_source_process_rss_mib']==500.
    assert result['summed_source_worker_elapsed_seconds']==4.
    with pytest.raises(ValueError):aggregate([dict(proof,funnel={'0':dict(unexpected_quantile=[1,2,3])})])
