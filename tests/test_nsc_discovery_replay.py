from deep_pm.nsc_discovery_replay import compare


def test_replay_matches_physical_identity_despite_wrong_motion_and_keeps_losses():
    def row(sdss,nsc,motion):return dict(kind='extra_r_anchor',magnitude=21.5,detection_ids=['SDSS:'+sdss]+nsc,pm_icrs_masyr=motion,pm_error_masyr=[1.,1.])
    old=[row('a',['NSC:a'],[0.,0.]),row('b',['NSC:b'],[0.,0.]),row('c',['NSC:c'],[0.,0.])]
    new=[row('a',['NSC:a'],[1000.,0.]),row('b',['NSC:b','NSC:extra'],[0.,0.])]
    result=compare(old,new)['by_frozen_class']['extra_r_anchor__r21']
    assert result['frozen_sources']==3
    assert result['identity_status']=={'identical_physical_members':1,'same_sdss_changed_nsc_members':1,'no_unique_same_sdss_member_identity':1}
    assert result['maximum_absolute_pm_change_masyr']==1000.
