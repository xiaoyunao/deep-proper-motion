import pandas as pd
from deep_pm.full_coordinate_continuity import choose_pairs


def test_changed_physical_identity_requires_mutual_unique_support():
    q=pd.DataFrame([(1,11,8,10,10),(1,12,7,10,10),(2,12,8,10,10),(3,13,6,10,10),(3,14,6,10,10),(4,15,5,10,10)],columns=['old_src','new_src','shared','old_n','new_n'])
    out=choose_pairs(q)
    assert set(zip(out.old_src,out.new_src))=={(1,11),(2,12)}


def test_nonreciprocal_other_candidate_does_not_erase_an_identity_tie():
    q=pd.DataFrame([(1,11,6,10,10),(1,12,6,10,10),(2,12,8,10,10)],columns=['old_src','new_src','shared','old_n','new_n'])
    out=choose_pairs(q)
    assert set(zip(out.old_src,out.new_src))=={(2,12)}
