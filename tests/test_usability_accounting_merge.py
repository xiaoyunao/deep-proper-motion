from deep_pm.usability_accounting_merge import continuity,overlap_edges
import hashlib
import numpy as np


def hashes(values):return np.asarray([hashlib.sha256(v.encode()).digest()[:16] for v in values],dtype='V16')


def test_changed_measurements_are_not_automatically_new_stars():
    old=hashes(['a','b','c','d','e','f']);new=hashes(['a','b','x','d','e','y'])
    edges,shared=overlap_edges(old,np.array([0,0,0,1,1,1],np.uint32),new,np.array([0,0,0,1,1,1],np.uint32))
    linked=continuity(edges,np.array([3,3]),np.array([3,3]))
    assert shared==4 and linked['unique_reciprocal_continuities']==2


def test_equal_competing_member_tracks_remain_ambiguous():
    linked=continuity({0:3,1:3},np.array([6]),np.array([3,3]))
    assert linked['unique_reciprocal_continuities']==0
