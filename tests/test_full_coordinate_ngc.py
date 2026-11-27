import numpy as np

from deep_pm.astrometry import project,unproject
from deep_pm.full_coordinate_ngc import reexpress_positions


def test_saved_bright_apparent_positions_keep_their_sky_coordinates_across_charts():
    sky=np.array([[359.99,35.],[.01,35.],[.02,35.001]])
    old=(350.,30.);new=(25.,0.);positions=project(*sky.T,old)
    assert reexpress_positions(positions,old,old) is positions
    converted=reexpress_positions(positions,old,new)
    recovered=unproject(converted,new)
    np.testing.assert_allclose(recovered,sky,atol=2e-11,rtol=0)


def test_same_origin_different_projection_reexpresses_saved_bright_positions():
    from deep_pm.stereographic import Stereographic
    from deep_pm.astrometry import project
    old=(359.,40.)
    new=Stereographic(old)
    ra,dec=[1.,70.],[29.,18.]
    tangent=project(ra,dec,old)
    np.testing.assert_allclose(reexpress_positions(tangent,old,new),project(ra,dec,new),rtol=2e-14)
