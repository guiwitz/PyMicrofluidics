from pymicrofluidics.mfdesign import Design, Feature
import numpy as np

def test_define_polygon():

    polygon = Feature.define_polygon([[0, 0],[1,0]])
    assert len(polygon.coord) == 1
    assert len(polygon.coord[0]) == 2

def test_mirroring():

    polygon = Feature.define_polygon([[0, 0],[1,0]])
    mirror_polygon = polygon.mirror_feature(0)
    np.testing.assert_array_equal(mirror_polygon.coord[0], np.array([[0,0],[-1,0]]))