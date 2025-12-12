# Pytest
import pytest

# Numpy
import numpy as np

# Cython BBox
try:
    import cython_bbox
    CYTHON_BBOX_AVAILABLE = True
except ImportError:
    CYTHON_BBOX_AVAILABLE = False
    pass

# Utils
from alutils import BBox

# Logging
from alutils import get_logger
get_logger("alutils").setLevel("CRITICAL")

def test_general():

    with pytest.raises(Exception): b = BBox(1, 1, -3, 0)

    for _ in range(100):
        bbox = BBox.random()

        with pytest.raises(Exception): bbox.x = "cat"
        with pytest.raises(Exception): bbox.x = 9
        with pytest.raises(Exception): b = bbox * -2

if CYTHON_BBOX_AVAILABLE:
    def test_cython():

        for _ in range(100):
            bbox = BBox.random()
            assert BBox.iou(bbox, bbox) == pytest.approx(1)
            assert BBox.iou(bbox, bbox * 2) == pytest.approx(0.25)

            bbox_shift = bbox + (bbox.w, bbox.h)
            assert BBox.iou(bbox, bbox_shift) == pytest.approx(0)

            bbox_shift = bbox + (bbox.w / 2, 0)
            assert BBox.iou(bbox, bbox_shift) == pytest.approx(1/3)

def test_operators():

    for _ in range(100):
        bbox = BBox.random()

        assert bbox + 7 == BBox(bbox.x + 7, bbox.y + 7, bbox.w, bbox.h)
        assert bbox + (-3, -71) == BBox(bbox.x - 3, bbox.y - 71, bbox.w, bbox.h)
        assert (bbox * 3).area == pytest.approx(bbox.area * 3**2)

def test_containment():

    for _ in range(100):
        bbox = BBox.random()

        assert bbox in bbox
        assert bbox in (bbox + (0, 0))
        assert (bbox.x, bbox.y) in bbox
        assert [bbox.x, bbox.y] in bbox
        assert (bbox.x2, bbox.y2) in bbox
        assert (bbox.x + bbox.w/3, bbox.y + bbox.h/2) in bbox
        assert bbox.corners()[0] in bbox
        assert bbox * 0.8 in bbox
        assert bbox * 0.5 in bbox
        assert bbox * 0.49 + (bbox.w/4, bbox.h/4) in bbox

        assert not bbox in (bbox + (bbox.w, bbox.h))
        assert not (bbox.x1 - 1, bbox.y1 - 1) in bbox
        assert not (bbox.x2 + 1, bbox.y2 + 1) in bbox
        assert not bbox * 0.5 + (bbox.w/2, bbox.h/2) in bbox
