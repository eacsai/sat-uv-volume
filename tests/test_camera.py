import numpy as np
import pytest

from sat_uv import SatelliteCamera, Window, window_around


def cam(**kw):
    base = dict(width=1000, height=800, m_per_px=0.5, centre_e=100.0, centre_n=-50.0, h0=10.0)
    base.update(kw)
    return SatelliteCamera(**base)


def test_centre_and_axes():
    c = cam()
    assert np.allclose(c.project_enu([100.0, -50.0, 10.0]), (500.0, 400.0))
    col, row = c.project_enu([110.0, -50.0, 10.0])         # 10 m east -> right
    assert np.isclose(col, 520.0) and np.isclose(row, 400.0)
    col, row = c.project_enu([100.0, -40.0, 10.0])         # 10 m north -> up
    assert np.isclose(col, 500.0) and np.isclose(row, 380.0)


def test_lean_and_offset_signs():
    p = [100.0, -50.0, 30.0]                                 # 20 m above the datum
    col, row = cam(lean_n=0.5).project_enu(p)
    assert np.isclose(row, 400.0 - 20.0) and np.isclose(col, 500.0)   # leans north by 10 m
    col, row = cam(lean_e=0.5).project_enu(p)
    assert np.isclose(col, 500.0 + 20.0)                                # leans east by 10 m
    col, row = cam(b_e=3.0, b_n=2.0).project_enu([100.0, -50.0, 10.0])
    assert np.isclose(col, 506.0) and np.isclose(row, 396.0)


@pytest.mark.parametrize("frame", ["enu", "eds"])
def test_matrix_matches_projection(frame):
    c = cam(b_e=1.2, b_n=-3.4, lean_e=-0.2, lean_n=0.6)
    X = np.random.default_rng(1111).normal(scale=50.0, size=(50, 3))
    col, row = c.project(X, frame)
    M = c.matrix(frame)
    got = (M @ np.concatenate([X, np.ones((50, 1))], 1).T)
    assert np.allclose(got[0], col) and np.allclose(got[1], row)


def test_eds_frame():
    c = cam()
    # EDS point (x South, y Down, z East): 10 m south of the centre at the datum height
    col, row = c.project([50.0 + 10.0, -10.0, 100.0], "eds")
    assert np.isclose(col, 500.0) and np.isclose(row, 420.0)


def test_window_and_dict():
    c = cam(lean_n=0.4)
    P = np.array([[90.0, -60.0, 10.0], [120.0, -30.0, 70.0]])
    w = window_around(c, P)
    col, row = c.project_enu(P)
    u, v = w.uv(col, row)
    assert (u > 0).all() and (u < 1).all() and (v > 0).all() and (v < 1).all()
    img = np.arange(1000 * 800, dtype=np.int64).reshape(800, 1000)
    crop = Window(-5, 790, 20).crop(img)
    assert crop.shape == (20, 20) and crop[0, 0] == 0 and crop[0, 5] == img[790, 0] and crop[15, 10] == 0
    assert SatelliteCamera.from_dict(c.to_dict()) == c
