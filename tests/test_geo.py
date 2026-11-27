import math

import numpy as np

from sat_uv import geo


def test_mercator_resolution():
    assert abs(geo.mercator_m_per_px(0.0, 19) - 0.29858214) < 1e-7
    # the CrossGeo HK satellite: 1810 px over 500 m at 22.3255 N
    assert abs(geo.mercator_m_per_px(22.3255, 19) * 1810 - 500.0) < 0.2


def test_enu_axes_and_scale():
    o = (22.0, 114.0, 10.0)
    lat = math.radians(o[0])
    m = geo.WGS84_A * (1 - geo.WGS84_E2) / (1 - geo.WGS84_E2 * math.sin(lat) ** 2) ** 1.5   # metres per radian, north
    n = geo.WGS84_A / math.sqrt(1 - geo.WGS84_E2 * math.sin(lat) ** 2) * math.cos(lat)       # metres per radian, east
    north = geo.lla_to_enu(o[0] + 0.001, o[1], o[2], o)
    east = geo.lla_to_enu(o[0], o[1] + 0.001, o[2], o)
    up = geo.lla_to_enu(o[0], o[1], o[2] + 7.0, o)
    assert abs(north[1] - m * math.radians(0.001)) < 0.01 and abs(north[0]) < 1e-6
    assert abs(east[0] - n * math.radians(0.001)) < 0.01 and abs(east[1]) < 0.01
    assert np.allclose(up, [0, 0, 7.0], atol=1e-6)


def test_lla_round_trip():
    o = (22.3255, 114.2108, 7.5)
    P = np.array([[120.0, -340.0, 55.0], [-10.0, 3.0, -2.0]])
    lat, lon, alt = geo.enu_to_lla(P, o)
    assert np.allclose(geo.lla_to_enu(lat, lon, alt, o), P, atol=1e-6)


def test_eds_enu():
    # EDS x = South, y = Down, z = East
    assert np.allclose(geo.to_enu([1.0, 0, 0], "eds"), [0, -1, 0])
    assert np.allclose(geo.to_enu([0, 1.0, 0], "eds"), [0, 0, -1])
    assert np.allclose(geo.to_enu([0, 0, 1.0], "eds"), [1, 0, 0])
    X = np.random.default_rng(1111).normal(size=(10, 3))
    assert np.allclose(geo.from_enu(geo.to_enu(X, "eds"), "eds"), X)
