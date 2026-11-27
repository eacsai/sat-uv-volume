"""Geodesy and world frames.

- WGS84 latitude / longitude / ellipsoidal height <-> ECEF <-> a local East-North-Up (ENU) tangent frame.
- Two gravity-aligned metric world frames are understood by the rest of the package:
    "enu": x = East, y = North, z = Up
    "eds": x = South (-North), y = Down, z = East   (the CrossGeo / tri-view convention)
- Web-Mercator ground resolution (metres per pixel) of a zoom level at a latitude.
"""
from __future__ import annotations

import math

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
MERCATOR_Z0_M_PER_PX = 2.0 * math.pi * WGS84_A / 256.0  # 156543.03392804097 m per pixel at zoom 0, equator

# enu = ENU_FROM[frame] @ x  (rows: East, North, Up)
ENU_FROM = {
    "enu": np.eye(3),
    "eds": np.array([[0.0, 0.0, 1.0],     # East  = z
                     [-1.0, 0.0, 0.0],    # North = -x
                     [0.0, -1.0, 0.0]]),  # Up    = -y
}


def enu_matrix(frame: str) -> np.ndarray:
    """3x3 rotation R with enu = R @ x for a point x given in `frame` ("enu" or "eds")."""
    try:
        return ENU_FROM[frame].copy()
    except KeyError:
        raise ValueError("unknown frame %r (expected 'enu' or 'eds')" % (frame,)) from None


def to_enu(X, frame: str = "enu") -> np.ndarray:
    """Points (..., 3) in `frame` -> ENU."""
    return np.asarray(X, np.float64) @ enu_matrix(frame).T


def from_enu(P, frame: str = "enu") -> np.ndarray:
    """ENU points (..., 3) -> `frame`."""
    return np.asarray(P, np.float64) @ enu_matrix(frame)


def lla_to_ecef(lat, lon, alt) -> np.ndarray:
    """WGS84 geodetic (degrees, degrees, metres) -> ECEF metres, shape (..., 3)."""
    lat, lon, alt = (np.asarray(a, np.float64) for a in (lat, lon, alt))
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(la) ** 2)
    return np.stack([(n + alt) * np.cos(la) * np.cos(lo),
                     (n + alt) * np.cos(la) * np.sin(lo),
                     (n * (1.0 - WGS84_E2) + alt) * np.sin(la)], -1)


def ecef_to_lla(P) -> tuple:
    """ECEF metres (..., 3) -> (lat, lon, alt) in degrees, degrees, metres (iterative, converges to < 1e-9 deg)."""
    P = np.asarray(P, np.float64)
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def _ecef_to_enu_rotation(lat0: float, lon0: float) -> np.ndarray:
    la, lo = math.radians(lat0), math.radians(lon0)
    return np.array([[-math.sin(lo), math.cos(lo), 0.0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def lla_to_enu(lat, lon, alt, origin) -> np.ndarray:
    """WGS84 points -> local ENU metres about origin = (lat0, lon0, alt0)."""
    R = _ecef_to_enu_rotation(origin[0], origin[1])
    return (lla_to_ecef(lat, lon, alt) - lla_to_ecef(*origin)) @ R.T


def enu_to_lla(P, origin) -> tuple:
    """Local ENU metres about origin -> WGS84 (lat, lon, alt)."""
    R = _ecef_to_enu_rotation(origin[0], origin[1])
    return ecef_to_lla(np.asarray(P, np.float64) @ R + lla_to_ecef(*origin))


def mercator_m_per_px(lat_deg: float, zoom: int) -> float:
    """Ground metres per pixel of a Web-Mercator (256 px tiles) image at a latitude and zoom level."""
    return MERCATOR_Z0_M_PER_PX * math.cos(math.radians(lat_deg)) / (2 ** zoom)
