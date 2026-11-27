"""Reader of the satellite camera of one CrossGeo pair directory (pair_new layout: coords.json, ground_<k>_satellite.png).

World frame of a flight: "EDS" (x South, y Down, z East, metres), a local tangent frame of WGS84 (checked on one HK
flight: 96 records, rotation 0.002 deg, scale 0.999996, residual median 5 mm against lat/lon/alt).
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .camera import SatelliteCamera
from .geo import mercator_m_per_px


def read_coords(pair_dir) -> dict:
    """The records of coords.json by name."""
    with open(os.path.join(pair_dir, "coords.json")) as f:
        d = json.load(f)
    if d.get("world_system", "EDS") != "EDS":
        raise ValueError("expected world_system EDS, got %r" % d.get("world_system"))
    return {r["name"]: r for r in d["records"]}


def station(pair_dir, side: int) -> dict:
    """Street-view station of a side: lat, lon, alt and its EDS position."""
    r = read_coords(pair_dir)["ground_%d_pano" % side]
    return {"lat": float(r["ENU"]["latitude"]), "lon": float(r["ENU"]["longitude"]), "alt": float(r["ENU"]["altitude"]),
            "eds": np.asarray(r["EDS"]["t"], np.float64)}


def satellite_camera(pair_dir, side: int, zoom: int = 19, footprint_m: float = 500.0, tol: float = 0.01) -> tuple:
    """(image RGB, SatelliteCamera with b = lean = 0, info) of ground_<side>_satellite.png.

    The PNG is a north-up Web-Mercator crop centred on the station's GPS position, so the camera centre is the
    station's EDS position and m_per_px is the Mercator resolution of the zoom level at its latitude. info records the
    footprint check (side_px * m_per_px against footprint_m)."""
    path = os.path.join(pair_dir, "ground_%d_satellite.png" % side)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    img = np.ascontiguousarray(bgr[..., ::-1])
    h, w = img.shape[:2]
    st = station(pair_dir, side)
    g = mercator_m_per_px(st["lat"], zoom)
    x, _y, z = st["eds"]
    cam = SatelliteCamera(width=w, height=h, m_per_px=g, centre_e=float(z), centre_n=float(-x))
    info = {"path": path, "station_lat": st["lat"], "station_lon": st["lon"], "zoom": zoom, "m_per_px": g,
            "footprint_m": w * g, "footprint_ok": abs(w * g - footprint_m) <= tol * footprint_m}
    return img, cam, info

