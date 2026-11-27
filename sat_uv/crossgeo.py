"""Readers of one CrossGeo pair directory (pair_new layout): the satellite cameras (coords.json, ground_<k>_satellite.png)
and the UAV views (uav_<k>_depth.npy, uav_<k>_rgb.npy pose, uav_<k>_rgb.jpg).

World frame of a flight: "EDS" (x South, y Down, z East, metres), a local tangent frame of WGS84 (checked on one HK
flight: 96 records, rotation 0.002 deg, scale 0.999996, residual median 5 mm against lat/lon/alt).
"""
from __future__ import annotations

import io
import json
import os
import pickle

import cv2
import numpy as np
import numpy.lib.format as npf

from .camera import SatelliteCamera
from .geo import mercator_m_per_px
from .uav import UavView

_POSE_GLOBALS = {(m, n) for m in ("numpy.core.multiarray", "numpy._core.multiarray") for n in ("_reconstruct", "scalar")}
_POSE_GLOBALS |= {("numpy", "ndarray"), ("numpy", "dtype")}


class _PoseUnpickler(pickle.Unpickler):
    """Unpickles numpy arrays and plain containers only."""

    def find_class(self, module, name):
        if (module, name) not in _POSE_GLOBALS:
            raise pickle.UnpicklingError("pose file names a forbidden global %s.%s" % (module, name))
        return super().find_class(module, name)


def load_pose(path) -> tuple:
    """(K 3x3, c2w 4x4) float64 of a CrossGeo pose .npy (a 0-d object array holding a dict with "intrinsics" and
    "c2w"), read without executing arbitrary pickle code."""
    with open(path, "rb") as f:
        data = f.read()
    f = io.BytesIO(data)
    version = npf.read_magic(f)
    _shape, _fortran, dtype = (npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0)(f)
    if not dtype.hasobject:
        raise ValueError("%s: not a pose file (dtype %s)" % (path, dtype))
    arr = _PoseUnpickler(f).load()
    m = arr.item() if isinstance(arr, np.ndarray) else arr
    return np.asarray(m["intrinsics"], np.float64), np.asarray(m["c2w"], np.float64)


def uav_views(pair_dir, names=("uav_1", "uav_2")) -> list:
    """The UAV views of a pair: camera-Z depth [540, 960], K on the depth grid, c2w (EDS, OpenCV axes), RGB."""
    out = []
    for n in names:
        K, c2w = load_pose(os.path.join(pair_dir, n + "_rgb.npy"))
        depth = np.load(os.path.join(pair_dir, n + "_depth.npy"), allow_pickle=False).astype(np.float32)
        bgr = cv2.imread(os.path.join(pair_dir, n + "_rgb.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(os.path.join(pair_dir, n + "_rgb.jpg"))
        out.append(UavView(depth=depth, K=K, c2w=c2w, rgb=np.ascontiguousarray(bgr[..., ::-1]), name=n))
    return out


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

