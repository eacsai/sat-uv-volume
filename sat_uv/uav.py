"""UAV views -> coloured world points, to calibrate the satellite camera without a building mesh.

A UAV view is a camera-Z depth map (H x W, metres), its intrinsics K on the depth grid, its camera-to-world pose c2w
(OpenCV axes: x right, y down, z forward) and its RGB image (any size; resized to the depth grid). Pixel (r, c) of the
depth map looks along K^-1 [c + centre, r + centre, 1]; CrossGeo's UAV depth maps use centre = 0.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geo import to_enu


@dataclass
class UavView:
    depth: np.ndarray     # [H, W] camera-Z depth, metres
    K: np.ndarray         # 3 x 3 on the depth grid
    c2w: np.ndarray       # 4 x 4, OpenCV camera axes
    rgb: np.ndarray       # [h, w, 3] uint8
    name: str = ""


def back_project(view: UavView, min_depth: float = 0.5, max_depth: float = 400.0, centre: float = 0.0,
                 stride: int = 1) -> tuple:
    """(P [n, 3] world points in the frame of c2w, rgb [n, 3] uint8) of every stride-th depth pixel with
    min_depth < d < max_depth."""
    d = np.asarray(view.depth, np.float64)
    H, W = d.shape
    K = np.asarray(view.K, np.float64)
    c2w = np.asarray(view.c2w, np.float64)
    rgb = np.asarray(view.rgb)
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    rows, cols = np.arange(0, H, stride), np.arange(0, W, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    z = d[::stride, ::stride]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    zz = z[ok]
    cam = np.stack([(cc[ok] + centre - K[0, 2]) / K[0, 0] * zz, (rr[ok] + centre - K[1, 2]) / K[1, 1] * zz, zz], -1)
    P = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return P, np.asarray(rgb[rr[ok], cc[ok]], np.uint8)


def uav_points(views, frame: str = "eds", centre_en=None, radius_m=None, **kw) -> tuple:
    """(P_enu [n, 3], rgb [n, 3]) of all views, converted to ENU; with centre_en = (E, N) and radius_m, only the
    points with |E - E0| <= radius_m and |N - N0| <= radius_m."""
    Ps, Cs = [], []
    for v in views:
        P, C = back_project(v, **kw)
        Ps.append(P)
        Cs.append(C)
    P = to_enu(np.concatenate(Ps, 0) if Ps else np.zeros((0, 3)), frame)
    C = np.concatenate(Cs, 0) if Cs else np.zeros((0, 3), np.uint8)
    if centre_en is not None and radius_m is not None:
        keep = (np.abs(P[:, 0] - centre_en[0]) <= radius_m) & (np.abs(P[:, 1] - centre_en[1]) <= radius_m)
        P, C = P[keep], C[keep]
    return P, C
