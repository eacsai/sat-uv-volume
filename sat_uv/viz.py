"""Figures: mesh edges and voxels on images, the lean score map, UV volume slices."""
from __future__ import annotations

import cv2
import numpy as np

_TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1].copy()


def colour(values, vmin: float, vmax: float) -> np.ndarray:
    t = np.clip((np.asarray(values, np.float64) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    return _TURBO[(t * 255).astype(np.uint8)]


def label(img, text, y: int = 24, scale: float = 0.6):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit_height(img, height: int):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * height / h))), height), interpolation=cv2.INTER_AREA)


def hstack(images, height: int | None = None):
    height = height or min(i.shape[0] for i in images)
    return np.concatenate([fit_height(i, height) for i in images], 1)


def vstack(images):
    w = max(i.shape[1] for i in images)
    return np.concatenate([np.pad(i, ((0, 0), (0, w - i.shape[1]), (0, 0))) for i in images], 0)


def voxel_overlay(image, uv_raw, occ, values, vmin: float, vmax: float, radius: int = 2, order=None,
                  alpha: float = 0.55):
    """Occupied voxels drawn at their (u, v) on `image` (RGB), coloured by `values`; drawn in `order`
    (later on top; default ascending values)."""
    img = np.ascontiguousarray(image).copy()
    h, w = img.shape[:2]
    u = uv_raw[0][occ[:, 0], occ[:, 1], occ[:, 2]]
    v = uv_raw[1][occ[:, 0], occ[:, 1], occ[:, 2]]
    col = colour(values, vmin, vmax)
    over = img.copy()
    for k in (np.argsort(values) if order is None else order):
        if 0 <= u[k] <= 1 and 0 <= v[k] <= 1:
            x, y = int(u[k] * w), int(v[k] * h)
            cv2.rectangle(over, (x - radius, y - radius), (x + radius, y + radius), tuple(int(c) for c in col[k]), -1)
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


def edge_overlay(crop, window, col, row, rgb=(0, 255, 0)):
    """Image-pixel points (col, row) drawn on the window crop (already resized to any square size)."""
    out = np.ascontiguousarray(crop).copy()
    s = out.shape[0] / window.size
    x = ((np.asarray(col) - window.col0) * s).astype(int)
    y = ((np.asarray(row) - window.row0) * s).astype(int)
    ok = (x >= 0) & (x < out.shape[1]) & (y >= 0) & (y < out.shape[0])
    out[y[ok], x[ok]] = rgb
    return out


def score_map(result, cell: int = 24, marks=(), title: str = "edge NCC vs lean: lean_e right, lean_n up"):
    """The coarse lean grid of a CalibResult, lean_e to the right, lean_n up; marks: (lean_e, lean_n, rgb) circles."""
    g = result.lean_grid
    s = np.flipud(result.coarse_ncc)
    img = colour(s, np.nanmin(s), np.nanmax(s))
    img = cv2.resize(img, (cell * len(g), cell * len(g)), interpolation=cv2.INTER_NEAREST).copy()
    step = (g[-1] - g[0]) / (len(g) - 1)
    for le, ln, rgb in marks:
        cv2.circle(img, (int((le - g[0]) / step * cell + cell / 2), int((g[-1] - ln) / step * cell + cell / 2)), 8, rgb, 2)
    img = cv2.copyMakeBorder(img, 44, 8, 8, 8, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    label(img, title, 18, 0.5)
    label(img, "range %.2f .. %.2f" % (-g[-1], g[-1]), 38, 0.5)
    return img


def uv_slice(vol: dict, occ, axis: int = 2, index: int | None = None, size: int = 384, names=("axis0", "axis1"),
             prefix: str = ""):
    """2x2 panel: ssuv, u, v of the slice at `index` along `axis` (default: the slice with most occupied
    voxels); image rows = the second remaining axis, columns = the first; occupied voxels white."""
    others = [a for a in range(3) if a != axis]
    if index is None:
        index = int(np.bincount(occ[:, axis]).argmax()) if len(occ) else vol["ssuv"].shape[1] // 2
    occ_img = np.zeros(vol["ssuv"].shape[1:3], bool)
    sel = occ[:, axis] == index
    occ_img[occ[sel, others[1]], occ[sel, others[0]]] = True
    panels = []
    for name, arr in (("ssuv (mask)", vol["ssuv"][0]), ("u", vol["uv_volume"][0]), ("v", vol["uv_volume"][1])):
        sl = np.take(arr, index, axis=axis).T
        im = cv2.resize(colour(sl, 0, 1), (size, size), interpolation=cv2.INTER_NEAREST).copy()
        im[cv2.resize(occ_img.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0] = 255
        panels.append(label(im, "%s%s | rows %s, cols %s" % (prefix, name, names[1], names[0]), 20, 0.45))
    return panels, index
