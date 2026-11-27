"""Calibrate the satellite camera (ground offset b and lean) from the satellite image and either

- a mesh of the building (calibrate_satellite): for a candidate lean, points sampled densely on the mesh surface are
  projected with the camera (b = 0) onto a working grid; every grid cell keeps the highest point (seen from above, roofs
  hide what is below them). The edges of that predicted height image (and of its silhouette) are slid over the gradient
  magnitude of the satellite image by normalised cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED) within
  +-b_max_m; the best shift is b, its score is the lean's score.
- coloured UAV points (calibrate_satellite_uav, points from uav.uav_points): the same, but the predicted image is the
  luminance of the highest UAV point of every cell (texture of roofs and ground, not only edges), and the correlation
  counts only the cells the UAV views cover (masked NCC), so occlusion holes do not create false edges.

Leans are searched on a coarse grid, then on a fine grid around the best one.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import SatelliteCamera
from .geo import to_enu


@dataclass
class CalibConfig:
    work_m_per_px: float = 0.5      # resolution of the working grid
    lean_max: float = 1.0           # coarse lean grid: [-lean_max, lean_max]^2 ...
    lean_step: float = 0.1          # ... in steps of lean_step
    fine_half: float = 0.1          # fine grid: best coarse lean +- fine_half ...
    fine_step: float = 0.02         # ... in steps of fine_step
    b_max_m: float = 12.0           # search range of b (metres, both axes)
    pad_m: float = 10.0             # margin of the rendered template
    blur_px: float = 1.0            # Gaussian blur before the Sobel gradients (working-grid pixels)
    samples_per_m2: float = 12.0    # surface sampling density
    min_samples: int = 300_000
    max_samples: int = 1_500_000
    second_min_dist: float = 0.25   # the reported second peak lies farther than this from the best coarse lean
    seed: int = 1111


@dataclass
class UavCalibConfig(CalibConfig):
    work_m_per_px: float = 1.0      # UAV depth pixels are ~1 m on the ground from 150-250 m away
    ground_percentile: float = 5.0  # datum h0 = this percentile of the UAV points' heights (when not given)
    erode_px: int = 1               # covered cells within this distance of a hole are not scored
    fill_min: float = 0.3           # an empty cell is filled from its 3x3 neighbours when they cover >= this fraction


def sample_surface(vertices, faces, n: int, seed: int = 1111) -> np.ndarray:
    """n points uniformly distributed (by area) on a triangle mesh."""
    V = np.asarray(vertices, np.float64)
    T = V[np.asarray(faces, np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(T), size=int(n), p=area / area.sum())
    r1, r2 = rng.random(int(n)), rng.random(int(n))
    s = np.sqrt(r1)
    a, b, c = 1.0 - s, s * (1.0 - r2), s * r2
    return a[:, None] * T[tri, 0] + b[:, None] * T[tri, 1] + c[:, None] * T[tri, 2]


def mesh_area(vertices, faces) -> float:
    V = np.asarray(vertices, np.float64)
    T = V[np.asarray(faces, np.int64)]
    return float(0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1).sum())


def luminance(image) -> np.ndarray:
    img = np.asarray(image, np.float32)
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def gradient_magnitude(gray, blur_px: float) -> np.ndarray:
    g = cv2.GaussianBlur(np.asarray(gray, np.float32), (0, 0), blur_px) if blur_px > 0 else np.asarray(gray, np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _parabola(cm: float, c0: float, cp: float) -> float:
    d = cm - 2.0 * c0 + cp
    if not np.isfinite(d) or d >= 0:
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / d, -0.5, 0.5))


def masked_ncc(I, T, M) -> np.ndarray:
    """Normalised cross-correlation of template T over image I at every shift (shape (I.h - T.h + 1, I.w - T.w + 1),
    like cv2.matchTemplate TM_CCOEFF_NORMED), counting only the template pixels where M > 0. -1 where undefined."""
    I = np.asarray(I, np.float64)
    m = np.asarray(M, np.float64) > 0
    if m.sum() < 2:
        return np.full((I.shape[0] - T.shape[0] + 1, I.shape[1] - T.shape[1] + 1), -1.0)
    sd = I.std()
    I = ((I - I.mean()) / (sd if sd > 0 else 1.0)).astype(np.float32)
    Mf = m.astype(np.float32)
    Tm = np.where(m, np.asarray(T, np.float64), 0.0)
    n = float(m.sum())
    t_mean = Tm.sum() / n
    Tc = np.where(m, Tm - t_mean, 0.0).astype(np.float32)          # zero-mean template on the mask
    sTT = float((Tc.astype(np.float64) ** 2).sum())
    sI = cv2.matchTemplate(I, Mf, cv2.TM_CCORR).astype(np.float64)
    sII = cv2.matchTemplate(I * I, Mf, cv2.TM_CCORR).astype(np.float64)
    sIT = cv2.matchTemplate(I, Tc, cv2.TM_CCORR).astype(np.float64)
    den = np.sqrt(np.maximum(sII - sI * sI / n, 0.0) * sTT)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-9, sIT / den, -1.0)
    return out


def _peak(S: np.ndarray) -> tuple:
    """(value, row, col) of the maximum of S with sub-pixel parabola offsets."""
    a, c = np.unravel_index(int(np.argmax(S)), S.shape)
    da = _parabola(S[a - 1, c], S[a, c], S[a + 1, c]) if 0 < a < S.shape[0] - 1 else 0.0
    dc = _parabola(S[a, c - 1], S[a, c], S[a, c + 1]) if 0 < c < S.shape[1] - 1 else 0.0
    return float(S[a, c]), a + da, c + dc


class _WorkGrid:
    """The satellite image's gradient on a working grid of cfg.work_m_per_px, padded for shifts of +-b_max_m."""

    def __init__(self, image_rgb, camera: SatelliteCamera, cfg: CalibConfig):
        self.cfg = cfg
        self.cam = camera
        h, w = image_rgb.shape[:2]
        if (w, h) != (camera.width, camera.height):
            raise ValueError("image is %dx%d, camera expects %dx%d" % (w, h, camera.width, camera.height))
        self.nw = max(8, int(round(w * camera.m_per_px / self.cfg.work_m_per_px)))
        self.nh = max(8, int(round(h * camera.m_per_px / self.cfg.work_m_per_px)))
        self.sx, self.sy = self.nw / w, self.nh / h           # working-grid pixels per image pixel
        self.mx, self.my = camera.m_per_px / self.sx, camera.m_per_px / self.sy  # metres per working pixel
        work = cv2.resize(np.asarray(image_rgb), (self.nw, self.nh), interpolation=cv2.INTER_AREA)
        self.grad = gradient_magnitude(luminance(work), self.cfg.blur_px).astype(np.float32)
        self.pad = int(round(self.cfg.pad_m / min(self.mx, self.my)))
        self.rs_x = int(round(self.cfg.b_max_m / self.mx))
        self.rs_y = int(round(self.cfg.b_max_m / self.my))
        py, px = self.pad + self.rs_y, self.pad + self.rs_x
        self.gpad = cv2.copyMakeBorder(self.grad, py, py, px, px, cv2.BORDER_CONSTANT, value=0)

    def _cells(self, camera: SatelliteCamera, P) -> tuple:
        col, row = camera.project_enu(P)
        return np.floor(row * self.sy).astype(np.int64), np.floor(col * self.sx).astype(np.int64)

    def _region(self, E, i0, j0, extra_y, extra_x):
        a0 = i0 + self.pad + self.rs_y - extra_y
        c0 = j0 + self.pad + self.rs_x - extra_x
        if a0 < 0 or c0 < 0:
            return None
        reg = self.gpad[a0:a0 + E.shape[0] + 2 * extra_y, c0:c0 + E.shape[1] + 2 * extra_x]
        if reg.shape != (E.shape[0] + 2 * extra_y, E.shape[1] + 2 * extra_x):
            return None
        return reg

    def _shift(self, camera: SatelliteCamera) -> tuple:
        """(dx, dy) working-grid pixels of camera's b, after writing it with this grid's datum."""
        cam = camera.rebase(self.cam.h0)
        return cam, cam.b_e / self.mx, -cam.b_n / self.my


class MeshEdgeRenderer(_WorkGrid):
    """Predicted edge images of a mesh in a satellite image, on a working grid, and their match to the image."""

    def __init__(self, image_rgb, camera: SatelliteCamera, points_enu, cfg: CalibConfig | None = None):
        super().__init__(image_rgb, camera, cfg or CalibConfig())
        self.P = np.asarray(points_enu, np.float64)
        self.h = self.P[:, 2] - camera.h0
        self.hmin = float(np.percentile(self.h, 1.0))
        self.htop = float(np.percentile(self.h, 99.5))

    def edges(self, lean_e: float, lean_n: float) -> tuple:
        """(E, i0, j0): the predicted edge image for this lean and b = 0; E[a, c] is working-grid pixel (i0 + a, j0 + c)."""
        cam = self.cam.with_params(b_e=0.0, b_n=0.0, lean_e=float(lean_e), lean_n=float(lean_n))
        i, j = self._cells(cam, self.P)
        i0, j0 = int(i.min()) - self.pad, int(j.min()) - self.pad
        hh, ww = int(i.max()) - i0 + self.pad + 1, int(j.max()) - j0 + self.pad + 1
        img = np.full(hh * ww, -np.inf)
        np.maximum.at(img, (i - i0) * ww + (j - j0), self.h)    # highest point wins
        img = img.reshape(hh, ww)
        cov = np.isfinite(img).astype(np.float32)
        R = np.where(cov > 0, np.clip(img - self.hmin, 0.0, None) / max(self.htop - self.hmin, 1.0), 0.0).astype(np.float32)
        k3 = np.ones((3, 3), np.uint8)
        R = cv2.morphologyEx(R, cv2.MORPH_CLOSE, k3)
        cov = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, k3)
        E = gradient_magnitude(R, self.cfg.blur_px) + gradient_magnitude(cov, self.cfg.blur_px)
        return E.astype(np.float32), i0, j0

    def match(self, lean_e: float, lean_n: float) -> tuple:
        """(ncc, b_e, b_n) of the best shift of this lean's edges within +-b_max_m; ncc = -1 if out of the image."""
        E, i0, j0 = self.edges(lean_e, lean_n)
        reg = self._region(E, i0, j0, self.rs_y, self.rs_x)
        if reg is None or E.max() <= 0:
            return -1.0, 0.0, 0.0
        s, a, c = _peak(cv2.matchTemplate(reg, E, cv2.TM_CCOEFF_NORMED))
        dy, dx = a - self.rs_y, c - self.rs_x
        return s, dx * self.mx, -dy * self.my

    def score(self, camera: SatelliteCamera) -> float:
        """NCC of the edges of `camera` (its lean and its b, no search) against the image gradient."""
        camera, dx, dy = self._shift(camera)
        E, i0, j0 = self.edges(camera.lean_e, camera.lean_n)
        reg = self._region(E, i0, j0, 0, 0)
        if reg is None or E.max() <= 0:
            return float("nan")
        Es = cv2.warpAffine(E, np.float32([[1, 0, dx], [0, 1, dy]]), (E.shape[1], E.shape[0]), flags=cv2.INTER_LINEAR)
        return float(cv2.matchTemplate(reg, Es, cv2.TM_CCOEFF_NORMED)[0, 0])

    def edge_pixels(self, camera: SatelliteCamera, percentile: float = 60.0) -> tuple:
        """(col, row) image-pixel coordinates of the strongest predicted edges of `camera` (for figures)."""
        camera = camera.rebase(self.cam.h0)
        E, i0, j0 = self.edges(camera.lean_e, camera.lean_n)
        if not (E > 0).any():
            return np.zeros(0), np.zeros(0)
        yy, xx = np.nonzero(E > np.percentile(E[E > 0], percentile))
        col = (xx + j0 + 0.5) / self.sx + camera.b_e / camera.m_per_px
        row = (yy + i0 + 0.5) / self.sy - camera.b_n / camera.m_per_px
        return col, row


class UavTextureRenderer(_WorkGrid):
    """Predicted luminance of the satellite image from coloured UAV points (highest point of a cell wins), on a working
    grid, and its masked match to the image's gradient (only cells covered by UAV points count)."""

    def __init__(self, image_rgb, camera: SatelliteCamera, points_enu, rgb, cfg: UavCalibConfig | None = None):
        super().__init__(image_rgb, camera, cfg or UavCalibConfig())
        P = np.asarray(points_enu, np.float64)
        if len(P) == 0:
            raise ValueError("no UAV points")
        h = P[:, 2] - camera.h0
        o = np.argsort(h, kind="stable")                         # rank = index: higher points have higher ranks
        self.P, self.h = P[o], h[o]
        self.rgb = np.asarray(rgb, np.uint8)[o]
        self.lum = luminance(self.rgb.astype(np.float32)[None])[0]
        self.fill_k = np.ones((3, 3), np.float32) / 9.0

    def _top(self, i, j, hh, ww):
        """Index (rank) of the highest point of every cell, -1 where empty."""
        best = np.full(hh * ww, -1, np.int64)
        np.maximum.at(best, i * ww + j, np.arange(len(i), dtype=np.int64))
        return best.reshape(hh, ww)

    def _fill(self, vals, valid):
        """Fill empty cells from their 3x3 neighbours (when they cover >= fill_min); returns (vals, valid)."""
        v = valid.astype(np.float32)
        den = cv2.filter2D(v, -1, self.fill_k, borderType=cv2.BORDER_CONSTANT)
        if vals.ndim == 2:
            num = cv2.filter2D(vals * v, -1, self.fill_k, borderType=cv2.BORDER_CONSTANT)
        else:
            num = np.stack([cv2.filter2D(vals[..., k] * v, -1, self.fill_k, borderType=cv2.BORDER_CONSTANT)
                            for k in range(vals.shape[2])], -1)
            den = den[..., None]
        fill = (~valid) & (den.reshape(valid.shape) >= self.cfg.fill_min)
        out = vals.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[fill] = (num / np.maximum(den, 1e-6))[fill]
        return out, valid | fill

    def render(self, lean_e: float, lean_n: float) -> tuple:
        """(L, valid, i0, j0): luminance of the highest UAV point per working-grid cell for this lean and b = 0."""
        cam = self.cam.with_params(b_e=0.0, b_n=0.0, lean_e=float(lean_e), lean_n=float(lean_n))
        i, j = self._cells(cam, self.P)
        i0, j0 = int(i.min()) - self.pad, int(j.min()) - self.pad
        hh, ww = int(i.max()) - i0 + self.pad + 1, int(j.max()) - j0 + self.pad + 1
        best = self._top(i - i0, j - j0, hh, ww)
        valid = best >= 0
        L = np.zeros((hh, ww), np.float32)
        L[valid] = self.lum[best[valid]]
        L, valid = self._fill(L, valid)
        return L, valid, i0, j0

    def template(self, lean_e: float, lean_n: float) -> tuple:
        """(T, M, i0, j0): gradient magnitude of the rendered luminance and the mask of cells that are scored."""
        L, valid, i0, j0 = self.render(lean_e, lean_n)
        T = gradient_magnitude(L, self.cfg.blur_px)
        r = int(self.cfg.erode_px) + 1                          # the Sobel stencil reaches 1 cell into a hole
        M = cv2.erode(valid.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        return T, M, i0, j0

    def match(self, lean_e: float, lean_n: float) -> tuple:
        """(ncc, b_e, b_n) of the best shift of this lean's predicted texture within +-b_max_m."""
        T, M, i0, j0 = self.template(lean_e, lean_n)
        reg = self._region(T, i0, j0, self.rs_y, self.rs_x)
        if reg is None or M.sum() < 50:
            return -1.0, 0.0, 0.0
        s, a, c = _peak(masked_ncc(reg, T, M))
        dy, dx = a - self.rs_y, c - self.rs_x
        return s, dx * self.mx, -dy * self.my

    def score(self, camera: SatelliteCamera) -> float:
        """Masked NCC of `camera` (its lean and its b, no search) against the image gradient."""
        camera, dx, dy = self._shift(camera)
        T, M, i0, j0 = self.template(camera.lean_e, camera.lean_n)
        reg = self._region(T, i0, j0, 0, 0)
        if reg is None or M.sum() < 50:
            return float("nan")
        A = np.float32([[1, 0, dx], [0, 1, dy]])
        Ts = cv2.warpAffine(T, A, (T.shape[1], T.shape[0]), flags=cv2.INTER_LINEAR)
        Ms = cv2.warpAffine(M.astype(np.float32), A, (T.shape[1], T.shape[0]), flags=cv2.INTER_LINEAR) > 0.99
        return float(masked_ncc(reg, Ts, Ms)[0, 0])

    def predicted_rgb(self, camera: SatelliteCamera, window, size: int) -> tuple:
        """(rgb [size, size, 3] uint8, valid [size, size]) of the UAV points seen through `camera` in a satellite window
        resampled to size x size (highest point wins, small holes filled)."""
        col, row = camera.project_enu(self.P)
        u, v = window.uv(col, row)
        i, j = np.floor(v * size).astype(np.int64), np.floor(u * size).astype(np.int64)
        ok = (i >= 0) & (i < size) & (j >= 0) & (j < size)
        best = np.full(size * size, -1, np.int64)
        np.maximum.at(best, i[ok] * size + j[ok], np.nonzero(ok)[0])
        best = best.reshape(size, size)
        valid = best >= 0
        img = np.zeros((size, size, 3), np.float32)
        img[valid] = self.rgb[best[valid]]
        img, valid = self._fill(img, valid)
        return np.clip(img, 0, 255).astype(np.uint8), valid


@dataclass
class CalibResult:
    camera: SatelliteCamera
    ncc: float
    lean_grid: np.ndarray        # (L,) coarse lean values, used for both axes
    coarse_ncc: np.ndarray       # (L, L): [index of lean_n, index of lean_e]
    second: dict                 # best coarse lean farther than second_min_dist from the best: {"ncc", "lean_e", "lean_n"}
    lean0: dict                  # lean = 0 with its own best b: {"ncc", "b_e", "b_n"}
    seconds: float
    renderer: object = field(repr=False, default=None)   # MeshEdgeRenderer or UavTextureRenderer

    def summary(self) -> dict:
        return {"camera": self.camera.to_dict(), "ncc": self.ncc, "second_peak": self.second, "lean0": self.lean0,
                "lean_grid": [float(x) for x in self.lean_grid], "coarse_ncc": self.coarse_ncc.tolist(),
                "seconds": self.seconds}


def calibrate_satellite(image_rgb, camera: SatelliteCamera, vertices, faces, frame: str = "enu",
                        cfg: CalibConfig | None = None) -> CalibResult:
    """Fit b and lean of `camera` (its width, height, m_per_px, centre and h0 are kept) to the satellite image, using
    only the mesh (vertices in `frame`). The camera's own b and lean are not used as a prior."""
    cfg = cfg or CalibConfig()
    t0 = time.time()
    V = to_enu(vertices, frame)
    n = int(np.clip(mesh_area(V, faces) * cfg.samples_per_m2, cfg.min_samples, cfg.max_samples))
    pts = sample_surface(V, faces, n, cfg.seed)
    rend = MeshEdgeRenderer(image_rgb, camera, pts, cfg)
    return _fit(rend, camera, cfg, t0)


def _fit(rend, camera: SatelliteCamera, cfg: CalibConfig, t0: float) -> CalibResult:
    """Coarse then fine lean search with rend.match; b of the best lean."""
    k = int(round(cfg.lean_max / cfg.lean_step))
    grid = np.round(np.arange(-k, k + 1) * cfg.lean_step, 6)
    coarse = np.full((len(grid), len(grid)), np.nan)
    for a, ln in enumerate(grid):
        for c, le in enumerate(grid):
            coarse[a, c] = rend.match(le, ln)[0]
    a, c = np.unravel_index(int(np.nanargmax(coarse)), coarse.shape)
    best_le, best_ln = float(grid[c]), float(grid[a])
    far = [(coarse[x, y], float(grid[y]), float(grid[x])) for x in range(len(grid)) for y in range(len(grid))
           if math.hypot(grid[y] - best_le, grid[x] - best_ln) > cfg.second_min_dist and np.isfinite(coarse[x, y])]
    sec = max(far) if far else (float("nan"), float("nan"), float("nan"))
    kf = int(round(cfg.fine_half / cfg.fine_step))
    best = None
    for dn in np.arange(-kf, kf + 1) * cfg.fine_step:
        for de in np.arange(-kf, kf + 1) * cfg.fine_step:
            le, ln = round(best_le + de, 6), round(best_ln + dn, 6)
            s, be, bn = rend.match(le, ln)
            if best is None or s > best[0]:
                best = (s, le, ln, be, bn)
    s0, be0, bn0 = rend.match(0.0, 0.0)
    cam = camera.with_params(lean_e=best[1], lean_n=best[2], b_e=best[3], b_n=best[4])
    return CalibResult(camera=cam, ncc=float(best[0]), lean_grid=grid, coarse_ncc=coarse,
                       second={"ncc": float(sec[0]), "lean_e": float(sec[1]), "lean_n": float(sec[2])},
                       lean0={"ncc": float(s0), "b_e": float(be0), "b_n": float(bn0)},
                       seconds=time.time() - t0, renderer=rend)


def calibrate_satellite_uav(image_rgb, camera: SatelliteCamera, points, rgb, frame: str = "enu",
                            cfg: UavCalibConfig | None = None, h0: float | None = None) -> CalibResult:
    """Fit b and lean of `camera` (width, height, m_per_px and centre are kept) to the satellite image using only
    coloured UAV points (uav.uav_points). h0 None: the datum is the cfg.ground_percentile-th percentile of the points'
    heights. The camera's own b and lean are not used as a prior."""
    cfg = cfg or UavCalibConfig()
    t0 = time.time()
    P = to_enu(points, frame)
    if h0 is None:
        h0 = float(np.percentile(P[:, 2], cfg.ground_percentile))
    camera = camera.with_params(h0=float(h0))
    rend = UavTextureRenderer(image_rgb, camera, P, rgb, cfg)
    return _fit(rend, camera, cfg, t0)


def score_camera(image_rgb, camera: SatelliteCamera, vertices, faces, frame: str = "enu",
                 cfg: CalibConfig | None = None) -> float:
    """Edge NCC of a given camera (no search) for a mesh, e.g. to check a camera on a building it was not fitted on."""
    cfg = cfg or CalibConfig()
    V = to_enu(vertices, frame)
    n = int(np.clip(mesh_area(V, faces) * cfg.samples_per_m2, cfg.min_samples, cfg.max_samples))
    return MeshEdgeRenderer(image_rgb, camera, sample_surface(V, faces, n, cfg.seed), cfg).score(camera)
