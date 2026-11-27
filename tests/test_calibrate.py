import cv2
import numpy as np

from sat_uv import CalibConfig, SatelliteCamera, calibrate_satellite, sample_surface, score_camera

TRUE = dict(lean_e=-0.3, lean_n=0.5, b_e=2.0, b_n=-3.0)


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def scene():
    """Two boxes in ENU (E, N, U), ground at U = 0."""
    V1, F1 = box((-20, -15, 0), (20, 15, 60))
    V2, F2 = box((35, 10, 0), (55, 30, 25))
    return np.concatenate([V1, V2]), np.concatenate([F1, F2 + 8])


def render(cam, V, F, seed=1111):
    """Synthetic satellite image: textured ground, roofs and walls shaded by orientation, highest point wins."""
    rng = np.random.default_rng(seed)
    H, W = cam.height, cam.width
    ground = cv2.GaussianBlur(rng.normal(120, 40, (H, W)).astype(np.float32), (0, 0), 3.0)
    img = ground.copy()
    shade = {(0, 0, 1): 215.0, (1, 0, 0): 150.0, (-1, 0, 0): 95.0, (0, 1, 0): 170.0, (0, -1, 0): 70.0}
    pts, val = [], []
    T = V[F]
    for t in T:
        nrm = np.cross(t[1] - t[0], t[2] - t[0])
        nrm = np.round(nrm / np.linalg.norm(nrm)).astype(int)
        if tuple(nrm) == (0, 0, -1):
            continue
        p = sample_surface(t, np.array([[0, 1, 2]]), 60000, seed)
        pts.append(p)
        val.append(np.full(len(p), shade[tuple(nrm)]))
    P, I = np.concatenate(pts), np.concatenate(val)
    col, row = cam.project_enu(P)
    i, j = np.floor(row).astype(int), np.floor(col).astype(int)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    i, j, h, I = i[ok], j[ok], P[ok, 2], I[ok]
    order = np.lexsort((h, i * W + j))           # by pixel, then height: the last of each pixel is the highest
    key = (i * W + j)[order]
    last = np.r_[key[1:] != key[:-1], True]
    img.reshape(-1)[key[last]] = I[order][last]
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 4, (H, W)).astype(np.float32)
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, -1)


def test_calibration_recovers_lean_and_offset():
    V, F = scene()
    base = SatelliteCamera(width=640, height=640, m_per_px=0.5, centre_e=10.0, centre_n=5.0, h0=0.0)
    image = render(base.with_params(**TRUE), V, F)
    cfg = CalibConfig(min_samples=120_000, max_samples=120_000)
    res = calibrate_satellite(image, base, V, F, "enu", cfg)
    c = res.camera
    assert abs(c.lean_e - TRUE["lean_e"]) <= 0.05 and abs(c.lean_n - TRUE["lean_n"]) <= 0.05
    assert abs(c.b_e - TRUE["b_e"]) <= 0.75 and abs(c.b_n - TRUE["b_n"]) <= 0.75
    assert res.ncc > res.lean0["ncc"] + 0.05
    assert res.coarse_ncc.shape == (21, 21)
    good = score_camera(image, c, V, F, "enu", cfg)
    bad = score_camera(image, c.with_params(b_e=c.b_e + 6.0), V, F, "enu", cfg)
    assert good > bad + 0.05


def test_frame_eds_equivalent():
    V, F = scene()
    base = SatelliteCamera(width=640, height=640, m_per_px=0.5, centre_e=10.0, centre_n=5.0, h0=0.0)
    image = render(base.with_params(**TRUE), V, F)
    cfg = CalibConfig(min_samples=60_000, max_samples=60_000, lean_max=0.6)
    Veds = np.stack([-V[:, 1], -V[:, 2], V[:, 0]], 1)        # ENU -> EDS
    a = calibrate_satellite(image, base, V, F, "enu", cfg).camera
    b = calibrate_satellite(image, base, Veds, F, "eds", cfg).camera
    assert np.allclose([a.lean_e, a.lean_n, a.b_e, a.b_n], [b.lean_e, b.lean_n, b.b_e, b.b_n], atol=1e-6)
