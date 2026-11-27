"""UAV texture calibration of the satellite camera: rebase, back-projection, masked NCC, calibration, pose loading."""
import os
import pickle

import cv2
import numpy as np
import pytest
from numpy.lib import format as npf

from sat_uv import SatelliteCamera, Window
from sat_uv.calibrate import UavCalibConfig, UavTextureRenderer, calibrate_satellite_uav, masked_ncc
from sat_uv.crossgeo import load_pose, uav_views
from sat_uv.geo import to_enu
from sat_uv.uav import UavView, back_project, uav_points

SEED = 1111
TRUE = dict(lean_e=-0.3, lean_n=0.5, b_e=2.0, b_n=-3.0)
BOXES = [(-20.0, -15.0, 20.0, 15.0, 60.0), (35.0, 10.0, 55.0, 30.0, 25.0)]   # E0, N0, E1, N1, height (ENU metres)
SHADE = {"e": 150.0, "w": 95.0, "n": 170.0, "s": 70.0}                         # facade brightness by facing
UAV_GROUND = (-45.0, -40.0, 75.0, 55.0)                                         # E0, N0, E1, N1 of the UAV ground
HOLE = (52.0, -35.0, 67.0, -20.0)                                               # 15 m x 15 m occluded: no UAV points
POSE_K = np.array([[500.0, 0.0, 480.0], [0.0, 500.0, 270.0], [0.0, 0.0, 1.0]])
POSE_C2W = np.array([[1.0, 0, 0, 1.5], [0, 1.0, 0, -20.0], [0, 0, 1.0, 3.0], [0, 0, 0, 1.0]])


def test_rebase_keeps_projection():
    P = np.random.default_rng(SEED).normal(scale=40.0, size=(200, 3)) + [10.0, -5.0, 30.0]
    cam = SatelliteCamera(width=800, height=600, m_per_px=0.4, centre_e=10.0, centre_n=-5.0, h0=12.0, **TRUE)
    reb = cam.rebase(-3.5)
    assert reb.h0 == -3.5 and cam.h0 == 12.0 and (reb.lean_e, reb.lean_n) == (cam.lean_e, cam.lean_n)
    assert np.allclose(reb.project_enu(P), cam.project_enu(P), rtol=0.0, atol=1e-9)
    assert np.isclose(reb.b_e, cam.b_e - 15.5 * cam.lean_e) and np.isclose(reb.b_n, cam.b_n - 15.5 * cam.lean_n)
    assert abs(reb.b_e - cam.b_e) > 1.0 and abs(reb.b_n - cam.b_n) > 1.0


def _view(depth, rgb=None, c2w=None, f=100.0):
    H, W = depth.shape
    if rgb is None:                                       # colour = pixel index: (row, col, 7)
        r, c = np.mgrid[0:H, 0:W]
        rgb = np.stack([r, c, np.full_like(r, 7)], -1).astype(np.uint8)
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])
    return UavView(depth=depth, K=K, c2w=np.eye(4) if c2w is None else c2w, rgb=rgb, name="test")


def test_back_project_plane_and_pixel_centre():
    H, W, f = 40, 60, 100.0
    depth = np.full((H, W), 50.0, np.float32)
    depth[0, 0], depth[0, 1], depth[0, 2] = np.nan, 0.0, 500.0         # NaN, too near, too far
    view = _view(depth)
    for centre in (0.0, 0.5):
        P, rgb = back_project(view, centre=centre)
        assert P.shape == (H * W - 3, 3) and rgb.shape == (H * W - 3, 3) and rgb.dtype == np.uint8
        r, c = rgb[:, 0].astype(np.float64), rgb[:, 1].astype(np.float64)
        assert not ((r == 0) & (c <= 2)).any()
        assert np.allclose(P[:, 2], 50.0)
        assert np.allclose(P[:, 0], (c + centre - W / 2) / f * 50.0)
        assert np.allclose(P[:, 1], (r + centre - H / 2) / f * 50.0)
        mid = P[(r == H // 2) & (c == W // 2)]                         # 0.5 px off centre -> 0.5 / f * 50 = 0.25 m
        assert len(mid) == 1 and np.allclose(mid[0], [0.5 * centre, 0.5 * centre, 50.0], rtol=0.0, atol=1e-9)
    assert len(back_project(view, max_depth=600.0)[0]) == H * W - 2
    a = np.radians(30.0)                                  # c2w is applied (match points by their pixel colour)
    c2w = np.eye(4)
    c2w[:3, :3] = [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    c2w[:3, 3] = [5.0, -2.0, 7.0]
    P0, rgb0 = back_project(view)
    P1, rgb1 = back_project(_view(depth, c2w=c2w))
    o0, o1 = np.lexsort((rgb0[:, 1], rgb0[:, 0])), np.lexsort((rgb1[:, 1], rgb1[:, 0]))
    assert np.allclose(P1[o1], P0[o0] @ c2w[:3, :3].T + c2w[:3, 3])
    big = np.zeros((2 * H, 2 * W, 3), np.uint8)           # rgb at twice the depth size is resized to the grid
    big[20:32, 40:56] = (200, 50, 30)                     # depth-grid rows 10..15, cols 20..27
    P, rgb = back_project(_view(np.full((H, W), 50.0, np.float32), rgb=big))
    r = np.rint(P[:, 1] / 50.0 * f + H / 2).astype(int)
    c = np.rint(P[:, 0] / 50.0 * f + W / 2).astype(int)
    core = (r >= 11) & (r < 15) & (c >= 21) & (c < 27)
    away = (r < 8) | (r >= 18) | (c < 18) | (c >= 30)
    assert core.sum() == 24 and np.abs(rgb[core].astype(int) - (200, 50, 30)).max() <= 2 and rgb[away].max() <= 2


def test_uav_points_concat_enu_and_crop():
    depth = np.full((30, 40), 20.0, np.float32)
    c2w = np.eye(4)
    c2w[:3, 3] = [3.0, -1.0, 12.0]
    views = [_view(depth), _view(depth * 1.5, c2w=c2w)]
    P, rgb = uav_points(views, frame="eds")
    (Pa, ra), (Pb, rb) = back_project(views[0]), back_project(views[1])
    assert np.allclose(P, to_enu(np.concatenate([Pa, Pb]), "eds")) and (rgb == np.concatenate([ra, rb])).all()
    # EDS z = East: view 0 lies at E = 20, view 1 at E = 42; N = -x is a multiple of 0.2 m (no point on the border)
    Pc, rc = uav_points(views, frame="eds", centre_en=(20.0, 0.1), radius_m=2.05)
    keep = (np.abs(P[:, 0] - 20.0) <= 2.05) & (np.abs(P[:, 1] - 0.1) <= 2.05)
    assert 0 < len(Pc) < len(Pa) and np.allclose(Pc, P[keep]) and (rc == rgb[keep]).all()


def _look(F, e, n, s=0.5, origin=-160.0):
    """Texture value at world (E, N): F[i, j] is the cell of N = origin + (i + .5) s, E = origin + (j + .5) s."""
    i = np.clip(((np.asarray(n) - origin) / s).astype(np.int64), 0, F.shape[0] - 1)
    j = np.clip(((np.asarray(e) - origin) / s).astype(np.int64), 0, F.shape[1] - 1)
    return F[i, j].astype(np.float64)


def _textures(rng, n=640, s=0.5):
    """World-anchored textures: ground (smooth noise + bright stripes) and roofs; facades reuse the roof field."""
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * s - 160.0         # y = North, x = East of the cell centres

    def noise(sigma_m):
        f = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), sigma_m / s)
        return (f - f.mean()) / f.std()
    ground = 115.0 + 28.0 * noise(2.0)
    for a, o in zip(rng.uniform(0, np.pi, 4), rng.uniform(-40.0, 40.0, 4)):   # stripes 3 m wide
        ground[np.abs(np.cos(a) * x + np.sin(a) * y - o) < 1.5] = 235.0
    ground = cv2.GaussianBlur(ground.astype(np.float32), (0, 0), 2.0)
    return ground, (200.0 + 22.0 * noise(1.5)).astype(np.float32)


def _ground(tex, e0, n0, e1, n1, step, hole=None):
    e, n = np.meshgrid(np.arange(e0 + step / 2, e1, step), np.arange(n0 + step / 2, n1, step))
    e, n = e.ravel(), n.ravel()
    keep = np.ones(e.size, bool)
    for be0, bn0, be1, bn1, _ in BOXES:                   # no ground inside the buildings
        keep &= ~((e > be0) & (e < be1) & (n > bn0) & (n < bn1))
    if hole is not None:
        keep &= ~((e > hole[0]) & (e < hole[2]) & (n > hole[1]) & (n < hole[3]))
    e, n = e[keep], n[keep]
    return np.stack([e, n, np.zeros_like(e)], 1), _look(tex[0], e, n)


def _buildings(tex, step):
    P, L = [], []
    for e0, n0, e1, n1, h in BOXES:
        e, n = np.meshgrid(np.arange(e0 + step / 2, e1, step), np.arange(n0 + step / 2, n1, step))
        P.append(np.stack([e.ravel(), n.ravel(), np.full(e.size, h)], 1))
        L.append(_look(tex[1], e.ravel(), n.ravel()))
        for side, shade in SHADE.items():
            s0, s1 = (n0, n1) if side in "ew" else (e0, e1)
            s, u = np.meshgrid(np.arange(s0 + step / 2, s1, step), np.arange(step / 2, h, step))
            s, u = s.ravel(), u.ravel()
            fixed = np.full(s.size, {"e": e1, "w": e0, "n": n1, "s": n0}[side])
            P.append(np.stack([fixed, s, u] if side in "ew" else [s, fixed, u], 1))
            L.append(shade + 0.8 * (_look(tex[1], s, u - 100.0) - 200.0))
    return np.concatenate(P), np.concatenate(L)


def _satellite(cam, tex, rng):
    """Every surface projected with the true camera, highest point wins per pixel, holes filled."""
    W, H, half = cam.width, cam.height, 0.5 * cam.width * cam.m_per_px + 15.0
    Pg, Lg = _ground(tex, cam.centre_e - half, cam.centre_n - half, cam.centre_e + half, cam.centre_n + half, 0.25)
    Pb, Lb = _buildings(tex, 0.2)
    P, L = np.concatenate([Pg, Pb]), np.concatenate([Lg, Lb])
    col, row = cam.project_enu(P)
    i, j = np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    key, h, L = (i * W + j)[ok], P[ok, 2], L[ok]
    order = np.lexsort((h, key))                          # by pixel, then height: the last of each pixel is the highest
    key = key[order]
    last = np.r_[key[1:] != key[:-1], True]
    img = np.full(H * W, np.nan, np.float32)
    img[key[last]] = L[order][last]
    img = img.reshape(H, W)
    img[np.isnan(img)] = np.nanmean(img)
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 3, (H, W)).astype(np.float32)
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, -1)


def _uav_cloud(tex, rng):
    """Sparser (0.5 m) jittered (+-0.15 m) sample of the visible surfaces with their colours, one ground patch missing."""
    Pg, Lg = _ground(tex, *UAV_GROUND, 0.5, HOLE)
    Pb, Lb = _buildings(tex, 0.5)
    P = np.concatenate([Pg, Pb])
    P = P + rng.uniform(-0.15, 0.15, P.shape)
    L = np.clip(np.concatenate([Lg, Lb]) + rng.normal(0, 3, len(P)), 0, 255).astype(np.uint8)
    return P, np.repeat(L[:, None], 3, 1)


def test_uav_calibration_recovers_lean_and_b():
    rng = np.random.default_rng(SEED)
    tex = _textures(rng)
    base = SatelliteCamera(width=500, height=500, m_per_px=0.5, centre_e=15.0, centre_n=8.0, h0=0.0)
    truth = base.with_params(**TRUE)
    image = _satellite(truth, tex, rng)
    P, rgb = _uav_cloud(tex, rng)
    cfg = UavCalibConfig(lean_max=0.6, lean_step=0.1, fine_half=0.1, fine_step=0.02)
    junk = base.with_params(lean_e=0.2, lean_n=-0.2, b_e=-5.0, b_n=4.0, h0=7.0)   # not a prior: must be ignored
    res = calibrate_satellite_uav(image, junk, P, rgb, frame="enu", cfg=cfg, h0=0.0)
    c = res.camera
    assert c.h0 == 0.0
    assert abs(c.lean_e - TRUE["lean_e"]) <= 0.05 and abs(c.lean_n - TRUE["lean_n"]) <= 0.05
    assert abs(c.b_e - TRUE["b_e"]) <= 0.75 and abs(c.b_n - TRUE["b_n"]) <= 0.75
    assert res.ncc > res.lean0["ncc"] + 0.05
    cfg2 = UavCalibConfig(lean_max=0.1, lean_step=0.1, fine_half=0.02, fine_step=0.02)
    r2 = calibrate_satellite_uav(image, base, P, rgb, cfg=cfg2)                  # h0 None -> ground percentile
    assert np.isclose(r2.camera.h0, np.percentile(P[:, 2], cfg2.ground_percentile), rtol=0.0, atol=1e-3)
    rend = UavTextureRenderer(image, base, P, rgb, cfg)
    off = truth.with_params(b_e=TRUE["b_e"] + 6.0)
    assert rend.score(truth) > rend.score(off) + 0.05
    win, size = Window(150, 170, 180), 90                 # ground E -37..53, N -39..51: all inside the UAV data
    pred, valid = rend.predicted_rgb(truth, win, size)
    assert pred.shape == (size, size, 3) and pred.dtype == np.uint8 and valid.shape == (size, size)
    assert valid.dtype == bool and valid.mean() > 0.3
    sat = cv2.resize(win.crop(image), (size, size), interpolation=cv2.INTER_AREA)[..., 0].astype(np.float64)

    def corr(p, v):
        return np.corrcoef(p[..., 0][v].astype(np.float64), sat[v])[0, 1]
    good = corr(pred, valid)
    assert good > 0.5 and good > corr(*rend.predicted_rgb(off, win, size)) + 0.1


def test_masked_ncc_ignores_masked_pixels():
    rng = np.random.default_rng(SEED)
    I = cv2.GaussianBlur(rng.normal(0, 1, (90, 110)).astype(np.float32), (0, 0), 2.0)
    T = I[25:55, 40:80].copy()
    M = np.ones(T.shape, np.float32)
    bad = T.copy()
    bad[:12, :18] = rng.uniform(-50.0, 50.0, (12, 18))  # garbage, masked out
    M[:12, :18] = 0.0
    S = masked_ncc(I, bad, M)
    assert S.shape == (61, 71)
    a, b = np.unravel_index(int(np.nanargmax(S)), S.shape)
    assert (a, b) == (25, 40) and S[25, 40] > 0.99
    assert masked_ncc(I, bad, np.ones_like(M))[25, 40] < 0.9                     # the mask is what saves it
    ref = cv2.matchTemplate(I, T, cv2.TM_CCOEFF_NORMED)
    assert np.allclose(masked_ncc(I, T, np.ones_like(M)), ref, rtol=0.0, atol=1e-3)


_CALLS = []


def _mark(*args):                                         # harmless stand-in for os.system; records any call
    _CALLS.append(args)
    return 0


class _Evil:
    def __init__(self, fn, args):
        self.fn, self.args = fn, args

    def __reduce__(self):
        return self.fn, self.args


def _pose(K, c2w, **extra):
    d = {"intrinsics": K, "c2w": c2w}
    d.update(extra)
    return np.array(d, dtype=object)                      # 0-d object array, as in the CrossGeo pose files


def test_load_pose_roundtrip_and_refuses_evil(tmp_path):
    path = str(tmp_path / "uav_1_rgb.npy")
    np.save(path, _pose(POSE_K.astype(np.float32), POSE_C2W, raw_data={"note": "x", "v": [1, 2]}), allow_pickle=True)
    K2, c2w2 = load_pose(path)
    assert K2.dtype == np.float64 and c2w2.dtype == np.float64 and K2.shape == (3, 3) and c2w2.shape == (4, 4)
    assert np.allclose(K2, POSE_K) and np.allclose(c2w2, POSE_C2W)
    for k, (fn, args) in enumerate([(_mark, ("pwned",)), (os.getcwd, ())]):
        raw, nested = str(tmp_path / ("raw%d.npy" % k)), str(tmp_path / ("nested%d.npy" % k))
        with open(raw, "wb") as f:                        # valid object-array header, then a bare evil payload
            npf.write_array_header_1_0(f, npf.header_data_from_array_1_0(np.empty((), object)))
            pickle.dump(_Evil(fn, args), f, protocol=3)
        np.save(nested, np.array(_Evil(fn, args), dtype=object), allow_pickle=True)   # inside a genuine array
        for p in (raw, nested):
            with pytest.raises(pickle.UnpicklingError):
                load_pose(p)
    assert _CALLS == []


def test_uav_views_reads_pair(tmp_path):
    for k, name in enumerate(("uav_1", "uav_2")):
        np.save(str(tmp_path / (name + "_depth.npy")), np.full((540, 960), 30.0 + k, np.float32))
        np.save(str(tmp_path / (name + "_rgb.npy")), _pose(POSE_K, POSE_C2W), allow_pickle=True)
        img = np.zeros((540, 960, 3), np.uint8)
        img[200:340, 400:560] = (0, 0, 255)               # BGR red in the middle
        cv2.imwrite(str(tmp_path / (name + "_rgb.jpg")), img)
    views = uav_views(str(tmp_path))
    assert len(views) == 2
    for k, v in enumerate(views):
        assert v.depth.shape == (540, 960) and np.allclose(v.depth, 30.0 + k)
        assert np.allclose(v.K, POSE_K) and np.allclose(v.c2w, POSE_C2W)
        h, w = v.rgb.shape[:2]
        assert v.rgb.shape[2] == 3 and v.rgb[h // 2, w // 2, 0] > 200 and v.rgb[h // 2, w // 2, 2] < 60   # RGB



# --- datum invariance of the scores and the hole mask of the UAV template (added by the parent) ---------------------

def _noise_image(n, seed=1111):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 3)
    img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)
    return np.repeat(img[..., None], 3, 2)


def test_scores_do_not_depend_on_the_datum():
    from sat_uv import MeshEdgeRenderer, UavTextureRenderer, UavCalibConfig, CalibConfig, sample_surface
    img = _noise_image(300)
    cam = SatelliteCamera(width=300, height=300, m_per_px=0.5, centre_e=0.0, centre_n=0.0, h0=0.0,
                          b_e=1.0, b_n=-2.0, lean_e=0.3, lean_n=-0.2)
    rng = np.random.default_rng(1111)
    P = np.column_stack([rng.uniform(-40, 40, 20000), rng.uniform(-40, 40, 20000), rng.uniform(0, 30, 20000)])
    col, row = cam.project_enu(P)
    rgb = img[np.clip(row.astype(int), 0, 299), np.clip(col.astype(int), 0, 299)]
    ru = UavTextureRenderer(img, cam, P, rgb, UavCalibConfig())
    moved = cam.rebase(cam.h0 + 10.0)                           # same camera, other datum
    wrong = cam.with_params(h0=cam.h0 + 10.0)                   # other camera (3 m / 2 m shift)
    assert np.isclose(ru.score(moved), ru.score(cam), atol=1e-6)
    assert ru.score(wrong) < ru.score(cam) - 0.02
    V = np.array([[x, y, z] for x in (-20, 20) for y in (-15, 15) for z in (0, 40)], float)
    F = np.array([[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3], [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]])
    rm = MeshEdgeRenderer(img, cam, sample_surface(V, F, 50000, 1111), CalibConfig())
    assert np.isclose(rm.score(moved), rm.score(cam), atol=1e-6)
    assert not np.isclose(rm.score(wrong), rm.score(cam), atol=1e-3)
    c1, r1 = rm.edge_pixels(cam)
    c2, r2 = rm.edge_pixels(moved)
    assert np.allclose(c1, c2) and np.allclose(r1, r2)


def test_uav_template_does_not_score_holes():
    from sat_uv import UavTextureRenderer, UavCalibConfig
    img = _noise_image(200)
    cam = SatelliteCamera(width=200, height=200, m_per_px=0.5, centre_e=0.0, centre_n=0.0)
    g = np.arange(-30, 30, 0.25)
    E, N = np.meshgrid(g, g)
    keep = ~((np.abs(E - 5) < 6) & (np.abs(N + 4) < 6))       # 12 m x 12 m hole centred at (5, -4)
    P = np.column_stack([E[keep], N[keep], np.zeros(keep.sum())])
    rgb = np.full((len(P), 3), 128, np.uint8)
    ru = UavTextureRenderer(img, cam, P, rgb, UavCalibConfig())
    T, M, i0, j0 = ru.template(0.0, 0.0)
    # cell (a, c) of the template is working-grid pixel (i0 + a, j0 + c); 1 m cells, image centre = world (0, 0)
    e_c = (np.arange(M.shape[1]) + j0 + 0.5) * ru.mx - 100 * 0.5 * 1.0
    n_c = -((np.arange(M.shape[0]) + i0 + 0.5) * ru.my - 100 * 0.5 * 1.0)
    Ee, Nn = np.meshgrid(e_c, n_c)
    in_hole = (np.abs(Ee - 5) < 6) & (np.abs(Nn + 4) < 6)       # the hole (its edge cells are filled, then eroded)
    assert not M[in_hole].any()
    inner = (np.abs(Ee) < 25) & (np.abs(Nn) < 25) & ~((np.abs(Ee - 5) < 10) & (np.abs(Nn + 4) < 10))
    assert M[inner].all()
