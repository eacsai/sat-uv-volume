import numpy as np

from sat_uv import (Cube, SatelliteCamera, Window, cube_around, occupancy, satellite_uv_volume, uv_volume_from_uv,
                    voxel_centres, voxel_points)

R = 16


def box_mesh(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def test_voxel_centres_layout():
    c = voxel_centres(R)
    assert c.shape == (3, R, R, R)
    assert np.isclose(c[0, 0, 0, 0], -0.5 + 0.5 / R) and np.isclose(c[0, -1, 0, 0], 0.5 - 0.5 / R)
    assert np.allclose(c[0, 3], c[0, 3, 0, 0]) and np.allclose(c[2, :, :, 5], c[2, 0, 0, 5])
    cube = Cube([10.0, 20.0, 30.0], 8.0)
    P = voxel_points(cube, R).reshape(R, R, R, 3)
    assert np.allclose(P[1, 2, 3], cube.from_cube(c[:, 1, 2, 3]))


def test_mask_before_clamp():
    u = np.array([0.5, 1.2, 0.995, -0.3, 0.02])
    v = np.array([0.5, 0.5, 0.5, 0.5, 0.98])
    n = R ** 3
    uu, vv = np.full(n, 0.5), np.full(n, 0.5)
    uu[:5], vv[:5] = u, v
    out = uv_volume_from_uv(uu, vv, None, R)
    m = out["ssuv"].reshape(-1)[:5]
    assert m.tolist() == [1, 0, 0, 0, 1]
    assert np.isclose(out["uv_volume"].reshape(2, -1)[0, 1], 1.0) and np.isclose(out["uv_raw"].reshape(2, -1)[0, 1], 1.2)
    assert np.isclose(out["uv_volume"].reshape(2, -1)[0, 3], 0.0)
    assert out["uv_volume"].dtype == np.float32 and out["ssuv"].shape == (1, R, R, R)


def test_uv_volume_layout_u_then_v():
    n = R ** 3
    out = uv_volume_from_uv(np.full(n, 0.2), np.full(n, 0.7), None, R)
    assert np.allclose(out["uv_volume"][0], 0.2) and np.allclose(out["uv_volume"][1], 0.7)


def test_satellite_volume_and_affine():
    cam = SatelliteCamera(width=1200, height=1200, m_per_px=0.3, centre_e=5.0, centre_n=-3.0, h0=-2.0,
                          b_e=1.5, b_n=-0.7, lean_e=-0.2, lean_n=0.6)
    cube = Cube([-4.0, 3.0, 10.0], 60.0)   # EDS
    win = Window(300, 250, 500)
    out = satellite_uv_volume(cam, win, cube, "eds", R)
    X = voxel_points(cube, R)
    col, row = cam.project(X, "eds")
    u, v = win.uv(col, row)
    assert np.allclose(out["uv_raw"][0].reshape(-1), u, atol=1e-5) and np.allclose(out["uv_raw"][1].reshape(-1), v, atol=1e-5)
    p = voxel_centres(R).reshape(3, -1)
    uv = out["camera_cube"] @ np.concatenate([p, np.ones((1, p.shape[1]))], 0)
    assert np.allclose(uv[0], u, atol=1e-9) and np.allclose(uv[1], v, atol=1e-9)


def test_cube_and_occupancy():
    V, F = box_mesh((0, 0, 0), (10, 4, 6))
    cube = cube_around(V, 1.1)
    assert np.allclose(cube.centre, [5, 2, 3]) and np.isclose(cube.side, 11.0)
    occ = occupancy(V, F, cube, R)
    assert len(occ) > 0 and (occ >= 0).all() and (occ < R).all()
    p = cube.from_cube((occ + 0.5) / R - 0.5)
    d = np.minimum.reduce([np.abs(p[:, 0] - 0), np.abs(p[:, 0] - 10), np.abs(p[:, 1] - 0), np.abs(p[:, 1] - 4),
                           np.abs(p[:, 2] - 0), np.abs(p[:, 2] - 6)])
    assert (d <= cube.side / R * 0.9).all()          # surface voxels only
