#!/usr/bin/env python
"""Ground-truth satellite UV volume of one building, from its mesh, for the satellite PNG(s) of one CrossGeo pair.

    python examples/crossgeo_satellite.py --pair-dir PAIR_DIR --mesh building.ply --out OUT [--sat-sides 1 2] [--figures]

The mesh must be in the flight's world frame (CrossGeo "EDS": x South, y Down, z East, metres), the frame of the pair's
poses. Writes, under OUT:
  cube.json, occupancy.npy                     the cube around the mesh and its occupied voxels (M, 3)
  satellite_<k>.npz                            the UV volume (uv_volume, ssuv, uv_raw, camera_cube)
  cameras.json                                 the satellite camera(s) (world and cube coordinates)
  satellite_<k>_crop.png                       the satellite window the satellite UV volume refers to
  satellite_<k>_calibration.json               calibration result (camera, NCC, second peak, lean = 0 score, grid)
  figures/*.png                                with --figures
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sat_uv import (CalibConfig, calibrate_satellite, cube_around, occupancy, satellite_uv_volume,  # noqa: E402
                    voxel_points, window_around)
from sat_uv import crossgeo, viz  # noqa: E402


def load_mesh(path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.triangles, np.int64)
    if len(V) == 0 or len(F) == 0:
        raise SystemExit("empty mesh: %s" % path)
    return V, F


def save_npz(path, vol):
    np.savez_compressed(path, **{k: v for k, v in vol.items() if isinstance(v, np.ndarray)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-dir", required=True)
    ap.add_argument("--mesh", required=True, help="building mesh in the flight's EDS frame (ply / obj / glb)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sat-sides", type=int, nargs="*", default=[1, 2], help="satellite PNGs to calibrate (1 and/or 2)")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--pad", type=float, default=1.1, help="cube side = pad * largest mesh extent")
    ap.add_argument("--zoom", type=int, default=19, help="Web-Mercator zoom level of the satellite PNGs")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    fig_dir = os.path.join(a.out, "figures")
    if a.figures:
        os.makedirs(fig_dir, exist_ok=True)

    V, F = load_mesh(a.mesh)
    cube = cube_around(V, a.pad)
    occ = occupancy(V, F, cube, a.res)
    np.save(os.path.join(a.out, "occupancy.npy"), occ)
    h0 = float(np.percentile(-V[:, 1], 1.0))                      # mesh base, world "up" = -y in EDS
    occ_up = -voxel_points(cube, a.res).reshape(a.res, a.res, a.res, 3)[occ[:, 0], occ[:, 1], occ[:, 2], 1]
    occ_h = occ_up - h0
    vmax = float(np.percentile(occ_h, 99)) if len(occ_h) else 1.0
    with open(os.path.join(a.out, "cube.json"), "w") as f:
        json.dump({"cube": cube.to_dict(), "res": a.res, "frame": "eds", "h0_up_m": h0, "n_occupied": int(len(occ)),
                   "axes": "axis 0 = x (South), axis 1 = y (Down), axis 2 = z (East)"}, f, indent=1)
    print("cube side %.1f m, voxel %.2f m, %d occupied voxels" % (cube.side, cube.side / a.res, len(occ)))
    cameras = {}

    for side in a.sat_sides:
        img, cam0, info = crossgeo.satellite_camera(a.pair_dir, side, zoom=a.zoom)
        cam0 = cam0.with_params(h0=h0)
        res = calibrate_satellite(img, cam0, V, F, frame="eds", cfg=CalibConfig())
        cam = res.camera
        win = window_around(cam, V, "eds")
        if not win.inside(cam.width, cam.height):
            print("satellite_%d: the building's window leaves the image; the crop is zero-padded" % side)
        vol = satellite_uv_volume(cam, win, cube, "eds", a.res)
        name = "satellite_%d" % side
        save_npz(os.path.join(a.out, name + ".npz"), vol)
        crop = win.crop(img)
        cv2.imwrite(os.path.join(a.out, name + "_crop.png"), crop[..., ::-1])
        summ = res.summary()
        summ.update({"window": win.to_dict(), "image": info, "camera_cube_2x4": vol["camera_cube"].tolist()})
        with open(os.path.join(a.out, name + "_calibration.json"), "w") as f:
            json.dump(summ, f, indent=1)
        cameras[name] = {"kind": "satellite", "camera": cam.to_dict(), "window": win.to_dict(),
                         "camera_cube_2x4": vol["camera_cube"].tolist()}
        print("%s: lean (E %.3f, N %.3f), b (E %.2f, N %.2f) m, NCC %.3f; lean 0: NCC %.3f; second peak %.3f at (%.2f, %.2f)"
              % (name, cam.lean_e, cam.lean_n, cam.b_e, cam.b_n, res.ncc, res.lean0["ncc"], res.second["ncc"],
                 res.second["lean_e"], res.second["lean_n"]))
        if a.figures:
            S = 640
            base = cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC)
            flat = cam.with_params(lean_e=0.0, lean_n=0.0, b_e=res.lean0["b_e"], b_n=res.lean0["b_n"])
            vol0 = satellite_uv_volume(flat, win, cube, "eds", a.res)
            panels = [viz.label(base.copy(), "satellite window (north up)")]
            for c_, t_ in ((flat, "lean = 0, b fitted: NCC %.3f" % res.lean0["ncc"]),
                           (cam, "calibrated lean (%.2f E, %.2f N): NCC %.3f" % (cam.lean_e, cam.lean_n, res.ncc))):
                col, row = res.renderer.edge_pixels(c_)
                panels.append(viz.label(viz.edge_overlay(base, win, col, row), t_))
            cv2.imwrite(os.path.join(fig_dir, name + "_1_mesh_edges.png"), viz.hstack(panels)[..., ::-1])
            r = max(1, int(round(cube.side / a.res / cam.m_per_px * S / win.size / 2)))
            panels = [viz.label(base.copy(), "satellite window (north up)")]
            for v_, t_ in ((vol0, "voxels, lean = 0"), (vol, "voxels, calibrated")):
                panels.append(viz.label(viz.voxel_overlay(base, v_["uv_raw"], occ, occ_h, 0, vmax, r), t_))
            cv2.imwrite(os.path.join(fig_dir, name + "_2_voxels.png"), viz.hstack(panels)[..., ::-1])
            cv2.imwrite(os.path.join(fig_dir, name + "_3_score_map.png"), viz.score_map(
                res, marks=[(cam.lean_e, cam.lean_n, (255, 255, 255)), (0.0, 0.0, (0, 0, 0))])[..., ::-1])
            panels, k = viz.uv_slice(vol, occ, axis=2, names=("x South", "y Down"))
            p0, _ = viz.uv_slice(vol0, occ, axis=2, index=k, names=("x South", "y Down"), prefix="lean 0: ")
            panels.append(p0[2])
            cv2.imwrite(os.path.join(fig_dir, name + "_4_uv_slice.png"),
                        viz.vstack([np.concatenate(panels[:2], 1), np.concatenate(panels[2:], 1)])[..., ::-1])

    with open(os.path.join(a.out, "cameras.json"), "w") as f:
        json.dump(cameras, f, indent=1)


if __name__ == "__main__":
    main()
