#!/usr/bin/env python
"""Ground-truth satellite UV volume of one building for the satellite PNG(s) of one CrossGeo pair, with the satellite
camera calibrated from the building mesh (--method mesh) or from the pair's own two UAV views (--method uav).

    python examples/crossgeo_satellite.py --pair-dir PAIR_DIR --mesh building.ply --out OUT [--sat-sides 1 2]
        [--method mesh|uav] [--compare] [--figures]

The mesh must be in the flight's world frame (CrossGeo "EDS": x South, y Down, z East, metres), the frame of the pair's
poses; it defines the cube (and, for --method uav, only the centre of the +-uav-radius window of UAV points).
--compare runs both methods and writes satellite_<k>_compare.json (and compare figures). Writes, under OUT:
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

from sat_uv import (CalibConfig, UavCalibConfig, calibrate_satellite, calibrate_satellite_uav,  # noqa: E402
                    cube_around, occupancy, satellite_uv_volume, to_enu, uav_points, voxel_points, window_around)
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


def compare(res_m, res_u) -> dict:
    """Mesh camera vs UAV camera: parameters, where they put the building (datum-independent), cross scores."""
    cm = res_m.camera
    cu = res_u.camera.rebase(cm.h0)
    P = res_m.renderer.P                                       # points on the mesh surface, ENU
    h = P[:, 2] - cm.h0
    c1, r1 = cm.project_enu(P)
    c2, r2 = cu.project_enu(P)
    d = np.hypot(c1 - c2, r1 - r2) * cm.m_per_px
    roof, base = h >= np.percentile(h, 98), h <= np.percentile(h, 2) + 3.0
    cam_d = lambda r: {"camera": r.camera.to_dict(), "ncc": r.ncc, "lean0": r.lean0, "second_peak": r.second,
                       "seconds": r.seconds}
    return {"mesh": cam_d(res_m), "uav": cam_d(res_u),
            "lean_diff": [cu.lean_e - cm.lean_e, cu.lean_n - cm.lean_n],
            "b_diff_at_mesh_datum": [cu.b_e - cm.b_e, cu.b_n - cm.b_n],
            "position_gap_m": {"roof_median": float(np.median(d[roof])), "roof_height_m": float(np.median(h[roof])),
                               "base_median": float(np.median(d[base])), "all_median": float(np.median(d)),
                               "max": float(d.max()),
                               "how": "distance in the satellite image (metres) between where the two cameras put the "
                                      "same mesh surface points; roof = top 2% of heights, base = lowest 3 m"},
            "cross_ncc": {"mesh_edges": {"mesh_camera": res_m.renderer.score(cm),
                                         "uav_camera": res_m.renderer.score(res_u.camera)},
                          "uav_texture": {"uav_camera": res_u.renderer.score(res_u.camera),
                                          "mesh_camera": res_u.renderer.score(cm)}}}


def compare_figures(fig_dir, name, img, V, res_m, res_u, S=640):
    cm, cu = res_m.camera, res_u.camera
    win = window_around(cm, V, "eds")
    base = cv2.resize(win.crop(img), (S, S), interpolation=cv2.INTER_CUBIC)
    G, M = (0, 255, 0), (255, 0, 255)
    em, eu = res_m.renderer.edge_pixels(cm), res_m.renderer.edge_pixels(cu)
    both = viz.edge_overlay(viz.edge_overlay(base, win, *em, rgb=G), win, *eu, rgb=M)
    panels = [viz.label(base.copy(), "satellite window (north up)"),
              viz.label(viz.edge_overlay(base, win, *em, rgb=G), "A: mesh (dense flight UAVs), lean (%.2f, %.2f)"
                        % (cm.lean_e, cm.lean_n), 24, 0.5),
              viz.label(viz.edge_overlay(base, win, *eu, rgb=M), "B: pair's 2 UAVs, lean (%.2f, %.2f)"
                        % (cu.lean_e, cu.lean_n), 24, 0.5),
              viz.label(both, "mesh edges: green = A, magenta = B", 24, 0.5)]
    cv2.imwrite(os.path.join(fig_dir, name + "_compare_1_edges.png"), viz.hstack(panels)[..., ::-1])
    n_work = max(16, int(round(win.size * cm.m_per_px / res_u.renderer.cfg.work_m_per_px)))
    rows = []
    for c_, t_ in ((cm, "A"), (cu, "B")):
        pred, ok = res_u.renderer.predicted_rgb(c_, win, n_work)  # at the working resolution, then enlarged
        pred = cv2.resize(pred, (S, S), interpolation=cv2.INTER_NEAREST)
        ok = cv2.resize(ok.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
        raw = np.where(ok[..., None], pred, 60).astype(np.uint8)
        mix = np.where(ok[..., None], 0.5 * base + 0.5 * pred, base).astype(np.uint8)
        rows.append(viz.hstack([viz.label(base.copy(), "satellite window (north up)"),
                                viz.label(raw, "pair's 2 UAV points through camera %s" % t_, 24, 0.5),
                                viz.label(mix, "camera %s: blend with satellite" % t_, 24, 0.5)]))
    cv2.imwrite(os.path.join(fig_dir, name + "_compare_2_uav_render.png"), viz.vstack(rows)[..., ::-1])
    marks = [(cm.lean_e, cm.lean_n, (0, 255, 0)), (cu.lean_e, cu.lean_n, (255, 0, 255)), (0.0, 0.0, (0, 0, 0))]
    cv2.imwrite(os.path.join(fig_dir, name + "_compare_3_score_maps.png"), viz.hstack([
        viz.score_map(res_m, marks=marks, title="A: mesh edge NCC (green A, magenta B)"),
        viz.score_map(res_u, marks=marks, title="B: 2-UAV texture NCC (green A, magenta B)")])[..., ::-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-dir", required=True)
    ap.add_argument("--mesh", required=True, help="building mesh in the flight's EDS frame (ply / obj / glb)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sat-sides", type=int, nargs="*", default=[1, 2], help="satellite PNGs to calibrate (1 and/or 2)")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--pad", type=float, default=1.1, help="cube side = pad * largest mesh extent")
    ap.add_argument("--zoom", type=int, default=19, help="Web-Mercator zoom level of the satellite PNGs")
    ap.add_argument("--method", choices=("mesh", "uav"), default="mesh", help="camera used for the UV volume")
    ap.add_argument("--compare", action="store_true", help="run both methods and compare them")
    ap.add_argument("--uav-names", nargs="*", default=["uav_1", "uav_2"], help="UAV views of the pair to use")
    ap.add_argument("--uav-radius", type=float, default=80.0, help="UAV points within +-R m (E, N) of the cube centre")
    ap.add_argument("--uav-pixel-centre", type=float, default=0.0, help="ray offset of a UAV depth pixel (CrossGeo: 0)")
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
    use_mesh, use_uav = a.method == "mesh" or a.compare, a.method == "uav" or a.compare
    if use_uav:
        ce, cn = to_enu(cube.centre[None], "eds")[0][:2]
        Pu, Cu = uav_points(crossgeo.uav_views(a.pair_dir, a.uav_names), "eds", (ce, cn), a.uav_radius,
                            centre=a.uav_pixel_centre)
        print("UAV points: %d from %s within +-%.0f m of the cube centre" % (len(Pu), "+".join(a.uav_names), a.uav_radius))

    for side in a.sat_sides:
        img, cam0, info = crossgeo.satellite_camera(a.pair_dir, side, zoom=a.zoom)
        cam0 = cam0.with_params(h0=h0)
        res_m = calibrate_satellite(img, cam0, V, F, frame="eds", cfg=CalibConfig()) if use_mesh else None
        res_u = calibrate_satellite_uav(img, cam0, Pu, Cu, frame="enu", cfg=UavCalibConfig()) if use_uav else None
        res = res_m if a.method == "mesh" else res_u
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
        summ.update({"method": a.method, "window": win.to_dict(), "image": info,
                     "camera_cube_2x4": vol["camera_cube"].tolist()})
        with open(os.path.join(a.out, name + "_calibration.json"), "w") as f:
            json.dump(summ, f, indent=1)
        cameras[name] = {"kind": "satellite", "camera": cam.to_dict(), "window": win.to_dict(),
                         "camera_cube_2x4": vol["camera_cube"].tolist()}
        for tag, r_ in (("mesh", res_m), ("uav", res_u)):
            if r_ is None:
                continue
            c_ = r_.camera
            print("%s [%s]: lean (E %.3f, N %.3f), b (E %.2f, N %.2f) m at h0 %.1f, NCC %.3f; lean 0: NCC %.3f; "
                  "second peak %.3f at (%.2f, %.2f); %.0f s" % (name, tag, c_.lean_e, c_.lean_n, c_.b_e, c_.b_n, c_.h0,
                                                              r_.ncc, r_.lean0["ncc"], r_.second["ncc"],
                                                              r_.second["lean_e"], r_.second["lean_n"], r_.seconds))
        if a.compare:
            cmp_ = compare(res_m, res_u)
            cmp_["uav"]["n_points"], cmp_["uav"]["radius_m"] = int(len(Pu)), a.uav_radius
            with open(os.path.join(a.out, name + "_compare.json"), "w") as f:
                json.dump(cmp_, f, indent=1)
            gp = cmp_["position_gap_m"]
            print("%s compare: lean diff (E %.3f, N %.3f); position gap roof %.2f m, base %.2f m (median); cross NCC: "
                  "mesh edges %.3f (mesh cam) vs %.3f (UAV cam), UAV texture %.3f (UAV cam) vs %.3f (mesh cam)"
                  % (name, cmp_["lean_diff"][0], cmp_["lean_diff"][1], gp["roof_median"], gp["base_median"],
                     cmp_["cross_ncc"]["mesh_edges"]["mesh_camera"], cmp_["cross_ncc"]["mesh_edges"]["uav_camera"],
                     cmp_["cross_ncc"]["uav_texture"]["uav_camera"], cmp_["cross_ncc"]["uav_texture"]["mesh_camera"]))
            if a.figures:
                compare_figures(fig_dir, name, img, V, res_m, res_u)
        if a.figures and res_m is not None:
            res = res_m
            cam = res.camera
            win = window_around(cam, V, "eds")
            crop = win.crop(img)
            vol = satellite_uv_volume(cam, win, cube, "eds", a.res)
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
