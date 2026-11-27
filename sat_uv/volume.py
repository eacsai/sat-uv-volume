"""The cube around a building, its voxels, occupancy and the satellite UV volume (GenSfM / CUPID layout).

Cube: centre c (3,) and side S in the world frame of the data; cube coordinates p = (X - c) / S lie in [-0.5, 0.5]^3,
axes = the world frame's axes. Voxel (i, j, k) of a res^3 grid has its centre at p = ((i, j, k) + 0.5) / res - 0.5
(meshgrid indexing 'ij', exactly as GenSfM's SparseUVStructure._get_uv_volume).

A UV volume of a view is the dict
    uv_volume  float32 [2, R, R, R]  (u, v) of every voxel centre, clamped to [0, 1]
    ssuv       uint8   [1, R, R, R]  1 where 0.01 <= u, v <= 0.99 on the unclamped values, as GenSfM's const_ssuv mask
    uv_raw     float32 [2, R, R, R]  (u, v) before the clamp
plus the satellite camera expressed in cube coordinates (camera_cube).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import SatelliteCamera, Window

MASK_LO, MASK_HI = 0.01, 0.99


@dataclass
class Cube:
    centre: np.ndarray
    side: float

    def __post_init__(self):
        self.centre = np.asarray(self.centre, np.float64).reshape(3)
        self.side = float(self.side)

    def to_cube(self, X) -> np.ndarray:
        return (np.asarray(X, np.float64) - self.centre) / self.side

    def from_cube(self, p) -> np.ndarray:
        return self.centre + self.side * np.asarray(p, np.float64)

    def to_dict(self) -> dict:
        return {"centre": [float(c) for c in self.centre], "side": self.side}

    @classmethod
    def from_dict(cls, d: dict) -> "Cube":
        return cls(d["centre"], d["side"])


def cube_around(vertices, pad: float = 1.1) -> Cube:
    """Axis-aligned cube centred on the bounding box of `vertices`, side = pad * its largest extent."""
    V = np.asarray(vertices, np.float64)
    lo, hi = V.min(0), V.max(0)
    return Cube(0.5 * (lo + hi), pad * float((hi - lo).max()))


def voxel_centres(res: int = 64) -> np.ndarray:
    """Cube coordinates of the voxel centres, [3, res, res, res] (meshgrid 'ij')."""
    x = (np.arange(res, dtype=np.float64) + 0.5) / res - 0.5
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), 0)


def voxel_points(cube: Cube, res: int = 64) -> np.ndarray:
    """World points of the voxel centres, (res^3, 3), in the order of a C-order flatten of [res, res, res]."""
    return cube.from_cube(voxel_centres(res).reshape(3, -1).T)


def occupancy(vertices, faces, cube: Cube, res: int = 64) -> np.ndarray:
    """Indices (M, 3) of the voxels the mesh surface passes through, as TRELLIS' dataset_toolkits/voxelize.py does it
    (vertices normalised to the cube and clipped to it, open3d VoxelGrid of voxel size 1 / res)."""
    import open3d as o3d

    p = np.clip(cube.to_cube(vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p), o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1.0 / res, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    idx = np.array([v.grid_index for v in grid.get_voxels()], np.int64).reshape(-1, 3)
    idx = idx[(idx >= 0).all(1) & (idx < res).all(1)]
    return np.unique(idx, axis=0).astype(np.int32)


def uv_volume_from_uv(u, v, valid=None, res: int = 64) -> dict:
    """The UV volume dict of flat per-voxel (u, v) (length res^3, C order); `valid` further restricts the mask."""
    u = np.asarray(u, np.float64).reshape(-1)
    v = np.asarray(v, np.float64).reshape(-1)
    mask = (u >= MASK_LO) & (u <= MASK_HI) & (v >= MASK_LO) & (v <= MASK_HI)
    if valid is not None:
        mask &= np.asarray(valid, bool).reshape(-1)
    uv_raw = np.stack([u, v]).reshape(2, res, res, res).astype(np.float32)
    return {"uv_volume": np.clip(uv_raw, 0.0, 1.0), "ssuv": mask.reshape(1, res, res, res).astype(np.uint8),
            "uv_raw": uv_raw}


def satellite_uv_volume(camera: SatelliteCamera, window: Window, cube: Cube, frame: str = "enu", res: int = 64) -> dict:
    """UV volume of the satellite window; also `camera_cube` = 2x4 A with [u, v] = A @ [p, 1] (p cube coords)."""
    col, row = camera.project(voxel_points(cube, res), frame)
    u, v = window.uv(col, row)
    out = uv_volume_from_uv(u, v, None, res)
    M = camera.matrix(frame)
    A = np.zeros((2, 4))
    A[:, :3] = M[:, :3] * cube.side
    A[:, 3] = M[:, :3] @ cube.centre + M[:, 3]
    A[0, 3] -= window.col0
    A[1, 3] -= window.row0
    out["camera_cube"] = (A / window.size).astype(np.float64)
    return out

