"""The satellite camera: an affine "leaned" parallel projection of a north-up image.

A north-up satellite image (e.g. a Web-Mercator crop) is close to an orthophoto for the ground, but every elevated
point is displaced horizontally in proportion to its height, because the satellite looked slightly off-nadir and the
imagery is orthorectified on the terrain, not on the buildings. With h = U - h0 the height above a datum h0:

    col = (E - centre_e + b_e + lean_e * h) / m_per_px + width  / 2
    row = (-(N - centre_n) - b_n - lean_n * h) / m_per_px + height / 2

(E, N, U) are local ENU metres, (col, row) are continuous pixel coordinates with the origin at the top-left corner of
the image (pixel k spans [k, k + 1)), image right = East, image down = South.

- centre_e, centre_n: the world position of the image centre (for a CrossGeo satellite PNG: the street-view station).
- m_per_px: ground resolution (Web-Mercator: geo.mercator_m_per_px(lat, zoom)).
- b = (b_e, b_n): where ground (h = 0) content sits relative to that nominal placement, metres East / North.
- lean = (lean_e, lean_n): horizontal displacement East / North in metres per metre of height.
The map is affine in the world point, so it is a 2x4 matrix (`matrix`).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from .geo import enu_matrix, to_enu


@dataclass
class SatelliteCamera:
    width: int
    height: int
    m_per_px: float
    centre_e: float
    centre_n: float
    h0: float = 0.0
    b_e: float = 0.0
    b_n: float = 0.0
    lean_e: float = 0.0
    lean_n: float = 0.0

    def project_enu(self, P) -> tuple:
        """(col, row) of ENU points (..., 3)."""
        P = np.asarray(P, np.float64)
        h = P[..., 2] - self.h0
        col = (P[..., 0] - self.centre_e + self.b_e + self.lean_e * h) / self.m_per_px + self.width / 2.0
        row = (-(P[..., 1] - self.centre_n) - self.b_n - self.lean_n * h) / self.m_per_px + self.height / 2.0
        return col, row

    def project(self, X, frame: str = "enu") -> tuple:
        """(col, row) of points (..., 3) given in `frame` ("enu" or "eds")."""
        return self.project_enu(to_enu(X, frame))

    def matrix(self, frame: str = "enu") -> np.ndarray:
        """2x4 matrix M with [col, row] = M @ [X, 1] for points X in `frame`."""
        g = self.m_per_px
        A = np.array([[1.0 / g, 0.0, self.lean_e / g],
                      [0.0, -1.0 / g, -self.lean_n / g]])
        c = np.array([(-self.centre_e + self.b_e - self.lean_e * self.h0) / g + self.width / 2.0,
                      (self.centre_n - self.b_n + self.lean_n * self.h0) / g + self.height / 2.0])
        return np.concatenate([A @ enu_matrix(frame), c[:, None]], 1)

    def with_params(self, **kw) -> "SatelliteCamera":
        return replace(self, **kw)

    def rebase(self, h0: float) -> "SatelliteCamera":
        """The same projection written with height datum h0: b absorbs lean * (h0 - self.h0)."""
        d = float(h0) - self.h0
        return replace(self, h0=float(h0), b_e=self.b_e + self.lean_e * d, b_n=self.b_n + self.lean_n * d)

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer))
                    else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "SatelliteCamera":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Window:
    """A square crop of the satellite image: pixels [col0, col0 + size) x [row0, row0 + size)."""
    col0: int
    row0: int
    size: int

    def uv(self, col, row) -> tuple:
        """Normalised window coordinates of image pixel coordinates (origin top-left, 1 = far edge)."""
        return (np.asarray(col, np.float64) - self.col0) / self.size, (np.asarray(row, np.float64) - self.row0) / self.size

    def crop(self, image: np.ndarray) -> np.ndarray:
        """The window of an image; parts outside the image are zero."""
        h, w = image.shape[:2]
        out = np.zeros((self.size, self.size) + image.shape[2:], image.dtype)
        r0, r1 = max(0, self.row0), min(h, self.row0 + self.size)
        c0, c1 = max(0, self.col0), min(w, self.col0 + self.size)
        if r1 > r0 and c1 > c0:
            out[r0 - self.row0:r1 - self.row0, c0 - self.col0:c1 - self.col0] = image[r0:r1, c0:c1]
        return out

    def inside(self, width: int, height: int) -> bool:
        return self.col0 >= 0 and self.row0 >= 0 and self.col0 + self.size <= width and self.row0 + self.size <= height

    def to_dict(self) -> dict:
        return {"col0": int(self.col0), "row0": int(self.row0), "size": int(self.size)}


def window_around(camera: SatelliteCamera, points, frame: str = "enu", margin_frac: float = 0.1,
                  min_margin_m: float = 5.0) -> Window:
    """The square window centred on the projection of `points` (e.g. the mesh vertices) that holds them
    all plus a margin of max(margin_frac * extent, min_margin_m)."""
    col, row = camera.project(points, frame)
    extent = max(float(col.max() - col.min()), float(row.max() - row.min()))
    margin = max(margin_frac * extent, min_margin_m / camera.m_per_px)
    size = int(math.ceil(extent + 2.0 * margin))
    col0 = int(round(0.5 * (col.min() + col.max()) - size / 2.0))
    row0 = int(round(0.5 * (row.min() + row.max()) - size / 2.0))
    return Window(col0, row0, size)
