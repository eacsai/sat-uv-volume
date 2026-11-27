"""sat-uv-volume: the ground-truth satellite UV volume of one building, from its mesh.

The UV volume follows GenSfM / CUPID (TRELLIS sparse structures): every voxel centre of a 64^3 cube around the building
gets the normalised image coordinate (u, v) it projects to in the satellite image, plus a mask that says whether it lands
inside the image window. The satellite image is not an orthophoto: tall objects lean. This package models it as an
affine "leaned" camera and calibrates that camera from a mesh of the building.
"""
from .geo import (lla_to_ecef, ecef_to_lla, lla_to_enu, enu_to_lla, to_enu, from_enu, enu_matrix, mercator_m_per_px)
from .camera import SatelliteCamera, Window, window_around
from .volume import Cube, cube_around, voxel_centres, voxel_points, occupancy, uv_volume_from_uv, satellite_uv_volume
from .calibrate import (CalibConfig, CalibResult, MeshEdgeRenderer, calibrate_satellite, score_camera, sample_surface)

__version__ = "0.1.0"
