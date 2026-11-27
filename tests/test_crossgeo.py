import json
import os

import cv2
import numpy as np

from sat_uv import crossgeo, geo


def test_satellite_camera_from_pair(tmp_path):
    d = str(tmp_path)
    rec = {"name": "ground_2_pano", "type": "ground_pano", "station_id": "s",
           "ENU": {"R": [], "t": [], "latitude": 22.3255, "longitude": 114.2108, "altitude": 7.5},
           "EDS": {"R": [], "t": [129.85, 193.03, -62.59]}}
    with open(os.path.join(d, "coords.json"), "w") as f:
        json.dump({"world_system": "EDS", "records": [rec]}, f)
    img = np.zeros((1810, 1810, 3), np.uint8)
    img[0, 0] = (0, 0, 255)                                   # BGR red at the top-left
    cv2.imwrite(os.path.join(d, "ground_2_satellite.png"), img)
    rgb, cam, info = crossgeo.satellite_camera(d, 2)
    assert tuple(rgb[0, 0]) == (255, 0, 0)                    # returned as RGB
    assert cam.width == 1810 and cam.height == 1810
    assert np.isclose(cam.centre_e, -62.59) and np.isclose(cam.centre_n, -129.85)   # E = z, N = -x
    assert np.isclose(cam.m_per_px, geo.mercator_m_per_px(22.3255, 19)) and info["footprint_ok"]
    col, row = cam.project([129.85, 0.0, -62.59], "eds")      # the station itself -> image centre
    assert np.isclose(col, 905.0) and np.isclose(row, 905.0)
