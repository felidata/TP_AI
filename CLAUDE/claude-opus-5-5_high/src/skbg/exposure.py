"""Per-pixel effective exposure time.

Physical model (same convention as the pre-existing MOSKITA routine
``moskita.exposure.make_pixel_time_map``; re-implemented here, not imported):
with EXPOSURE = 0 the CCD is cleared and immediately read continuously. A pixel
at stored position (row r, column c) is read at readout index k = r*NCOL + c,
so it has integrated charge for  t(r,c) = EXPOSURE + T_readout * k / (NROW*NCOL),
with T_readout = DATEEND - DATESTART. Units: days.
"""
from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86400.0


def exposure_map_days(timing: dict, rows: tuple[int, int], cols: tuple[int, int]) -> np.ndarray:
    r = np.arange(rows[0], rows[1])[:, None]
    c = np.arange(cols[0], cols[1])[None, :]
    k = r * timing["ncol"] + c
    t = timing["exposure_s"] + timing["readout_s"] * k / (timing["nrow"] * timing["ncol"])
    return t / SECONDS_PER_DAY
