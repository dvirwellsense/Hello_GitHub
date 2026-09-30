import json, numpy as np, cv2
import os
C = json.load(open(os.environ.get('CALIB_JSON', 'CalibrationResult_arc22.json')))
K = np.array(C['m_cameraMatrix'], float)
D = np.array(C['m_distCoeffs'], float)[:8]
XOFF = C['m_iXoffset']
W, H = C['m_undistROI']['Width'], C['m_undistROI']['Height']
def to_calib(img):
    """1280x720 -> 640x360, padded by XOFF each side -> 894x360 (the calibration frame)."""
    s = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(s, 0, 0, XOFF, W - 640 - XOFF, cv2.BORDER_CONSTANT)
def undistort_img(img):
    return cv2.undistort(to_calib(img), K, D)
def undistort_pts(p):
    """pixel points in raw 1280x720 -> normalized camera coords (x/z, y/z)."""
    q = np.asarray(p, np.float64).reshape(-1, 1, 2) / 2.0
    q[..., 0] += XOFF
    return cv2.undistortPoints(q, K, D).reshape(-1, 2)
