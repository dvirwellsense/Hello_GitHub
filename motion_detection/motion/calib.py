"""Fisheye calibration of the gantry camera (ARC22 calibration JSON format).

The calibration was computed on a 894x360 frame: the raw 1280x720 image is resized to
640x360 and padded horizontally by m_iXoffset (127) pixels on each side.
Distortion model: OpenCV rational model (k1, k2, p1, p2, k3, k4, k5, k6).
"""
import json

import cv2
import numpy as np


class Calibration:
    def __init__(self, path):
        c = json.load(open(path, encoding='utf-8-sig'))
        self.K = np.array(c['m_cameraMatrix'], float)
        self.D = np.array(c['m_distCoeffs'], float)[:8]
        self.x_offset = int(c['m_iXoffset'])
        self.width = int(c['m_undistROI']['Width'])
        self.height = int(c['m_undistROI']['Height'])
        self.pixels_per_mm = float(c['m_fPixelsInMM'])
        self.focal_px = float(self.K[0, 0])

    def to_calib_frame(self, img):
        """Raw 1280x720 image -> padded 894x360 frame the calibration refers to."""
        small = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        right = self.width - 640 - self.x_offset
        return cv2.copyMakeBorder(small, 0, 0, self.x_offset, right, cv2.BORDER_CONSTANT)

    def undistort_image(self, img):
        return cv2.undistort(self.to_calib_frame(img), self.K, self.D)

    def undistort_points(self, pts):
        """Raw 1280x720 pixel points -> normalized camera coordinates (x/z, y/z)."""
        q = np.asarray(pts, np.float64).reshape(-1, 1, 2) / 2.0
        q[..., 0] += self.x_offset
        # Default OpenCV iteration count (5) is not enough for this strong distortion near the borders.
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)
        return cv2.undistortPoints(q, self.K, self.D, None, None, None, crit).reshape(-1, 2)

    def distort_points(self, norm):
        """Normalized camera coordinates -> raw 1280x720 pixels (inverse of undistort_points)."""
        obj = np.c_[norm, np.ones(len(norm))].reshape(-1, 1, 3)
        q, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), self.K, self.D)
        q = q.reshape(-1, 2)
        q[:, 0] -= self.x_offset
        return q * 2.0
