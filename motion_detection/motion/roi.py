"""Region above the detector.

The detector is fixed under the bed, so its footprint is drawn once as a mask on the HOME frame
(raw 1280x720 image, white = above detector). For later frames the camera has moved; the mask is carried
along with a background homography accumulated pair by pair (bed plane). Body points are mapped back
to the home frame and looked up in the mask.
"""
import cv2
import numpy as np


class DetectorROI:
    def __init__(self, mask_path, calib, margin_px=0):
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(mask_path)
        if margin_px:
            m = cv2.dilate(m, np.ones((2 * margin_px + 1,) * 2, np.uint8))
        self.mask = m > 127
        self.calib = calib
        self.H = np.eye(3)  # home frame -> current frame a (normalized coordinates)

    @classmethod
    def from_rect(cls, rect, calib, shape=(720, 1280)):
        obj = cls.__new__(cls)
        x, y, w, h = rect
        obj.mask = np.zeros(shape, bool)
        obj.mask[y:y + h, x:x + w] = True
        obj.calib, obj.H = calib, np.eye(3)
        return obj

    def __call__(self, norm_pts):
        """norm_pts: normalized coordinates in the current frame -> inside the detector region."""
        if len(norm_pts) == 0:
            return np.zeros(0, bool)
        h = np.c_[norm_pts, np.ones(len(norm_pts))] @ np.linalg.inv(self.H).T
        raw = self.calib.distort_points(h[:, :2] / h[:, 2:3])
        x, y = np.round(raw[:, 0]).astype(int), np.round(raw[:, 1]).astype(int)
        ok = (x >= 0) & (y >= 0) & (x < self.mask.shape[1]) & (y < self.mask.shape[0])
        out = np.zeros(len(norm_pts), bool)
        out[ok] = self.mask[y[ok], x[ok]]
        return out

    def advance(self, H_ab):
        """Move to the next frame with the background homography a -> b."""
        if H_ab is not None:
            self.H = H_ab @ self.H

    def outline(self, shape=(720, 1280), step=8):
        """Raw-pixel polygon(s) of the region in the current frame, for drawing."""
        cnts, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in cnts:
            c = c.reshape(-1, 2)[::step].astype(float)
            n = self.calib.undistort_points(c)
            h = np.c_[n, np.ones(len(n))] @ self.H.T
            out.append(self.calib.distort_points(h[:, :2] / h[:, 2:3]).astype(np.int32))
        return out
