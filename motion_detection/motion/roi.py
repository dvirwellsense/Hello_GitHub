"""Region above the detector.

The detector is fixed under the bed, so its footprint is drawn once as a mask on the HOME frame
(raw 1280x720 image, white = above detector). For later frames the camera has moved; the mask is carried
along with a homography of the BED PLANE accumulated pair by pair. The homography is estimated only from
non-body points inside the (slightly enlarged) region, i.e. on the bed surface: the floor is far below the
bed and a homography fitted to it moves the region by several cm at 16 degrees.
Body points are mapped back to the home frame and looked up in the mask.
"""
import cv2
import numpy as np


BED_MARGIN_PX = 40  # enlargement of the region used to pick bed-surface points for the homography


class DetectorROI:
    def __init__(self, mask_path, calib, margin_px=0):
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(mask_path)
        if margin_px:
            m = cv2.dilate(m, np.ones((2 * margin_px + 1,) * 2, np.uint8))
        self._init(m > 127, calib)

    def _init(self, mask, calib):
        self.mask = mask
        k = 2 * BED_MARGIN_PX + 1
        self.bed = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        self.calib = calib
        self.H = np.eye(3)  # home frame -> current frame a (normalized coordinates)

    @classmethod
    def from_rect(cls, rect, calib, shape=(720, 1280)):
        obj = cls.__new__(cls)
        x, y, w, h = rect
        mask = np.zeros(shape, bool)
        mask[y:y + h, x:x + w] = True
        obj._init(mask, calib)
        return obj

    def __call__(self, norm_pts):
        """norm_pts: normalized coordinates in the current frame -> inside the detector region."""
        return self._lookup(norm_pts, self.mask)

    def near(self, norm_pts):
        """Inside the region enlarged by BED_MARGIN_PX (used to pick bed-surface points)."""
        return self._lookup(norm_pts, self.bed)

    def _lookup(self, norm_pts, mask):
        if len(norm_pts) == 0:
            return np.zeros(0, bool)
        h = np.c_[norm_pts, np.ones(len(norm_pts))] @ np.linalg.inv(self.H).T
        raw = self.calib.distort_points(h[:, :2] / h[:, 2:3])
        x, y = np.round(raw[:, 0]).astype(int), np.round(raw[:, 1]).astype(int)
        ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
        out = np.zeros(len(norm_pts), bool)
        out[ok] = mask[y[ok], x[ok]]
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
