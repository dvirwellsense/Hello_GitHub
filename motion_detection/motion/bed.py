"""Bed region and background point classes.

bed_mask: the bed top is the large dark board under the patient. Dark pixels (V < 90) are closed,
the largest component is kept and its convex hull is filled, so the patient lying on it is inside.

Background points (outside the dilated body mask) are classified per pair:
  bed      inside the bed mask and consistent with the bed-plane homography a->b (texture on the board:
           scratches, white squares, edges)
  offplane inside the bed mask but NOT consistent with the bed plane: reflections in the glossy board,
           cables, objects standing on the bed
  floor    outside the bed mask (floor, boxes, room; includes people standing next to the bed)
"""
import cv2
import numpy as np


def bed_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < 90).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        return np.zeros(img.shape[:2], np.uint8)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    cnts, _ = cv2.findContours(np.where(lab == k, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    return cv2.erode(m, np.ones((15, 15), np.uint8))  # stay off the bed rim


def classify_background(u0, u1, in_bed, plane_thr_norm):
    """u0/u1: normalized coords of background points; in_bed: bool. Returns labels array of str."""
    lab = np.where(in_bed, 'offplane', 'floor').astype(object)
    if in_bed.sum() >= 8:
        H, inl = cv2.findHomography(u0[in_bed], u1[in_bed], cv2.RANSAC, plane_thr_norm)
        if H is not None:
            idx = np.where(in_bed)[0]
            lab[idx[inl.ravel() > 0]] = 'bed'
    return lab
