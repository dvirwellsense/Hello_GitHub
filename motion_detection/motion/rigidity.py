"""Rigidity check for one frame pair.

Static scene => every body point in frame b lies on the epipolar line of its position in frame a.
Scores are in pixels of the undistorted calibration frame (half resolution; ~2.2 mm per px at the bed plane).
"""
import cv2
import numpy as np

from .tracking import detect, track


def epipolar_distance(F, a, b):
    """Distance of points b from the epipolar lines F a (same units as a, b)."""
    lines = np.c_[a, np.ones(len(a))] @ F.T
    return np.abs(np.sum(lines * np.c_[b, np.ones(len(b))], 1)) / np.hypot(lines[:, 0], lines[:, 1])


def cluster_score(pts, err, k=8):
    """Max over points of the median error of the point and its k-1 nearest neighbours.
    Requires a spatially coherent group to move; isolated bad tracks do not trigger."""
    if len(err) < k:
        return np.nan
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    nn = np.argsort(D, 1)[:, :k]
    return float(np.max(np.median(err[nn], axis=1)))


def evaluate_pair(img_a, img_b, angle_a, angle_b, body, specular, calib, gantry, bg_margin_px=81):
    """All scores for one adjacent pair. body/specular are masks of frame a."""
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (img_a, img_b))
    p0 = detect(ga)
    p1, ok = track(ga, gb, p0)
    p0, p1 = p0[ok], p1[ok]
    xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
    on_body = (body[yi, xi] > 0) & (specular[yi, xi] == 0)
    on_bg = cv2.dilate(body, np.ones((bg_margin_px, bg_margin_px), np.uint8))[yi, xi] == 0
    f = calib.focal_px
    n0, n1 = calib.undistort_points(p0) * f, calib.undistort_points(p1) * f
    out = dict(n_body=int(on_body.sum()), n_bg=int(on_bg.sum()),
               body_flow=float(np.median(np.linalg.norm(n1[on_body] - n0[on_body], axis=1))) if on_body.any() else np.nan)
    b0, b1 = n0[on_body], n1[on_body]

    # Current method: 2D similarity estimated on background, applied to the body.
    A, _ = cv2.estimateAffinePartial2D(n0[on_bg], n1[on_bg], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    r = np.linalg.norm(cv2.transform(b0.reshape(1, -1, 2), A).reshape(-1, 2) - b1, axis=1) if A is not None else np.full(len(b0), np.nan)
    out.update(affine_med=float(np.median(r)), affine_cl=cluster_score(b0, r))

    # Epipolar check, geometry from the image (fundamental matrix on background points).
    Fi, _ = cv2.findFundamentalMat(n0[on_bg], n1[on_bg], cv2.FM_RANSAC, 0.5, 0.999)
    e = epipolar_distance(Fi, b0, b1) if Fi is not None and Fi.shape == (3, 3) else np.full(len(b0), np.nan)
    out.update(epi_img_med=float(np.nanmedian(e)), epi_img_cl=cluster_score(b0, e))

    # Epipolar check, geometry from the encoder (gantry model).
    Kn = np.diag([1 / f, 1 / f, 1])
    Fe = Kn.T @ gantry.essential(angle_a, angle_b) @ Kn
    e = epipolar_distance(Fe, b0, b1)
    out.update(epi_enc_med=float(np.median(e)), epi_enc_cl=cluster_score(b0, e),
               epi_enc_bg=float(np.median(epipolar_distance(Fe, n0[on_bg], n1[on_bg]))))
    return out
