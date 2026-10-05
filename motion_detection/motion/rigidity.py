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


def static_cluster_score(pts, disp, k=30):
    """Same-angle pair: max over points of |median displacement vector of the k nearest points|."""
    if len(disp) < k:
        return np.nan
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    nn = np.argsort(D, 1)[:, :k]
    return float(np.max(np.linalg.norm(np.median(disp[nn], axis=1), axis=1)))


def evaluate_pair(img_a, img_b, angle_a, angle_b, body, specular, calib, gantry, bg_margin_px=81,
                  roi=None, same_angle_deg=0.02, detect_mode='global', roi_region=None, return_points=False,
                  body_fb_max=0.5, bed=None, bg_near=None):
    """All scores for one adjacent pair. body/specular are masks of frame a.
    Same-angle pairs (gantry stopped) are compared directly: any displacement is motion.
    roi: optional callable(normalized points of frame a) -> bool, the region above the detector.
    detect_mode: 'global' = corners detected on the whole image, then filtered to body/ROI (original).
                 'masked' = background points as in 'global'; body points detected separately inside
                 body & not-specular & roi_region (raster of the ROI in frame a), so the corner-quality
                 threshold is relative to the body itself and not to the strongest corner in the image.
                 'masked_abs' = like 'masked' but with the absolute quality threshold of 'global': body
                 points no longer compete with the background for the 3000-point cap, but are not weaker.
    return_points: also return per-point arrays (for inspection).
    body_fb_max: forward-backward limit (px) for BODY points (background points always use 0.5).
    bed: optional DetectorROI-like outline of the bed (normalized-point lookup, carried by the caller).
         When given, background points are split into bed (on the bed plane) / offplane (inside the bed
         outline, off the plane: reflections, cables) / floor, and the homography returned for carrying
         the ROI and the bed outline is fitted on bed-plane points only.
    bg_near: optional raw-pixel bool raster of frame a: "near the ROI" for the local bed statistics."""
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (img_a, img_b))
    p0 = detect(ga)
    p1, ok, fb = track(ga, gb, p0, return_fb=True)
    p0, p1, fb = p0[ok], p1[ok], fb[ok]
    xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
    on_body = (body[yi, xi] > 0) & (specular[yi, xi] == 0)
    on_bg = cv2.dilate(body, np.ones((bg_margin_px, bg_margin_px), np.uint8))[yi, xi] == 0
    n_detected_body = None
    if detect_mode in ('masked', 'masked_abs'):
        m = ((body > 0) & (specular == 0)).astype(np.uint8) * 255
        if roi_region is not None:
            m &= roi_region
        q0 = detect(ga, mask=m, absolute_quality=(detect_mode == 'masked_abs'))
        n_detected_body = len(q0)
        q1, qok, qfb = track(ga, gb, q0, return_fb=True)
        q0, q1, qfb = q0[qok], q1[qok], qfb[qok]
        keep = ~on_body  # background points stay exactly as in 'global'
        p0, p1, fb = np.vstack([p0[keep], q0]), np.vstack([p1[keep], q1]), np.r_[fb[keep], qfb]
        on_bg = np.r_[on_bg[keep], np.zeros(len(q0), bool)]
        on_body = np.r_[np.zeros(keep.sum(), bool), np.ones(len(q0), bool)]
    elif detect_mode != 'global':  # pragma: no cover
        raise ValueError(detect_mode)
    if body_fb_max < 0.5:
        on_body &= fb < body_fb_max
    f = calib.focal_px
    u0, u1 = calib.undistort_points(p0), calib.undistort_points(p1)
    n0, n1 = u0 * f, u1 * f
    # Bed-plane homography a -> b (normalized coords); carries the detector ROI from the home frame.
    # Only non-body points on the bed near the detector region: the floor is a different plane.
    H = None
    bg_stats = {}
    if bed is not None:
        from .bed import classify_background
        lab = np.full(len(u0), '', dtype=object)
        lab[on_bg] = classify_background(u0[on_bg], u1[on_bg], bed(u0[on_bg]), 1.0 / f)
        isbed = lab == 'bed'
        if isbed.sum() >= 8:
            H, _ = cv2.findHomography(u0[isbed], u1[isbed], cv2.RANSAC, 1.0 / f)
        px, py = p0[:, 0].astype(int), p0[:, 1].astype(int)
        near = np.ones(len(u0), bool) if bg_near is None else bg_near[py.clip(0, bg_near.shape[0] - 1), px.clip(0, bg_near.shape[1] - 1)]
        bg_stats = dict(_lab=lab, _near=near)
    else:
        on_bed = on_bg & roi.near(u0) if roi is not None else on_bg
        if on_bed.sum() < 15:
            on_bed = on_bg
        if on_bed.sum() >= 8:
            H, _ = cv2.findHomography(u0[on_bed], u1[on_bed], cv2.RANSAC, 1.0 / f)
    if roi is not None:
        on_body &= roi(u0)
    out = dict(_H=H, n_body=int(on_body.sum()), n_bg=int(on_bg.sum()),
               n_body_detected=n_detected_body if n_detected_body is not None else np.nan,
               fb_med_body=float(np.median(fb[on_body])) if on_body.any() else np.nan,
               fb_p90_body=float(np.percentile(fb[on_body], 90)) if on_body.any() else np.nan,
               body_flow=float(np.median(np.linalg.norm(n1[on_body] - n0[on_body], axis=1))) if on_body.any() else np.nan)
    b0, b1 = n0[on_body], n1[on_body]
    out['same_angle'] = bool(abs(angle_b - angle_a) < same_angle_deg)
    if out['same_angle']:
        # Remove the common image shift measured on the background (camera vibration), then look for
        # a coherent local displacement on the body.
        shift = np.median(n1[on_bg] - n0[on_bg], axis=0) if on_bg.sum() > 20 else np.zeros(2)
        d = b1 - b0 - shift
        out.update(static_med=float(np.linalg.norm(np.median(d, axis=0))) if len(d) else np.nan,
                   static_cl=static_cluster_score(b0, d))
    else:
        out.update(static_med=np.nan, static_cl=np.nan)
    if bg_stats and not out['same_angle']:
        # background statistics are recorded even when the body has too few points
        Kn0 = np.diag([1 / f, 1 / f, 1])
        Fe0 = Kn0.T @ gantry.essential(angle_a, angle_b) @ Kn0
        eb = epipolar_distance(Fe0, n0, n1)
        lab, near = bg_stats['_lab'], bg_stats['_near']
        for k in ('bed', 'offplane', 'floor'):
            sel = lab == k
            out[f'n_{k}'] = int(sel.sum())
            out[f'epi_{k}'] = float(np.median(eb[sel])) if sel.any() else np.nan
        sel = (lab == 'bed') & near
        out['n_bed_near'] = int(sel.sum())
        out['epi_bed_near'] = float(np.median(eb[sel])) if sel.any() else np.nan
    if len(b0) < 8 or out['same_angle']:
        out.update(affine_med=np.nan, affine_cl=np.nan, epi_img_med=np.nan, epi_img_cl=np.nan,
                   epi_enc_med=np.nan, epi_enc_cl=np.nan, epi_enc_cl30=np.nan, epi_enc_bg=np.nan,
                   epi_sens_px_per_deg=np.nan)
        return (out, None) if return_points else out

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
    # Sensitivity of the residual to an error in the angle of frame b (px per degree): how much a
    # timing/encoder error would move the score.
    Fp = Kn.T @ gantry.essential(angle_a, angle_b + 0.05) @ Kn
    sens = float(np.median(np.abs(epipolar_distance(Fp, b0, b1) - e)) / 0.05)
    out.update(epi_enc_med=float(np.median(e)), epi_enc_cl=cluster_score(b0, e), epi_enc_cl30=cluster_score(b0, e, 30),
               epi_enc_bg=float(np.median(epipolar_distance(Fe, n0[on_bg], n1[on_bg]))), epi_sens_px_per_deg=sens)
    if return_points:
        return out, dict(p0=p0[on_body], p1=p1[on_body], fb=fb[on_body], n0=b0, n1=b1, e=e, Fe=Fe,
                         bg_p0=p0[on_bg], bg_n0=n0[on_bg], bg_e=epipolar_distance(Fe, n0[on_bg], n1[on_bg]))
    return out
