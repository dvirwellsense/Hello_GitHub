"""Validate the selection of bed-plane background points (used by --guard bed) along a scan.

Per pair (moving gantry), background points inside the carried bed outline are tested against the
bed-plane homography a->b (RANSAC, 1 px); accepted = 'bed', rejected = 'offplane'. Reported:

  acceptance       accepted / points inside the outline (overall and near the ROI)
  transfer error   |H(a) - b| in px (undistorted half-res) for accepted, rejected, floor and BODY points.
                   Body points are 5-30 cm above the bed: they are a known near-plane surface, i.e. a
                   positive control for "close to the bed but not on it" (like reflections of the patient
                   in the glossy board). body_pass = fraction of body points that would pass the 1 px
                   plane test: the higher it is, the less the test can reject near-plane surfaces.
  baseline k       the same test over k frames (a = i-k) to see how a larger angle separates the plane.
  white squares    the four white stickers are ON the bed plane: their detected centroids are compared
                   with their HOME centroids carried by the accumulated bed homography (drift, raw px).
  contamination    epipolar residual of accepted bed points near the ROI by distance to the body mask,
                   for labelled-motion pairs vs the others.
Overlays (--overlay-frames): accepted (blue), rejected (orange), floor (green), body (grey), outline
(yellow), ROI (magenta), white-square prediction (cyan cross) vs detection (red circle).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.masks import phantom_mask
from motion.rigidity import epipolar_distance
from motion.roi import DetectorROI
from motion.scan import Scan
from motion.tracking import detect, track


def white_squares(img, bed_raster):
    """Centroids (raw px) of the white stickers inside the bed outline."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 190) & (hsv[..., 1] < 50) & (bed_raster > 0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if 2500 < area < 40000 and 0.5 < w / max(h, 1) < 2.0 and area > 0.6 * w * h:  # solid, roughly square
            out.append(cen[k])
    return np.array(out).reshape(-1, 2)


def transfer(H, u0, u1, f):
    if H is None or len(u0) == 0:
        return np.full(len(u0), np.nan)
    p = np.c_[u0, np.ones(len(u0))] @ H.T
    return np.linalg.norm(p[:, :2] / p[:, 2:3] - u1, axis=1) * f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--body-masks')
    ap.add_argument('--phantom', action='store_true')
    ap.add_argument('--roi', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--bed', default='data/bed_home_arc22.png')
    ap.add_argument('--near', type=float, default=150)
    ap.add_argument('--k', type=int, default=4, help='longer baseline (frames) for the plane test')
    ap.add_argument('--label', action='append', default=[], help='labelled motion range a-b')
    ap.add_argument('--overlay-frames', default='', help='comma list of frame indices for overlays')
    ap.add_argument('--carry', choices=('chain', 'direct'), default='chain',
                    help='chain: accumulate pairwise bed homographies (as evaluate_scan); direct: register every '
                         'frame to the HOME frame with SIFT on bed-outline features (no accumulation)')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    f = calib.focal_px
    Kn = np.diag([1 / f, 1 / f, 1])
    thr = 1.0 / f
    roi = (cv2.imread(a.roi, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    near_roi = cv2.distanceTransform(255 - roi, cv2.DIST_L2, 5) <= a.near
    bed = DetectorROI(a.bed, calib)
    labelled = [tuple(map(int, s.split('-'))) for s in a.label]
    overlay = {int(x) for x in a.overlay_frames.split(',') if x}
    out = Path(a.out)
    (out / 'overlays').mkdir(parents=True, exist_ok=True)
    exposure = scan.exposure_pairs()

    def body_of(i, img):
        if a.phantom:
            return cv2.erode(phantom_mask(img), np.ones((21, 21), np.uint8))
        for j in (i, i - 1):  # the last frame has no saved mask (masks are stored for frame a of each pair)
            m = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[j]):06d}.png'), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m
        raise FileNotFoundError(f'no body mask for frame {int(scan.index[i])}')

    def bed_raster():
        r = np.zeros((720, 1280), np.uint8)
        cv2.fillPoly(r, bed.outline(), 255)
        return r

    # HOME white squares (frame 0), carried later by the accumulated bed homography (= bed.H)
    img0 = scan.image(0)
    sift = cv2.SIFT_create(4000)
    home_body = cv2.dilate(body_of(0, img0), np.ones((61, 61), np.uint8))
    home_mask = ((bed_raster() > 0) & (home_body == 0)).astype(np.uint8) * 255
    kp0, des0 = sift.detectAndCompute(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), home_mask)
    pts0_n = calib.undistort_points(np.float32([k.pt for k in kp0])) if kp0 else np.zeros((0, 2))
    matcher = cv2.BFMatcher()

    def direct_H(img, body_img):
        """HOME -> img bed-plane homography from SIFT matches (normalized coords), and its inlier count."""
        m = (cv2.dilate(body_img, np.ones((61, 61), np.uint8)) == 0).astype(np.uint8) * 255
        kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), m)
        if des is None or des0 is None or len(kp) < 10:
            return None, 0
        good = [x for x, y in matcher.knnMatch(des0, des, k=2) if x.distance < 0.75 * y.distance]
        if len(good) < 10:
            return None, 0
        a0 = pts0_n[[g.queryIdx for g in good]]
        a1 = calib.undistort_points(np.float32([kp[g.trainIdx].pt for g in good]))
        Hd, inl = cv2.findHomography(a0, a1, cv2.RANSAC, 1.5 / f)
        return Hd, int(inl.sum()) if inl is not None else 0
    sq_home = white_squares(img0, bed_raster())
    sq_home_n = calib.undistort_points(sq_home) if len(sq_home) else np.zeros((0, 2))

    rows, contam = [], []
    imgs = {0: img0}
    prev = img0
    for i in range(1, len(scan)):
        imgs_draw = None
        cur = scan.image(i)
        imgs[i] = cur
        imgs.pop(i - a.k - 1, None)
        ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (prev, cur))
        body = body_of(i - 1, prev)
        p0 = detect(ga)
        p1, ok = track(ga, gb, p0)
        p0, p1 = p0[ok], p1[ok]
        xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
        on_bg = cv2.dilate(body, np.ones((81, 81), np.uint8))[yi, xi] == 0
        on_body = (body[yi, xi] > 0) & (roi[yi, xi] > 0)
        u0, u1 = calib.undistort_points(p0), calib.undistort_points(p1)
        inside = on_bg & bed(u0)
        H, inl = (None, None)
        accepted = np.zeros(len(u0), bool)
        if inside.sum() >= 8:
            H, inl = cv2.findHomography(u0[inside], u1[inside], cv2.RANSAC, thr)
            if H is not None:
                accepted[np.where(inside)[0][inl.ravel() > 0]] = True
        rejected = inside & ~accepted
        floor = on_bg & ~inside
        near = near_roi[yi, xi]
        angle_a, angle_b = scan.angle(i - 1, a.sync), scan.angle(i, a.sync)
        moving = abs(angle_b - angle_a) >= 0.02
        te = transfer(H, u0, u1, f)
        row = dict(frame=int(scan.index[i]), angle=float(angle_b), dangle=float(angle_b - angle_a),
                   exposure=bool(exposure[i]), moving=bool(moving),
                   labelled=any(lo <= int(scan.index[i]) <= hi for lo, hi in labelled),
                   n_inside=int(inside.sum()), n_accepted=int(accepted.sum()), n_rejected=int(rejected.sum()),
                   n_floor=int(floor.sum()), n_body=int(on_body.sum()),
                   accept_rate=float(accepted.sum() / max(inside.sum(), 1)),
                   n_inside_near=int((inside & near).sum()), n_accepted_near=int((accepted & near).sum()),
                   accept_rate_near=float((accepted & near).sum() / max((inside & near).sum(), 1)))
        for k_, sel in (('accepted', accepted), ('rejected', rejected), ('floor', floor), ('body', on_body)):
            row[f'te_{k_}'] = float(np.nanmedian(te[sel])) if sel.any() else np.nan
        row['body_pass'] = float(np.mean(te[on_body] < 1.0)) if on_body.any() and H is not None else np.nan
        row['floor_pass'] = float(np.mean(te[floor] < 1.0)) if floor.any() and H is not None else np.nan
        if moving:
            Fe = Kn.T @ gantry.essential(angle_a, angle_b) @ Kn
            e = epipolar_distance(Fe, u0 * f, u1 * f)
            for k_, sel in (('accepted_near', accepted & near), ('rejected_near', rejected & near), ('floor', floor)):
                row[f'epi_{k_}'] = float(np.median(e[sel])) if sel.any() else np.nan
            # contamination: accepted bed points near the ROI by distance to the body mask
            dist_body = cv2.distanceTransform(255 - body, cv2.DIST_L2, 5)
            sel = accepted & near
            for d, ee, (px_, py_) in zip(dist_body[yi[sel], xi[sel]], e[sel], p0[sel]):
                contam.append((row['frame'], row['labelled'], row['exposure'], float(d), float(ee), float(px_), float(py_)))
        # longer baseline k: same plane test between frame i-k and i
        if i - a.k >= 0 and (i - a.k) in imgs:
            gk = cv2.cvtColor(imgs[i - a.k], cv2.COLOR_BGR2GRAY)
            bk = body_of(i - a.k, imgs[i - a.k])
            q0 = detect(gk)
            q1, okk = track(gk, gb, q0, max_level=5)
            q0, q1 = q0[okk], q1[okk]
            qx, qy = q0[:, 0].astype(int), q0[:, 1].astype(int)
            qbg = cv2.dilate(bk, np.ones((81, 81), np.uint8))[qy, qx] == 0
            qbody = (bk[qy, qx] > 0) & (roi[qy, qx] > 0)
            w0, w1 = calib.undistort_points(q0), calib.undistort_points(q1)
            # outline position at frame i-k is not stored; use the current outline (shift over k frames is small)
            qin = qbg & bed(w0)
            if qin.sum() >= 8:
                Hk, il = cv2.findHomography(w0[qin], w1[qin], cv2.RANSAC, thr)
                tk = transfer(Hk, w0, w1, f)
                if row['frame'] in overlay:
                    vk = cur.copy()
                    cv2.polylines(vk, bed.outline(), True, (0, 255, 255), 2)
                    acc_k = np.zeros(len(w0), bool)
                    acc_k[np.where(qin)[0][il.ravel() > 0]] = True
                    for sel, col, rr in ((qbody, (170, 170, 170), 2), (acc_k, (230, 120, 40), 3), (qin & ~acc_k, (40, 110, 240), 4)):
                        for x, y in q1[sel]:
                            cv2.circle(vk, (int(x), int(y)), rr, col, -1)
                    cv2.putText(vk, f"frame {row['frame']}  baseline {a.k} frames: accepted {int(acc_k.sum())}/{int(qin.sum())}",
                                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imwrite(str(out / 'overlays' / f"{a.name}_{row['frame']:03d}_k{a.k}.jpg"), vk)
                row[f'accept_rate_k{a.k}'] = float(il.sum() / qin.sum())
                row[f'body_pass_k{a.k}'] = float(np.mean(tk[qbody] < 1.0)) if qbody.any() else np.nan
                row[f'te_body_k{a.k}'] = float(np.median(tk[qbody])) if qbody.any() else np.nan
        # white squares: detected vs carried HOME position (bed.H is home -> frame a; advance to b first)
        if a.carry == 'direct':
            Hd, nd = direct_H(cur, body_of(i, cur))
            row['direct_inliers'] = nd
            if Hd is not None and nd >= 20:
                bed.H = Hd
            else:
                bed.advance(H)  # fallback
        else:
            bed.advance(H)
        if len(sq_home_n):
            h = np.c_[sq_home_n, np.ones(len(sq_home_n))] @ bed.H.T
            pred = calib.distort_points(h[:, :2] / h[:, 2:3])
            det = white_squares(cur, bed_raster())
            d = [np.min(np.linalg.norm(det - p, axis=1)) for p in pred] if len(det) else []
            d = [x for x in d if x < 80]  # matched stickers (others covered by the body)
            row['squares_matched'] = len(d)
            row['square_drift_px'] = float(np.median(d)) if d else np.nan
            row['square_drift_max_px'] = float(np.max(d)) if d else np.nan
            if row['frame'] in overlay:
                v = cur.copy()
                for c in pred:
                    cv2.drawMarker(v, tuple(int(t) for t in c), (255, 255, 0), cv2.MARKER_CROSS, 30, 3)
                for c in det:
                    cv2.circle(v, tuple(int(t) for t in c), 14, (0, 0, 255), 3)
                imgs_draw = v
        if row['frame'] in overlay:
            v = imgs_draw if imgs_draw is not None else cur.copy()
            cv2.polylines(v, bed.outline(), True, (0, 255, 255), 2)
            cnt, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(v, cnt, -1, (255, 0, 255), 2)
            for sel, col, r in ((floor, (60, 180, 60), 2), (on_body, (170, 170, 170), 2), (accepted, (230, 120, 40), 3),
                                (rejected, (40, 110, 240), 4)):
                for x, y in p1[sel]:
                    cv2.circle(v, (int(x), int(y)), r, col, -1)
            cv2.putText(v, f"frame {row['frame']}  accepted {row['n_accepted']}/{row['n_inside']} "
                           f"({100 * row['accept_rate']:.0f}%), near ROI {row['n_accepted_near']}/{row['n_inside_near']}",
                        (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(str(out / 'overlays' / f"{a.name}_{row['frame']:03d}.jpg"), v)
        rows.append(row)
        prev = cur
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / f'{a.name}_bed_pairs.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, restval='')
        w.writeheader()
        w.writerows(rows)
    with open(out / f'{a.name}_bed_contamination.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['frame', 'labelled', 'exposure', 'dist_to_body_px', 'epi_px', 'x', 'y'])
        w.writerows(contam)
    print(f'{a.name}: {len(rows)} pairs, home squares {len(sq_home)}')


if __name__ == '__main__':
    main()
