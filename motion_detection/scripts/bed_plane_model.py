"""Fit the bed plane once per scan and predict the bed outline at every angle from the gantry model.

Chaining pairwise bed homographies drifts with the angle (a few tens of px at +-13 deg) because every
0.5 deg step is a slightly biased compromise between depths. Here the bed is a fixed plane in the world:
  * camera pose per frame: gantry model + encoder angle
  * plane (normal + offset, 3 parameters) fitted to the white stickers on the bed: their HOME centroids
    are back-projected onto the plane and re-projected into every frame where stickers are detected
  * per frame the HOME -> frame homography of that plane is exact, so nothing accumulates.
Outputs per scan: plane parameters, sticker error per frame (model vs chain), JSON + CSV.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.scan import Scan


def white_squares(img, allowed):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > 190) & (hsv[..., 1] < 50) & allowed).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if 2500 < area < 40000 and 0.5 < w / max(h, 1) < 2.0 and area > 0.6 * w * h:
            out.append(cen[k])
    return np.array(out).reshape(-1, 2)


class PlaneModel:
    """Bed plane in world coordinates: n . X = d, n = unit normal from two angles."""

    def __init__(self, gantry, calib, angle_home):
        self.g, self.c = gantry, calib
        P0 = gantry.projections([angle_home])[0]
        self.R0, self.t0 = P0[:, :3], P0[:, 3]
        self.C0 = -self.R0.T @ self.t0

    @staticmethod
    def normal(p):
        return Rot.from_rotvec([p[0], p[1], 0]).apply([0, 0, 1.0])

    def home_to_world(self, p, u):
        """Normalized HOME points -> 3D points on the plane."""
        n, d = self.normal(p), p[2]
        rays = (np.c_[u, np.ones(len(u))]) @ self.R0  # R0^T r_cam, world direction
        s = (d - n @ self.C0) / (rays @ n)
        return self.C0 + s[:, None] * rays

    def project(self, X, angle):
        P = self.g.projections([angle])[0]
        x = X @ P[:, :3].T + P[:, 3]
        return x[:, :2] / x[:, 2:3]

    def H(self, p, angle):
        """HOME -> frame homography (normalized coords) of the plane."""
        grid = np.array([[-0.6, -0.4], [0.6, -0.4], [0.6, 0.4], [-0.6, 0.4], [0, 0]])
        X = self.home_to_world(p, grid)
        Hm, _ = cv2.findHomography(grid, self.project(X, angle))
        return Hm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--chain-csv', help='bed_validation.py output (chain carry) for comparison')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allowed = np.ones((720, 1280), bool)
    allowed[:60] = allowed[-60:] = False
    allowed[:, :80] = allowed[:, -80:] = False
    home = white_squares(scan.image(0), allowed)
    if len(home) == 0:
        raise SystemExit('no stickers visible on the HOME frame')
    home_n = calib.undistort_points(home)
    angles = scan.angle(np.arange(len(scan)), a.sync)
    dets = [white_squares(scan.image(i), allowed) for i in range(len(scan))]
    pm = PlaneModel(gantry, calib, angles[0])

    def predict(p, i):
        return calib.distort_points(pm.project(pm.home_to_world(p, home_n), angles[i]))

    def matches(p, gate):
        res = []
        for i, d in enumerate(dets):
            if len(d) == 0:
                continue
            pr = predict(p, i)
            for k, q in enumerate(pr):
                j = int(np.argmin(np.linalg.norm(d - q, axis=1)))
                if np.linalg.norm(d[j] - q) < gate:
                    res.append((i, k, d[j]))
        return res

    # initial plane: perpendicular to the HOME viewing direction at depth z0 (grid search)
    view = pm.R0.T @ np.array([0, 0, 1.0])
    rv = Rot.align_vectors([view], [[0, 0, 1.0]])[0].as_rotvec()
    best = None
    for z0 in np.linspace(0.2, 3.0, 57):
        p = np.r_[rv[:2], view @ (pm.C0 + z0 * view)]
        m = matches(p, 60)
        if not m:
            continue
        err = np.median([np.linalg.norm(predict(p, i)[k] - q) for i, k, q in m])
        score = len(m) - err / 10
        if best is None or score > best[0]:
            best = (score, p)
    p = best[1]
    for gate in (60, 30, 15):
        m = matches(p, gate)

        def res(pp):
            return np.concatenate([predict(pp, i)[k] - q for i, k, q in m])
        p = least_squares(res, p, loss='soft_l1', f_scale=2.0).x
    m = matches(p, 15)
    rows = []
    chain = {}
    if a.chain_csv:
        chain = {int(x['frame']): x for x in csv.DictReader(open(a.chain_csv))}
    for i, d in enumerate(dets):
        pr = predict(p, i)
        errs = [np.min(np.linalg.norm(d - q, axis=1)) for q in pr] if len(d) else []
        errs = [e for e in errs if e < 40]
        fr = int(scan.index[i])
        rows.append(dict(frame=fr, angle=float(angles[i]), squares_detected=len(d), squares_matched=len(errs),
                         model_err_px=float(np.median(errs)) if errs else np.nan,
                         model_err_max_px=float(np.max(errs)) if errs else np.nan,
                         chain_drift_px=float(chain[fr]['square_drift_px']) if fr in chain and chain[fr].get('square_drift_px') not in (None, '', 'nan') else np.nan))
    with open(out / f'{a.name}_plane_model.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    n = PlaneModel.normal(p)
    summ = dict(scan=a.name, plane_normal_world=n.tolist(), plane_offset=float(p[2]), params=p.tolist(),
                camera_home_center=pm.C0.tolist(), home_stickers=len(home), matched_observations=len(m),
                distance_home_camera_to_plane=float(abs(n @ pm.C0 - p[2])),
                tilt_vs_home_view_deg=float(np.degrees(np.arccos(abs(n @ view)))),
                model_err_median_px=float(np.nanmedian([r['model_err_px'] for r in rows])),
                model_err_p90_px=float(np.nanpercentile([r['model_err_px'] for r in rows if np.isfinite(r['model_err_px'])], 90)))
    (out / f'{a.name}_plane_model.json').write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
