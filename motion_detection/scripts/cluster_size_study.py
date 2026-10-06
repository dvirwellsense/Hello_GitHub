"""One-factor study: the cluster size k of the score (median residual of the k nearest points, max over points).

For every pair in the exposure window of one scan, the body points are tracked exactly as in evaluate_scan.py
(E3, union masks, calibrated ROI) and the score is computed for several k. Nothing else changes: no threshold,
mask or tracking parameter. One CSV per scan: frame, n_body, one column per k.

Example:
  python scripts/cluster_size_study.py --scan <scan_dir> --body-masks <masks_dir> --out results/cluster_size_2026_10/A44.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motion.calib import Calibration  # noqa: E402
from motion.gantry import GantryModel  # noqa: E402
from motion.rigidity import cluster_score, evaluate_pair  # noqa: E402
from motion.roi import DetectorROI  # noqa: E402
from motion.scan import Scan  # noqa: E402

KS = (8, 12, 16, 20, 30, 45)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    calib, gantry, scan = Calibration(a.calib), GantryModel.load(a.gantry), Scan(a.scan)
    roi = DetectorROI(a.detector_mask, calib)
    region = roi.mask.astype(np.uint8) * 255
    exposure = scan.exposure_pairs()
    rows = []
    for i in range(1, len(scan)):
        if not exposure[i]:
            continue
        body = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i - 1]):06d}.png'), cv2.IMREAD_GRAYSCALE)
        r, pts = evaluate_pair(scan.image(i - 1), scan.image(i), scan.angle(i - 1, a.sync), scan.angle(i, a.sync), body,
                               np.zeros_like(body), calib, gantry, roi=roi, detect_mode='masked_abs', roi_region=region,
                               body_fb_max=0.1, return_points=True)
        row = dict(frame=int(scan.index[i]), n_body=r['n_body'])
        for k in KS:
            row[f'k{k}'] = round(cluster_score(pts['n0'], pts['e'], k), 4) if pts is not None else float('nan')
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['frame', 'n_body'] + [f'k{k}' for k in KS])
        w.writeheader()
        w.writerows(rows)
    print('wrote', a.out, len(rows), 'pairs')


if __name__ == '__main__':
    main()
