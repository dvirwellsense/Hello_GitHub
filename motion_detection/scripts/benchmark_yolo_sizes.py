"""Compare YOLO segmentation weights (n, s, m, 11x) with the hand-drawn masks: IoU / recall / precision inside the detector ROI and over
the whole image, and the time per frame. Frames and masks are given as image=mask pairs.

  python scripts/benchmark_yolo_sizes.py --weights yolov8n-seg.pt yolov8s-seg.pt yolov8m-seg.pt yolo11x-seg.pt \
      --pair A07_53.jpg=A07_53_mask.png --pair ... --out results/mask_comparison_2026_10/yolo_sizes.txt
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import motion.masks as M  # noqa: E402


def st(p, g, reg=None):
    if reg is not None:
        p, g = p & reg, g & reg
    i, u = (p & g).sum(), (p | g).sum()
    return i / u, i / g.sum(), i / max(p.sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--pair', action='append', required=True)
    ap.add_argument('--roi', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--out')
    a = ap.parse_args()
    roi = cv2.imread(a.roi, 0) > 0
    pairs = [(cv2.imread(p.split('=')[0]), cv2.imread(p.split('=')[1], 0) > 0, Path(p.split('=')[0]).stem) for p in a.pair]
    lines = []
    for w in a.weights:
        for sel, conf in (('union', 0.1), ('top', 0.25)):
            M.SEG_SELECT, M.SEG_CONF = sel, conf
            M.person_mask_seg(pairs[0][0], weights=w)   # warm up
            res, ts = [], []
            for im, gt, name in pairs:
                t = time.time()
                m = M.person_mask_seg(im, weights=w) > 0
                ts.append(time.time() - t)
                res.append((name, st(m, gt, roi), st(m, gt)))
            roi_iou = [r[1][0] for r in res]
            full_iou = [r[2][0] for r in res]
            lines.append(f'{Path(w).name:16s} {sel:5s} {conf}: ROI IoU mean {np.mean(roi_iou):.3f} min {min(roi_iou):.3f} recall min {min(r[1][1] for r in res):.3f} | '
                         f'full IoU mean {np.mean(full_iou):.3f} min {min(full_iou):.3f} | {1000 * np.mean(ts):.0f} ms/frame')
            lines.append('    per frame ROI IoU: ' + ', '.join(f'{r[0]} {r[1][0]:.3f}' for r in res))
    text = '\n'.join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
