"""Compare automatic body masks (YOLOv8 top / union) with a hand-drawn body mask on one frame, and score the same
static pair with each mask.

Human masks come from the other chat's annotation tool (policy: skin, clothing and worn shoes; no bed or reflections),
1280x720 PNG in raw image coordinates. Reports IoU / recall / precision over the whole image and inside the detector ROI,
and the pair score (k=8/16/30) of frame a -> a+1 computed with each mask (mine as used, human eroded 21 px, human raw).

  python scripts/compare_masks_with_human.py --scan <scan_folder> --frame 53 --human mask.png --my-masks <masks_dir> --out report.txt
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import motion.masks as M  # noqa: E402
from motion.calib import Calibration  # noqa: E402
from motion.gantry import GantryModel  # noqa: E402
from motion.rigidity import evaluate_pair  # noqa: E402
from motion.roi import DetectorROI  # noqa: E402
from motion.scan import Scan  # noqa: E402


def stats(pred, gt, reg=None):
    if reg is not None:
        pred, gt = pred & reg, gt & reg
    i, u = (pred & gt).sum(), (pred | gt).sum()
    return f'IoU {i / u:.3f} recall {i / gt.sum():.3f} precision {i / pred.sum():.3f} leak_px {(pred & ~gt).sum()} miss_px {(gt & ~pred).sum()}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--frame', type=int, required=True)
    ap.add_argument('--human', required=True)
    ap.add_argument('--my-masks', required=True, help='folder with body_<frame>.png (eroded masks as used in the results)')
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--roi', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--out')
    a = ap.parse_args()
    lines = []
    calib, gantry, scan = Calibration(a.calib), GantryModel.load(a.gantry), Scan(a.scan)
    roi = DetectorROI(a.roi, calib)
    roi_px = cv2.imread(a.roi, 0) > 0
    i = [k for k in range(1, len(scan)) if int(scan.index[k - 1]) == a.frame][0]
    img = scan.image(i - 1)
    gt = cv2.imread(a.human, 0) > 0
    for name, sel, conf in (('top 0.25', 'top', 0.25), ('union 0.1', 'union', 0.1)):
        M.SEG_SELECT, M.SEG_CONF = sel, conf
        m = M.person_mask_seg(img) > 0
        lines.append(f'YOLOv8m {name}: full image {stats(m, gt)}')
        lines.append(f'YOLOv8m {name}: detector ROI {stats(m, gt, roi_px)}')
    er = lambda x: cv2.erode(x, np.ones((21, 21), np.uint8))
    mine = cv2.imread(str(Path(a.my_masks) / f'body_{a.frame:06d}.png'), 0)
    human = gt.astype(np.uint8) * 255
    region = roi.mask.astype(np.uint8) * 255
    for name, mask in (('mine (as used, eroded 21)', mine), ('human, eroded 21', er(human)), ('human, raw', human)):
        r = evaluate_pair(scan.image(i - 1), scan.image(i), scan.angle(i - 1, 0.9), scan.angle(i, 0.9), mask, np.zeros_like(mask), calib,
                          gantry, roi=roi, detect_mode='masked_abs', roi_region=region, body_fb_max=0.1)
        lines.append(f'pair {a.frame}->{a.frame + 1} with {name:28s}: body points {r["n_body"]}, score k8 {r["epi_enc_cl"]:.2f} k16 {r["epi_enc_cl16"]:.2f} k30 {r["epi_enc_cl30"]:.2f}')
    text = '\n'.join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
