"""Does the RANSAC bed-plane consensus contain the points that are certainly ON the bed plane?

The white stickers lie on the board: corners detected on their edges (within 6 px of a sticker) are
on-plane anchors. For sampled frames, the bed-outline background points are split by a RANSAC plane
fit (1 px) over a 1-frame and a k-frame baseline; reported: acceptance of sticker-edge points vs other
board points vs head points (skin/hair outside the body mask).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.roi import DetectorROI
from motion.scan import Scan
from motion.tracking import detect, track


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--frames', required=True, help='first,last,step')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    calib = Calibration('data/CalibrationResult_arc22.json')
    scan = Scan(a.scan)
    f = calib.focal_px
    first, last, step = map(int, a.frames.split(','))
    rows = []

    def body(i):
        for j in (i, i - 1):
            m = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[j]):06d}.png'), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m

    for i in range(first, last + 1, step):
        for base in (1, a.k):
            A, B = scan.image(i - base), scan.image(i)
            ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (A, B))
            bm = body(i - base)
            p0 = detect(ga)
            p1, ok = track(ga, gb, p0, max_level=5)
            p0, p1 = p0[ok], p1[ok]
            xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
            bg = cv2.dilate(bm, np.ones((81, 81), np.uint8))[yi, xi] == 0
            hsv = cv2.cvtColor(A, cv2.COLOR_BGR2HSV)
            white = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, 190), (180, 50, 255)), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(white)
            stick = np.zeros_like(white)
            corners = []
            for c in range(1, n):
                x, y, w, h, ar = st[c]
                if 2500 < ar < 40000 and 0.5 < w / max(h, 1) < 2 and ar > 0.6 * w * h:
                    stick[lab == c] = 255
                    cnt, _ = cv2.findContours(np.where(lab == c, 255, 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    poly = cv2.approxPolyDP(max(cnt, key=cv2.contourArea), 0.04 * cv2.arcLength(max(cnt, key=cv2.contourArea), True), True)
                    corners += [q[0] for q in poly]
            near_stick = cv2.dilate(stick, np.ones((13, 13), np.uint8))[yi, xi] > 0
            # sticker CORNERS only (no aperture problem along straight edges)
            corners = np.array(corners, float).reshape(-1, 2)
            at_corner = (np.min(np.linalg.norm(p0[:, None] - corners[None], axis=2), axis=1) < 8) if len(corners) else np.zeros(len(p0), bool)
            # bed region at frame i-base: dark board + stickers (independent of the carried outline)
            dark = (hsv[..., 2] < 110).astype(np.uint8) * 255
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            nb, lb, sb, _ = cv2.connectedComponentsWithStats(dark)
            dark = np.where(lb == 1 + np.argmax(sb[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)  # the board only
            board = cv2.dilate(dark, np.ones((9, 9), np.uint8)) | cv2.dilate(stick, np.ones((13, 13), np.uint8))
            # head: skin-coloured (saturated, so not the grey floor), outside the body mask, on/next to the board
            skin = cv2.inRange(hsv, (0, 40, 70), (22, 200, 255))
            skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
            skin = cv2.dilate(skin, np.ones((21, 21), np.uint8)) & cv2.dilate(board, np.ones((61, 61), np.uint8))
            head = (skin[yi, xi] > 0) & ~near_stick
            inside = bg & ((board[yi, xi] > 0) | head)
            if inside.sum() < 10:
                continue
            u0, u1 = calib.undistort_points(p0), calib.undistort_points(p1)
            H, inl = cv2.findHomography(u0[inside], u1[inside], cv2.RANSAC, 1.0 / f)
            acc = np.zeros(len(u0), bool)
            acc[np.where(inside)[0][inl.ravel() > 0]] = True
            row = dict(frame=int(scan.index[i]), base=base, angle=float(scan.angle(i)), n_inside=int(inside.sum()))
            for k, sel in (('corner', inside & at_corner), ('sticker', inside & near_stick & ~at_corner), ('head', inside & head),
                           ('board', inside & ~near_stick & ~head)):
                row[f'n_{k}'] = int(sel.sum())
                row[f'acc_{k}'] = float(acc[sel].mean()) if sel.any() else np.nan
            rows.append(row)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / f'{a.name}_anchor.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for base in (1, a.k):
        r = [x for x in rows if x['base'] == base]
        g = lambda k: np.nanmedian([x[k] for x in r])
        s = lambda k: int(np.sum([x[k] for x in r]))
        print(f'{a.name} baseline {base}: accepted sticker-corner {100 * g("acc_corner"):.0f}% (n={s("n_corner")}), '
              f'sticker-edge {100 * g("acc_sticker"):.0f}% (n={s("n_sticker")}), '
              f'head {100 * g("acc_head"):.0f}% (n={s("n_head")}), other board {100 * g("acc_board"):.0f}% (n={s("n_board")}), frames {len(r)}')


if __name__ == '__main__':
    main()
