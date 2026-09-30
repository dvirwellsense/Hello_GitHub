"""Per adjacent pair: current method (2D similarity from background) vs epipolar residual (image-based F, and encoder-model E)."""
import sys, csv, glob, numpy as np, cv2
from calib import undistort_pts, K
from gantry import Pmats, theta, F_PX
from body import person_mask
scan, kind, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(scan + '/frames.csv', encoding='utf-8-sig')))
files = [scan + '/images/' + r['image_file'].split('\\')[-1] for r in rows]
a0 = np.array([float(r['arm_angle_before_capture']) for r in rows]); a1 = np.array([float(r['arm_angle_after_capture']) for r in rows])
g = np.load('gantry_fit.npy'); sign, cam = g[0], g[1:7]; lam = 0.9
if kind == 'phantom':
    from common import phantom_mask as body_mask, specular_mask
else:
    body_mask = person_mask; specular_mask = lambda im: np.zeros(im.shape[:2], np.uint8)
LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
def epi_dist(Fm, a, b):
    l = np.c_[a, np.ones(len(a))] @ Fm.T
    return np.abs(np.sum(l * np.c_[b, np.ones(len(b))], 1)) / np.hypot(l[:, 0], l[:, 1])
def cluster_max(pts, e, k=8):
    if len(e) < k: return np.nan
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2); nn = np.argsort(D, 1)[:, :k]
    return float(np.max(np.median(e[nn], axis=1)))
res = []
prev = cv2.imread(files[0]); g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
for i in range(1, len(files)):
    cur = cv2.imread(files[i]); g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    da = (a0[i] + lam * (a1[i] - a0[i])) - (a0[i - 1] + lam * (a1[i - 1] - a0[i - 1]))
    bm = body_mask(prev); sp = specular_mask(prev) & bm
    p0 = cv2.goodFeaturesToTrack(g0, 3000, 0.005, 7)
    p1, s, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK)
    pb, sb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
    ok = (s.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.5)
    p0, p1 = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
    onb = (bm[yi, xi] > 0) & (sp[yi, xi] == 0)
    bg = cv2.dilate(bm, np.ones((81, 81), np.uint8))[yi, xi] == 0
    n0, n1 = undistort_pts(p0) * F_PX, undistort_pts(p1) * F_PX
    A, _ = cv2.estimateAffinePartial2D(n0[bg], n1[bg], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    ra = np.linalg.norm(cv2.transform(n0[onb].reshape(1, -1, 2), A).reshape(-1, 2) - n1[onb], axis=1)
    Fi, _ = cv2.findFundamentalMat(n0[bg], n1[bg], cv2.FM_RANSAC, 0.5, 0.999)
    ei = epi_dist(Fi, n0[onb], n1[onb]) if Fi is not None and Fi.shape == (3, 3) else np.full(onb.sum(), np.nan)
    # encoder-model essential matrix
    Pa, Pb = Pmats(cam, theta(a0[[i - 1, i]], a1[[i - 1, i]], lam, sign))
    Rab = Pb[:, :3] @ Pa[:, :3].T; tab = Pb[:, 3] - Rab @ Pa[:, 3]
    E = np.array([[0, -tab[2], tab[1]], [tab[2], 0, -tab[0]], [-tab[1], tab[0], 0]]) @ Rab
    Kn = np.diag([1 / F_PX, 1 / F_PX, 1]); Fe = Kn.T @ E @ Kn
    ee = epi_dist(Fe, n0[onb], n1[onb])
    eb = epi_dist(Fe, n0[bg], n1[bg])
    res.append(dict(frame=i + 1, dangle=da, n_body=int(onb.sum()), n_bg=int(bg.sum()),
                    body_flow=float(np.median(np.linalg.norm(n1[onb] - n0[onb], axis=1))),
                    affine_med=float(np.median(ra)), affine_cl=cluster_max(n0[onb], ra),
                    epi_img_med=float(np.nanmedian(ei)), epi_img_cl=cluster_max(n0[onb], ei),
                    epi_enc_med=float(np.median(ee)), epi_enc_cl=cluster_max(n0[onb], ee), epi_enc_bg=float(np.median(eb))))
    prev, g0 = cur, g1
    if i % 25 == 0: print(i, flush=True)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, res[0].keys()); w.writeheader(); w.writerows(res)
