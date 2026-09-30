from common import *
f = K[0, 0]
def to_px(n): return n * f  # undistorted pixels (calibration frame scale), offset irrelevant for residuals
res = []
for i in range(1, len(files)):
    da = angle[i] - angle[i - 1]
    if abs(da) < 0.2: continue
    im0 = cv2.imread(files[i - 1]); g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY); g1 = gray(i)
    pm = phantom_mask(im0); sp = specular_mask(im0) & pm
    p0 = cv2.goodFeaturesToTrack(g0, 3000, 0.005, 7)
    p1, ok = track(g0, g1, p0)
    p0, p1 = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
    xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
    on_ph = (pm[yi, xi] > 0) & (sp[yi, xi] == 0)
    bg = cv2.dilate(pm, np.ones((41, 41), np.uint8))[yi, xi] == 0
    n0, n1 = to_px(undistort_pts(p0)), to_px(undistort_pts(p1))
    # (a) current method: 2D similarity from background, applied to phantom
    A, inl = cv2.estimateAffinePartial2D(n0[bg], n1[bg], method=cv2.RANSAC, ransacReprojThreshold=1.0)
    ra = np.linalg.norm(cv2.transform(n0[on_ph].reshape(1, -1, 2), A).reshape(-1, 2) - n1[on_ph], axis=1)
    rab = np.linalg.norm(cv2.transform(n0[bg].reshape(1, -1, 2), A).reshape(-1, 2) - n1[bg], axis=1)
    # (b) epipolar: fundamental from background only, distance of phantom points to epipolar lines
    F, finl = cv2.findFundamentalMat(n0[bg], n1[bg], cv2.FM_RANSAC, 0.5, 0.999)
    def epi(a, b):
        l = (F @ np.c_[a, np.ones(len(a))].T).T
        return np.abs(np.sum(l * np.c_[b, np.ones(len(b))], 1)) / np.hypot(l[:, 0], l[:, 1])
    re = epi(n0[on_ph], n1[on_ph])
    flow = np.median(np.linalg.norm(n1[on_ph] - n0[on_ph], axis=1))
    res.append((i + 1, da, on_ph.sum(), bg.sum(), flow, np.median(rab), np.median(ra), np.percentile(ra, 90), np.median(re), np.percentile(re, 90)))
res = np.array(res)
np.save('pairwise.npy', res)
print('frame dAng nPh nBg  flow  aff_bg  aff_ph_med aff_ph_p90 | epi_ph_med epi_ph_p90')
for r in res[::4]: print('%3d %+5.2f %4d %4d %6.2f %6.2f %8.2f %8.2f | %8.2f %8.2f' % tuple(r))
print('MEDIAN over pairs: affine phantom residual %.2f px, epipolar phantom residual %.2f px' % (np.median(res[:, 6]), np.median(res[:, 8])))
