from common import *; from gantry import *
import time
obs = np.load('tracks_sweep.npy'); lab = dict(np.load('tracks_sweep_lab.npy', allow_pickle=True))
rng = np.random.default_rng(0); tid = obs[:, 0].astype(int); lens = np.bincount(tid)
bg = [t for t, l in lab.items() if l == 'bg' and lens[t] >= 25]
sel = rng.choice(bg, 600, replace=False)
m = np.isin(tid, sel); o = obs[m]
u, pid = np.unique(o[:, 0].astype(int), return_inverse=True); fr = o[:, 1].astype(int); uv = undistort_pts(o[:, 2:4])
best = None
for sign in (1, -1):
    for tilt in (-20, 0, 20):
        t0 = time.time()
        cam, lam, X, r = fit(pid, uv, a_before[fr], a_after[fr], len(u), init_cam(1.0, tilt), sign)
        print('sign %+d tilt %+d: median %.3f px p90 %.3f  lam %.2f  (%.0fs)' % (sign, tilt, np.median(r), np.percentile(r, 90), lam, time.time() - t0), flush=True)
        if best is None or np.median(r) < best[0]: best = (np.median(r), sign, cam, lam)
np.save('gantry_fit.npy', np.r_[best[1], best[2], best[3]])
print('best sign', best[1], 'cam center(world)', cam_center(best[2]), 'lam', best[3])
