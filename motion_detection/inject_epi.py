"""Sensitivity: inject synthetic 'limb' motion into phantom tracks inside 8-frame windows, score with encoder-model rigidity residual."""
from common import *; from gantry import *
g = np.load('gantry_fit.npy'); sign, cam = g[0], g[1:7]; lam = 0.9
obs = np.load('tracks_sweep.npy'); lab = dict(np.load('tracks_sweep_lab.npy', allow_pickle=True))
tid = obs[:, 0].astype(int); fr_all = obs[:, 1].astype(int)
W, KMOVE = 8, 5  # motion happens before frame index 5 of the window
rng = np.random.default_rng(0)
def score_epi(pid, fr, uv, n, fs):
    """Encoder-pose epipolar distance between the two frames around the motion instant, spatially coherent."""
    fa, fb = fs[0] + KMOVE - 1, fs[0] + KMOVE
    Pa, Pb = Pmats(cam, theta(a_before[[fa, fb]], a_after[[fa, fb]], lam, sign))
    Rab = Pb[:, :3] @ Pa[:, :3].T; tab = Pb[:, 3] - Rab @ Pa[:, 3]
    E = np.array([[0, -tab[2], tab[1]], [tab[2], 0, -tab[0]], [-tab[1], tab[0], 0]]) @ Rab
    ia, ib = fr == fa, fr == fb
    xa = np.zeros((n, 2)); xb = np.zeros((n, 2)); xa[pid[ia]] = uv[ia]; xb[pid[ib]] = uv[ib]
    l = np.c_[xa, np.ones(n)] @ E.T
    e = np.abs(np.sum(l * np.c_[xb, np.ones(n)], 1)) / np.hypot(l[:, 0], l[:, 1]) * F_PX
    D = np.linalg.norm(xa[:, None] - xa[None], axis=2); nn = np.argsort(D, 1)[:, :8]
    return np.max(np.median(e[nn], axis=1))
def score(pid, fr, uv, n, fs):
    """Real-time style: triangulate from the first KMOVE frames (history), predict the newest frame, spatially-coherent error."""
    P = Pmats(cam, theta(a_before[fr], a_after[fr], lam, sign))
    hist = fr < fs[0] + KMOVE
    X = triangulate(P[hist], pid[hist], uv[hist], n)
    new = fr == fs[-1]
    q, _ = reproject(P[new], X[pid[new]])
    e = np.zeros(n); pts = np.zeros((n, 2))
    e[pid[new]] = np.linalg.norm((q - uv[new]) * F_PX, axis=1); pts[pid[new]] = uv[new]
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2); nn = np.argsort(D, 1)[:, :8]
    return np.max(np.median(e[nn], axis=1))
results = {}
score = score_epi
for s in range(44, 104, 3):
    fs = np.arange(s, s + W); inw = np.isin(fr_all, fs)
    cnt = np.bincount(tid[inw], minlength=tid.max() + 1)
    ts = [t for t in np.where(cnt == W)[0] if lab.get(t) == 'ph']
    o = obs[inw & np.isin(tid, ts)]
    u, pid = np.unique(o[:, 0].astype(int), return_inverse=True); fr = o[:, 1].astype(int)
    xy = o[:, 2:4].copy()
    results.setdefault(0.0, []).append(score(pid, fr, undistort_pts(xy), len(u), fs))
    # 'limb' = 30% of phantom points nearest to a random phantom point
    for mag_mm in (0.5, 1.0, 2.0, 3.0):
        for rep in range(4):
            c = xy[rng.integers(len(xy))]
            first_xy = np.array([xy[pid == k][0] for k in range(len(u))])
            limb = np.argsort(np.linalg.norm(first_xy - c, axis=1))[:max(3, int(0.3 * len(u)))]
            d = rng.normal(size=2); d /= np.linalg.norm(d)
            shift = d * mag_mm * 0.448 * 2   # mm -> full-res px at bed plane (0.448 px/mm at half res)
            xy2 = xy.copy(); mv = np.isin(pid, limb) & (fr >= s + KMOVE); xy2[mv] += shift
            results.setdefault(mag_mm, []).append(score(pid, fr, undistort_pts(xy2), len(u), fs))
base = np.array(results[0.0]); thr = base.max() * 1.2
print('static windows: score mean %.2f max %.2f -> threshold %.2f px (half-res)' % (base.mean(), base.max(), thr))
for m in (0.5, 1.0, 2.0, 3.0):
    v = np.array(results[m]); print('limb shift %.1f mm (image-plane, random dir): detected %3.0f%%  (median score %.2f)' % (m, 100 * np.mean(v > thr), np.median(v)))
