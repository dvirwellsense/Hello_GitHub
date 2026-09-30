from common import *; from gantry import *; from free_ba import free_ba
g = np.load('gantry_fit.npy'); sign, cam, lam = g[0], g[1:7], g[7]
obs = np.load('tracks_sweep.npy'); lab = dict(np.load('tracks_sweep_lab.npy', allow_pickle=True))
tid = obs[:, 0].astype(int); fr_all = obs[:, 1].astype(int)
W = 8
print('window  enc_model(bg)  free_pose(bg) | enc_model(ph)  free_pose(ph)   angle_step')
for s in range(44, 104, 6):
    fs = np.arange(s, s + W)
    inw = np.isin(fr_all, fs)
    cnt = np.bincount(tid[inw], minlength=tid.max() + 1)
    full = np.where(cnt == W)[0]
    out = []
    for kind in ('bg', 'ph'):
        ts = [t for t in full if lab.get(t) == kind]
        if kind == 'bg': ts = list(np.random.default_rng(0).choice(ts, min(400, len(ts)), replace=False))
        o = obs[inw & np.isin(tid, ts)]
        u, pid = np.unique(o[:, 0].astype(int), return_inverse=True); fr = o[:, 1].astype(int); uv = undistort_pts(o[:, 2:4])
        r_enc, X0, _ = residuals(cam, lam, sign, pid, uv, a_before[fr], a_after[fr], len(u))
        # refine sync inside window only (lam) to see if timing explains error
        best = min((np.median(np.linalg.norm(residuals(cam, l, sign, pid, uv, a_before[fr], a_after[fr], len(u))[0], axis=1)), l) for l in np.linspace(-0.5, 1.5, 21))
        frames, fidx = np.unique(fr, return_inverse=True)
        P0 = Pmats(cam, theta(a_before[frames], a_after[frames], lam, sign))
        if kind == 'bg':
            poses, X, r_free = free_ba(pid, fidx, uv, P0, X0, fixed=(0,), iters=60)
            bg_poses = poses
        else:
            # phantom: poses taken from background free BA (image-based ego-motion), only points triangulated
            Pp = np.array([np.c_[Rot.from_rotvec(p[:3]).as_matrix(), p[3:]] for p in bg_poses])[fidx]
            X = triangulate(Pp, pid, uv, len(u)); q, _ = reproject(Pp, X[pid]); r_free = np.linalg.norm((q - uv) * F_PX, axis=1)
        out += [np.median(np.linalg.norm(r_enc, axis=1)), best[0], best[1], np.median(r_free), len(u)]
    print('%3d-%3d  enc %.2f (best-lam %.2f @%.1f)  free %.2f  n=%d | ph: enc %.2f (best-lam %.2f)  free %.2f n=%d   %.2f deg' % (
        s + 1, s + W, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[8], out[9], np.mean(np.diff(angle[fs]))))
