"""Free bundle adjustment: independent pose per frame (no gantry model). Gives the floor achievable by tracking noise."""
from common import *; from gantry import *
from scipy.sparse import coo_matrix
def free_ba(pid, fidx, uv, P0, X0, fixed=(0,), iters=100):
    nF, nP = len(P0), len(X0)
    poses0 = np.array([np.r_[Rot.from_matrix(P[:, :3]).as_rotvec(), P[:, 3]] for P in P0])
    free = np.array([f for f in range(nF) if f not in fixed])
    def unpack(p):
        poses = poses0.copy(); poses[free] = p[:6 * len(free)].reshape(-1, 6)
        return poses, p[6 * len(free):].reshape(-1, 3)
    def res(p):
        poses, X = unpack(p)
        R = Rot.from_rotvec(poses[fidx, :3]).apply(X[pid]) + poses[fidx, 3:]
        return ((R[:, :2] / R[:, 2:3] - uv) * F_PX).ravel()
    n = len(uv); col_of = -np.ones(nF, int); col_of[free] = np.arange(len(free)) * 6
    r_i, c_i = [], []
    for k in range(2):
        rows = 2 * np.arange(n) + k
        hasp = col_of[fidx] >= 0
        for d in range(6): r_i.append(rows[hasp]); c_i.append(col_of[fidx][hasp] + d)
        for d in range(3): r_i.append(rows); c_i.append(6 * len(free) + 3 * pid + d)
    r_i, c_i = np.concatenate(r_i), np.concatenate(c_i)
    p0 = np.r_[poses0[free].ravel(), X0.ravel()]
    S = coo_matrix((np.ones(len(r_i)), (r_i, c_i)), shape=(2 * n, len(p0)))
    sol = least_squares(res, p0, jac_sparsity=S, loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=iters, method='trf')
    poses, X = unpack(sol.x)
    return poses, X, np.linalg.norm(res(sol.x).reshape(-1, 2), axis=1)
if __name__ == '__main__':
    g = np.load('gantry_fit.npy'); sign, cam, lam = g[0], g[1:7], g[7]
    obs = np.load('tracks_sweep.npy'); lab = dict(np.load('tracks_sweep_lab.npy', allow_pickle=True))
    rng = np.random.default_rng(0); tid = obs[:, 0].astype(int); lens = np.bincount(tid)
    bg = [t for t, l in lab.items() if l == 'bg' and lens[t] >= 25]
    sel = rng.choice(bg, 600, replace=False)
    o = obs[np.isin(tid, sel)]
    u, pid = np.unique(o[:, 0].astype(int), return_inverse=True); fr = o[:, 1].astype(int); uv = undistort_pts(o[:, 2:4])
    frames, fidx = np.unique(fr, return_inverse=True)
    th = theta(a_before[frames], a_after[frames], lam, sign)
    P0 = Pmats(cam, th)
    r_g, X0, _ = residuals(cam, lam, sign, pid, uv, a_before[fr], a_after[fr], len(u))
    poses, X, r = free_ba(pid, fidx, uv, P0, X0, fixed=(0, len(frames) - 1))
    print('gantry model median %.3f px | free per-frame BA median %.3f px p90 %.3f' % (np.median(np.linalg.norm(r_g, axis=1)), np.median(r), np.percentile(r, 90)))
    np.savez('free_ba.npz', frames=frames, poses=poses, X=X, sel=u)
    # recover the effective rotation angle of each free pose relative to gantry model axis
    R0 = Rot.from_rotvec(cam[:3]).as_matrix()
    ang = []
    for p in poses:
        Rf = Rot.from_rotvec(p[:3]).as_matrix(); M = R0.T @ Rf   # should be Rz(-th)
        ang.append(-np.degrees(np.arctan2(M[1, 0], M[0, 0])) * sign)
    ang = np.array(ang)
    for f, a, e in zip(frames[::3], ang[::3], (a_before[frames] + lam * (a_after[frames] - a_before[frames]))[::3]):
        print('frame %3d  image-angle %+7.2f  encoder-angle %+7.2f  diff %+5.2f' % (f + 1, a, e, a - e))
