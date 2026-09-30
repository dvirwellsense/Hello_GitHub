"""Gantry camera model. World z = rotation axis. At gantry angle th: x_cam ~ R_c Rz(-th) X + t_c.
Unknowns: cam (rotvec, t) + lam (sync: th = a_before + lam*(a_after-a_before)). Points triangulated linearly (variable projection)."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
F_PX = 467.0
def Rz(t):
    c, s = np.cos(t), np.sin(t); z, o = np.zeros_like(t), np.ones_like(t)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)
def Pmats(cam, th):
    R = Rot.from_rotvec(cam[:3]).as_matrix()
    M = np.einsum('ij,njk->nik', R, Rz(-th))
    return np.concatenate([M, np.broadcast_to(cam[3:6, None], (len(th), 3, 1))], 2)  # (n,3,4)
def cam_center(cam): return -Rot.from_rotvec(cam[:3]).as_matrix().T @ cam[3:6]
def init_cam(radius=1.0, tilt_deg=0.0):
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    R = Rot.from_euler('x', tilt_deg, degrees=True).as_matrix() @ R
    return np.r_[Rot.from_matrix(R).as_rotvec(), -R @ np.array([radius, 0, 0])]
def triangulate(P, pid, uv, npts):
    """Linear triangulation, P per observation (n,3,4), uv normalized. Returns X (npts,3)."""
    A1 = uv[:, 0:1] * P[:, 2] - P[:, 0]; A2 = uv[:, 1:2] * P[:, 2] - P[:, 1]
    AtA = np.zeros((npts, 4, 4))
    for A in (A1, A2): np.add.at(AtA, pid, A[:, :, None] * A[:, None, :])
    w, v = np.linalg.eigh(AtA); Xh = v[:, :, 0]
    return Xh[:, :3] / Xh[:, 3:4]
def reproject(P, X):
    x = np.einsum('nij,nj->ni', P, np.c_[X, np.ones(len(X))])
    return x[:, :2] / x[:, 2:3], x[:, 2]
def theta(a0, a1, lam, sign): return np.radians(sign * (a0 + lam * (a1 - a0)))
def residuals(cam, lam, sign, pid, uv, a0, a1, npts):
    P = Pmats(cam, theta(a0, a1, lam, sign)); X = triangulate(P, pid, uv, npts)
    q, z = reproject(P, X[pid])
    return (q - uv) * F_PX, X, z
def fit(pid, uv, a0, a1, npts, cam0, sign, lam0=0.5, fit_lam=True):
    def res(p):
        cam = p[:6]; lam = p[6] if fit_lam else lam0
        r, X, z = residuals(cam, lam, sign, pid, uv, a0, a1, npts)
        c0 = cam_center(cam)
        return np.r_[r.ravel(), 1e3 * c0[1], 1e3 * (np.hypot(c0[0], c0[1]) - 1), 1e3 * c0[2]]
    p0 = np.r_[cam0, lam0] if fit_lam else cam0
    sol = least_squares(res, p0, loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=300)
    cam = sol.x[:6]; lam = sol.x[6] if fit_lam else lam0
    r, X, z = residuals(cam, lam, sign, pid, uv, a0, a1, npts)
    return cam, lam, X, np.linalg.norm(r, axis=1)
