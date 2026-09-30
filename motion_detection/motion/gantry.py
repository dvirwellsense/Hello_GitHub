"""Gantry camera model: the camera is rigidly attached to the arc, which rotates about a fixed axis.

World frame: z = rotation axis. At gantry angle th a world point X maps to camera coordinates
    x_cam = R_c Rz(-sign*th) X + t_c
Unknowns: camera extrinsics (R_c, t_c) and the sign convention. Scale is fixed by setting the
camera-to-axis distance to 1 (epipolar checks are scale free).
"""
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    z, o = np.zeros_like(t), np.ones_like(t)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class GantryModel:
    def __init__(self, cam, sign=1.0):
        self.cam = np.asarray(cam, float)  # rotvec(3) + t(3)
        self.sign = float(sign)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d['cam'], d['sign'])

    def save(self, path, **extra):
        json.dump(dict(cam=self.cam.tolist(), sign=self.sign, **extra), open(path, 'w'), indent=2)

    def projections(self, angles_deg):
        """3x4 camera matrices (normalized coordinates) for the given gantry angles."""
        th = np.radians(self.sign * np.atleast_1d(angles_deg))
        R = Rot.from_rotvec(self.cam[:3]).as_matrix()
        M = np.einsum('ij,njk->nik', R, Rz(-th))
        return np.concatenate([M, np.broadcast_to(self.cam[3:6, None], (len(th), 3, 1))], 2)

    def essential(self, angle_a, angle_b):
        """Essential matrix mapping normalized points of view a to epipolar lines in view b."""
        Pa, Pb = self.projections([angle_a, angle_b])
        R = Pb[:, :3] @ Pa[:, :3].T
        t = Pb[:, 3] - R @ Pa[:, 3]
        return skew(t) @ R

    def camera_center(self):
        return -Rot.from_rotvec(self.cam[:3]).as_matrix().T @ self.cam[3:6]


def initial_camera(tilt_deg=0.0):
    """Camera looking at the axis from distance 1, with the axis along the image x direction."""
    R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    R = Rot.from_euler('x', tilt_deg, degrees=True).as_matrix() @ R
    return np.r_[Rot.from_matrix(R).as_rotvec(), -R @ np.array([1.0, 0, 0])]


def triangulate(P, pid, uv, npts):
    """Linear triangulation. P: (n,3,4) per observation, pid: point id per observation, uv: normalized."""
    A1 = uv[:, 0:1] * P[:, 2] - P[:, 0]
    A2 = uv[:, 1:2] * P[:, 2] - P[:, 1]
    AtA = np.zeros((npts, 4, 4))
    for A in (A1, A2):
        np.add.at(AtA, pid, A[:, :, None] * A[:, None, :])
    _, v = np.linalg.eigh(AtA)
    Xh = v[:, :, 0]
    return Xh[:, :3] / Xh[:, 3:4]


def reproject(P, X):
    x = np.einsum('nij,nj->ni', P, np.c_[X, np.ones(len(X))])
    return x[:, :2] / x[:, 2:3]


def reprojection_residuals(model, angles, pid, uv, npts, focal_px):
    P = model.projections(angles)
    X = triangulate(P, pid, uv, npts)
    return (reproject(P, X[pid]) - uv) * focal_px, X


def fit_gantry(pid, uv, angles, npts, focal_px, sign, cam0):
    """Fit camera extrinsics to static-scene tracks (variable projection: points re-triangulated each step).
    Gauge constraints: camera center at th=0 on the world x axis, at distance 1 from the axis."""
    def res(p):
        m = GantryModel(p, sign)
        r, _ = reprojection_residuals(m, angles, pid, uv, npts, focal_px)
        c0 = m.camera_center()
        return np.r_[r.ravel(), 1e3 * c0[1], 1e3 * (np.hypot(c0[0], c0[1]) - 1), 1e3 * c0[2]]
    sol = least_squares(res, cam0, loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=300)
    model = GantryModel(sol.x, sign)
    r, X = reprojection_residuals(model, angles, pid, uv, npts, focal_px)
    return model, X, np.linalg.norm(r, axis=1)
