"""Independent analytic infinitesimal CIEDE2000 tensor and augmented LS solve.

Derived from Sharma et al. (2005) equations; no finite-difference implementation
or training normal-equation assembly is called by these reference functions.
"""
import numpy as np


def analytic_tensor(lab):
    lab = np.asarray(lab, np.float64)
    light, a, b = lab.T
    chroma = np.hypot(a, b)
    gain = 1 + .5 * (1 - np.sqrt(chroma ** 7 / (chroma ** 7 + 25. ** 7)))
    ap = gain * a
    cp = np.hypot(ap, b)
    hue = np.degrees(np.arctan2(b, ap)) % 360
    angular = (1 - .17 * np.cos(np.deg2rad(hue - 30)) + .24 * np.cos(np.deg2rad(2 * hue))
               + .32 * np.cos(np.deg2rad(3 * hue + 6)) - .20 * np.cos(np.deg2rad(4 * hue - 63)))
    sl = 1 + .015 * (light - 50) ** 2 / np.sqrt(20 + (light - 50) ** 2)
    sc, sh = 1 + .045 * cp, 1 + .015 * cp * angular
    rotation = -2 * np.sqrt(cp ** 7 / (cp ** 7 + 25. ** 7)) * np.sin(np.deg2rad(60 * np.exp(-((hue - 275) / 25) ** 2)))
    radial = np.divide(ap, cp, out=np.ones_like(cp), where=cp > 0)
    tangent = np.divide(b, cp, out=np.zeros_like(cp), where=cp > 0)
    jacobian = np.zeros((len(lab), 3, 3))
    jacobian[:, 0, 0] = 1 / sl
    jacobian[:, 1, 1] = gain * radial / sc
    jacobian[:, 1, 2] = tangent / sc
    jacobian[:, 2, 1] = -gain * tangent / sh
    jacobian[:, 2, 2] = radial / sh
    coupling = np.broadcast_to(np.eye(3), jacobian.shape).copy()
    coupling[:, 1, 2] = coupling[:, 2, 1] = rotation / 2
    return jacobian.transpose(0, 2, 1) @ coupling @ jacobian


def augmented_svd(z, target, metric, weights, alpha):
    rank = z.shape[1]
    left = np.linalg.cholesky(metric).transpose(0, 2, 1) * np.sqrt(weights)[:, None, None]
    # Rows (observation, transformed color); columns (basis, output color).
    design = (left[:, :, None, :] * z[:, None, :, None]).reshape(3 * len(z), 3 * rank)
    rhs = np.einsum("nij,nj->ni", left, target).ravel()
    design = np.vstack([design, np.sqrt(alpha) * np.eye(3 * rank)])
    rhs = np.concatenate([rhs, np.zeros(3 * rank)])
    return np.linalg.lstsq(design, rhs, rcond=None)[0].reshape(rank, 3)
