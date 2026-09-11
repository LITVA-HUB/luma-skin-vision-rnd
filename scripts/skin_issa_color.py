"""Reproduce ISSA workbook native colorimetry without inventing spectral tails."""
import numpy as np


def spectral_xyz(reflectance_percent, cmf, spd):
    r, c, s = (np.asarray(v, dtype=np.float64) for v in (reflectance_percent, cmf, spd))
    if not all(np.isfinite(v).all() for v in (r,c,s)):
        raise ValueError('Explicitly selected measured support must be finite')
    if r.ndim != 2 or c.shape != (r.shape[1],3) or s.shape != (r.shape[1],):
        raise ValueError('Spectral dimensions disagree')
    denominator = np.dot(c[:,1],s)
    if denominator <= 0:
        raise ValueError('Invalid white normalization')
    return r @ (c*s[:,None]) / denominator


def source_lab(xyz, white):
    q = np.asarray(xyz,dtype=np.float64)/np.asarray(white,dtype=np.float64)
    # Match the source's explicit rounded piecewise constants; do not substitute
    # another white, observer or implicit sRGB transform.
    f = np.where(q > .008856, np.cbrt(q), 7.787*q+16/116)
    light = np.where(q[...,1] > .008856,116*np.cbrt(q[...,1])-16,903.3*q[...,1])
    return np.stack([light,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)
