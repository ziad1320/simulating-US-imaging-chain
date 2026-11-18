"""beamformer.py - custom delay-and-sum (DAS) beamformer with arbitrary receive weights.

Geometry (Eqs. 6-9 of the proposal):
  t_tx(x,z)   = (x sin(th) + z cos(th))/c + t_off(th)
  t_rx,i(x,z) = sqrt((x - x_i)^2 + z^2)/c
  y(x,z)      = sum_i w_i s_i(t_tx + t_rx,i)            (summed over angles BEFORE the envelope)
t_off(th) = -min_i(x_i sin th)/c is the constant that makes all MUST transmit delays positive
(the first element to fire defines t = 0). Verified against pymust.txdelay in 06_validate.py.
"""
import numpy as np
from scipy.signal import hilbert
from . import config as cfg


def subaperture_start(x0, n_sub=cfg.N_SUB):
    """Index of the first element of the n_sub-element window centred on x0 (clamped at the array edges)."""
    i0 = int(np.floor(x0 / cfg.PITCH + (cfg.N_EL - 1) / 2)) - (n_sub // 2 - 1)
    return int(np.clip(i0, 0, cfg.N_EL - n_sub))


def beamform(rf, angles, x_lines, z_grid, weights, n_sub=cfg.N_SUB, fs=cfg.FS, c=cfg.C):
    """rf: (n_angles, n_t, n_el) ; weights: (n_w, n_sub) or (n_sub,).
    Returns the coherently compounded RF image, shape (n_w, n_z, n_lines) (float64)."""
    weights = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    assert weights.shape[1] == n_sub
    xe = cfg.element_x()
    n_ang, n_t, _ = rf.shape
    out = np.zeros((weights.shape[0], len(z_grid), len(x_lines)))
    for a, th in enumerate(np.atleast_1d(angles)):
        s, co = np.sin(th), np.cos(th)
        t_off = -np.min(xe * s) / c
        for j, x0 in enumerate(x_lines):
            i0 = subaperture_start(x0, n_sub)
            xs = xe[i0:i0 + n_sub]
            ttx = (x0 * s + z_grid * co) / c + t_off                       # (n_z,)
            trx = np.sqrt((x0 - xs[None, :]) ** 2 + z_grid[:, None] ** 2) / c  # (n_z, n_sub)
            k = (ttx[:, None] + trx) * fs                                   # fractional sample index
            k0 = np.floor(k).astype(np.int64)
            fr = k - k0
            ok = (k0 >= 0) & (k0 < n_t - 1)
            k0 = np.clip(k0, 0, n_t - 2)
            ch = np.arange(i0, i0 + n_sub)[None, :]
            r = rf[a]
            v = r[k0, ch] * (1 - fr) + r[k0 + 1, ch] * fr                   # linear interpolation
            v *= ok
            out[:, :, j] += (v @ weights.T).T
    return out


def lines_for(x_min, x_max, upsample):
    """Scan-line positions: pitch/upsample spacing, inside the region where a full sub-aperture exists."""
    lim = ((cfg.N_EL - cfg.N_SUB) / 2) * cfg.PITCH + 0.5 * cfg.PITCH   # ~ +/-14.6 mm
    x_min, x_max = max(x_min, -lim), min(x_max, lim)
    dx = cfg.PITCH / upsample
    n = int(np.floor((x_max - x_min) / dx)) + 1
    return x_min + dx * np.arange(n)


def envelope(rf_img):
    """|analytic signal| along depth (axis 0). Works on (n_z, n_lines) or (n_w, n_z, n_lines)."""
    return np.abs(hilbert(rf_img, axis=-2))


def to_db(env, dr=cfg.DR_DB, ref=None):
    ref = np.max(env) if ref is None else ref
    return np.clip(20 * np.log10(env / ref + 1e-12), -dr, 0)


# ---------------- weights ----------------
def rect(n=cfg.N_SUB):
    return np.ones(n)


def nsi_weights(c=cfg.NSI_C, n=cfg.N_SUB):
    zm = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    dc1 = zm + c
    dc2 = dc1[::-1].copy()
    return zm, dc1, dc2


def nsi_image(rf, angles, x_lines, z_grid, c=cfg.NSI_C):
    """Returns (E_conv, E_NSI, E_ZM) envelopes from the SAME rf data (Eq. 15)."""
    zm, dc1, dc2 = nsi_weights(c)
    img = beamform(rf, angles, x_lines, z_grid, np.vstack([rect(), zm, dc1, dc2]))
    E = envelope(img)
    e_nsi = 0.5 * (E[2] + E[3]) - E[1]
    floor = 1e-4 * np.max(e_nsi)
    return E[0], np.maximum(e_nsi, floor), E[1]
