"""phantoms.py - virtual phantoms as scatterer lists (x, z, rc) in metres."""
import numpy as np
from . import config as cfg


def point_phantom():
    x, z = cfg.POINT_POS
    return np.array([x]), np.array([z]), np.array([1.0])


def _background(rng, xr, zr, density_per_mm2, rc_scale):
    area = (xr[1] - xr[0]) * 1e3 * (zr[1] - zr[0]) * 1e3
    n = int(density_per_mm2 * area)
    x = rng.uniform(xr[0], xr[1], n)
    z = rng.uniform(zr[0], zr[1], n)
    rc = rng.standard_normal(n) * rc_scale
    return x, z, rc


def wire_phantom(with_background=True):
    rng = np.random.default_rng(cfg.SEED)
    xs = [0.0] * len(cfg.WIRE_Z) + [-cfg.CLOSE_PAIR_DX, cfg.CLOSE_PAIR_DX]
    zs = list(cfg.WIRE_Z) + [cfg.CLOSE_PAIR_Z] * 2
    x = np.array(xs)
    z = np.array(zs)
    rc = np.ones(len(x))
    if with_background:
        bx, bz, brc = _background(rng, (-10e-3, 10e-3), (6e-3, 34e-3),
                                  cfg.WIRE_BG_DENSITY_PER_MM2, cfg.WIRE_BG_RC)
        x, z, rc = np.r_[x, bx], np.r_[z, bz], np.r_[rc, brc]
    return x, z, rc


def contrast_phantom(density=None):
    rng = np.random.default_rng(cfg.SEED)
    density = cfg.SCAT_PER_MM2 if density is None else density
    x, z, rc = _background(rng, cfg.CONTRAST_X, cfg.CONTRAST_Z, density, 1.0)
    for (cx, cz, r), f in ((cfg.ANECHOIC, cfg.ANECHOIC_FACTOR), (cfg.HYPER, cfg.HYPER_FACTOR)):
        inside = (x - cx) ** 2 + (z - cz) ** 2 < r ** 2
        rc[inside] *= f
    return x, z, rc
