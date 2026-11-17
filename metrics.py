"""metrics.py - quantitative evaluation: FWHM, sidelobes, contrast ratio, CNR, speckle SNR."""
import numpy as np


def _crossings(axis, prof_db, level=-6.0):
    """Interpolated positions of the left/right crossings of `level` around the global maximum."""
    k = int(np.argmax(prof_db))
    l = k
    while l > 0 and prof_db[l] > level:
        l -= 1
    r = k
    while r < len(prof_db) - 1 and prof_db[r] > level:
        r += 1
    if prof_db[l] > level or prof_db[r] > level:
        return np.nan, np.nan
    xl = np.interp(level, [prof_db[l], prof_db[l + 1]], [axis[l], axis[l + 1]])
    xr = np.interp(level, [prof_db[r], prof_db[r - 1]], [axis[r], axis[r - 1]])
    return xl, xr


def fwhm(axis, prof_db, level=-6.0):
    xl, xr = _crossings(axis, prof_db, level)
    return xr - xl


def peak_sidelobe_db(axis, prof_db):
    """Highest level outside the main lobe (main lobe ends at the first local minimum on each side)."""
    k = int(np.argmax(prof_db))
    l = k
    while l > 0 and prof_db[l - 1] < prof_db[l]:
        l -= 1
    r = k
    while r < len(prof_db) - 1 and prof_db[r + 1] < prof_db[r]:
        r += 1
    side = np.r_[prof_db[:l], prof_db[r + 1:]]
    return float(np.max(side)) if side.size else -np.inf


def disc_mask(X, Z, cx, cz, r0, r1=None):
    d = np.hypot(X - cx, Z - cz)
    return (d <= r0) if r1 is None else ((d >= r0) & (d <= r1))


def contrast_metrics(env, X, Z, cyst, bg_mask):
    """env: linear envelope image. cyst=(cx,cz,r). Inner ROI = 0.8 r disc; background ROI given."""
    cx, cz, r = cyst
    inner = disc_mask(X, Z, cx, cz, 0.8 * r)
    mi, mb = env[inner].mean(), env[bg_mask].mean()
    si, sb = env[inner].std(), env[bg_mask].std()
    return {"CR_dB": 20 * np.log10(mi / mb), "CNR": abs(mi - mb) / np.sqrt(si ** 2 + sb ** 2)}


def speckle_snr(env, mask):
    v = env[mask]
    return v.mean() / v.std()


def speckle_snr_blocks(env, mask, X, Z, size=3e-3):
    """Mean over blocks (size x size) fully inside `mask` of mean/std. Removes the effect of slow
    amplitude trends with depth/lateral position that would bias a single global mean/std."""
    bi = np.floor(X / size).astype(int)
    bj = np.floor(Z / size).astype(int)
    snrs = []
    for i in np.unique(bi[mask]):
        for j in np.unique(bj[mask]):
            blk = (bi == i) & (bj == j)
            if blk.sum() > 0 and mask[blk].mean() > 0.98:
                v = env[blk]
                snrs.append(v.mean() / v.std())
    return float(np.mean(snrs))
