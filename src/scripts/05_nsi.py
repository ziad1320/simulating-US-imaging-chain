"""Step 5 (optional) - Null Subtraction Imaging on the SAME saved RF data (Eq. 14-15).
Usage: python src/scripts/05_nsi.py [point|wires|contrast]"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import config as cfg
from lib import beamformer as bf
from lib import metrics as mt

what = sys.argv[1:] or ["point", "wires", "contrast"]
METF = f"{cfg.DATA}/metrics_05.json"
M = json.load(open(METF)) if os.path.exists(METF) else {}


def load(name):
    d = np.load(f"{cfg.DATA}/{name}.npz")
    return d["rf"], d["angles"]


def show(ax, db, x, z, title, dr):
    ax.imshow(db, extent=[x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3], cmap="gray", vmin=-dr, vmax=0, aspect="equal")
    ax.set_title(title); ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")


if "point" in what:
    rf, ang = load("point")
    xl = bf.lines_for(-5e-3, 5e-3, 4 * cfg.LATERAL_UPSAMPLE)   # finer lines: the NSI beam is very narrow
    zg = np.arange(16e-3, 22e-3, cfg.DZ)
    res = {}
    fig, a = plt.subplots(1, 2, figsize=(12, 4.5))
    ec = None
    for c in cfg.NSI_C_SWEEP:
        e_conv, e_nsi, e_zm = bf.nsi_image(rf, ang, xl, zg, c)
        iz = np.unravel_index(np.argmax(e_conv), e_conv.shape)[0]
        if ec is None:
            lat_c = bf.to_db(e_conv, 80)[iz]
            a[0].plot(xl * 1e3, lat_c, "k", lw=2, label="conventional DAS")
            res["conventional"] = dict(fwhm_lat_mm=mt.fwhm(xl * 1e3, lat_c), psl_dB=mt.peak_sidelobe_db(xl * 1e3, lat_c))
            ec = True
        ln = bf.to_db(e_nsi, 80)[np.unravel_index(np.argmax(e_nsi), e_nsi.shape)[0]]
        a[0].plot(xl * 1e3, ln, label=f"NSI c={c}")
        res[f"NSI_c={c}"] = dict(fwhm_lat_mm=mt.fwhm(xl * 1e3, ln), psl_dB=mt.peak_sidelobe_db(xl * 1e3, ln),
                                 peak_vs_conv_dB=float(20 * np.log10(e_nsi.max() / e_conv.max())))
        if c == cfg.NSI_C:
            show(a[1], bf.to_db(e_nsi, 60), xl, zg, f"NSI point (c={c}, 60 dB)", 60)
    a[0].set_xlim(-3, 3); a[0].set_ylim(-80, 3); a[0].axhline(-6, c="r", lw=.5); a[0].legend()
    a[0].set_title("Lateral profile at the point depth"); a[0].set_xlabel("x [mm]")
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig08_nsi_point.png", dpi=140); plt.close(fig)
    M["point"] = res
    print("NSI POINT", json.dumps(res, indent=1))

if "wires" in what:
    rf, ang = load("wires")
    xz = np.arange(-2e-3, 2e-3, cfg.PITCH / 64)   # fine lateral sampling: NSI peaks are ~20 um wide
    zz = np.arange(16.5e-3, 19.5e-3, cfg.DZ)
    e_conv, e_nsi, _ = bf.nsi_image(rf, ang, xz, zz, cfg.NSI_C)
    out = {}
    fig, a = plt.subplots(1, 3, figsize=(16, 5))
    for ax_, (n, e) in zip(a[:2], (("conventional", e_conv), ("NSI", e_nsi))):
        db = bf.to_db(e, cfg.DR_DB_WIRES_CLOSE); show(ax_, db, xz, zz, f"{n} (40 dB)", cfg.DR_DB_WIRES_CLOSE)
        iz = np.argmax(e.max(axis=1)); pr = db[iz]
        out[n] = dict(valley_dB=float(pr[np.argmin(np.abs(xz))] - pr.max()))
        a[2].plot(xz * 1e3, pr, label=n)
    a[2].legend(); a[2].set_title("Pair lateral profile"); a[2].set_xlabel("x [mm]")
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig09_nsi_pair.png", dpi=140); plt.close(fig)
    M["wires"] = out
    print("NSI PAIR", json.dumps(out, indent=1))

if "contrast" in what:
    rf, ang = load("contrast")
    xl = bf.lines_for(-14e-3, 14e-3, cfg.LATERAL_UPSAMPLE_CONTRAST)
    zg = np.arange(10e-3, 40e-3, cfg.DZ)
    X, Z = np.meshgrid(xl, zg)
    e_conv, e_nsi, _ = bf.nsi_image(rf, ang, xl, zg, cfg.NSI_C)
    bg_patch = ((np.abs(X) < 12e-3) & (Z > 13e-3) & (Z < 37e-3)
                & (np.hypot(X - cfg.ANECHOIC[0], Z - cfg.ANECHOIC[1]) > 1.6 * cfg.ANECHOIC[2])
                & (np.hypot(X - cfg.HYPER[0], Z - cfg.HYPER[1]) > 1.6 * cfg.HYPER[2]))   # all cyst-free speckle
    res = {}
    fig, a = plt.subplots(1, 2, figsize=(12, 7))
    for ax_, (n, e) in zip(a, (("conventional DAS", e_conv), (f"NSI (c={cfg.NSI_C})", e_nsi))):
        show(ax_, bf.to_db(e, cfg.DR_DB), xl, zg, n, cfg.DR_DB)
        r_ = {}
        for lab, cyst in (("anechoic", cfg.ANECHOIC), ("hyperechoic", cfg.HYPER)):
            bgm = mt.disc_mask(X, Z, cyst[0], cyst[1], 1.3 * cyst[2], 1.9 * cyst[2])
            r_[lab] = mt.contrast_metrics(e, X, Z, cyst, bgm)
        r_["speckle_SNR"] = float(mt.speckle_snr_blocks(e, bg_patch, X, Z))
        res[n] = r_
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig10_nsi_contrast.png", dpi=140); plt.close(fig)
    M["contrast"] = res
    print("NSI CONTRAST", json.dumps(res, indent=1))

json.dump(M, open(METF, "w"), indent=1)
