"""Step 4 + 7 - custom DAS beamforming, B-mode images, cross-check vs pymust, quantitative metrics.
Usage: python 04_das.py [point|wires|contrast]   (no argument = all)"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymust
import config as cfg
import beamformer as bf
import metrics as mt

what = sys.argv[1:] or ["point", "wires", "contrast"]
METF = f"{cfg.DATA}/metrics_04.json"
M = json.load(open(METF)) if os.path.exists(METF) else {}


def load(name):
    d = np.load(f"{cfg.DATA}/{name}.npz")
    return d["rf"], d["angles"]


def show(ax, db, x, z, title, dr):
    ax.imshow(db, extent=[x[0] * 1e3, x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3], cmap="gray", vmin=-dr, vmax=0, aspect="equal")
    ax.set_title(title); ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")


# ============================================================ POINT
if "point" in what:
    rf, ang = load("point")
    xl = bf.lines_for(-5e-3, 5e-3, cfg.LATERAL_UPSAMPLE)
    zg = np.arange(16e-3, 22e-3, cfg.DZ)
    img = bf.beamform(rf, ang, xl, zg, bf.rect())[0]
    env = bf.envelope(img)
    db = bf.to_db(env, 80)
    iz, ix = np.unravel_index(np.argmax(env), env.shape)
    lat = db[iz]; ax_prof = db[:, ix]
    f_lat = mt.fwhm(xl * 1e3, lat); f_ax = mt.fwhm(zg * 1e3, ax_prof)
    psl = mt.peak_sidelobe_db(xl * 1e3, lat)
    z_pk, x_pk = zg[iz] * 1e3, xl[ix] * 1e3
    D = cfg.APERTURE; F = cfg.POINT_POS[1] / D
    th_rx = 1.207 * cfg.LAMBDA * F * 1e3     # -6 dB width of a sinc (receive-only, rect. aperture)
    th_pe = 1.41 * cfg.LAMBDA * F * 1e3      # Eq. 11 of the proposal
    # theory: far-field receive pattern of the rectangular aperture at fc
    u = (xl - xl[ix]) * D / (cfg.LAMBDA * cfg.POINT_POS[1])
    th_db = 20 * np.log10(np.abs(np.sinc(u)) + 1e-12)
    # ---- cross-check against pymust DAS matrix (f-number = z0/D) ----
    p = cfg.make_param(); p.fnumber = F
    X, Z = np.meshgrid(xl, zg)
    d = pymust.txdelay(p, 0.0)
    Mx = pymust.dasmtx(np.array(rf[0].shape), X, Z, d, p)
    ref = (Mx @ rf[0].astype(np.float64).flatten(order="F")).reshape(X.shape, order="F")
    ref_db = bf.to_db(bf.envelope(ref), 80)
    lat_ref = ref_db[iz]
    m = lat > -30
    rms = float(np.sqrt(np.mean((lat[m] - lat_ref[m]) ** 2)))
    f_ref = mt.fwhm(xl * 1e3, lat_ref)
    M["point"] = dict(peak_z_mm=z_pk, peak_x_mm=x_pk, fwhm_lat_mm=f_lat, fwhm_ax_mm=f_ax, peak_sidelobe_dB=psl,
                      theory_fwhm_rx_mm=th_rx, theory_eq11_mm=th_pe, pymust_fwhm_lat_mm=f_ref, rms_diff_vs_pymust_dB=rms)
    print("POINT", json.dumps(M["point"], indent=1))
    fig, a = plt.subplots(1, 3, figsize=(16, 4.2))
    show(a[0], db, xl, zg, "Point target, DAS (80 dB)", 80)
    a[1].plot(xl * 1e3, lat, label="own DAS"); a[1].plot(xl * 1e3, lat_ref, "--", label="pymust dasmtx")
    a[1].plot(xl * 1e3, th_db, ":", c="k", label="sinc theory (fc)")
    a[1].axhline(-6, c="r", lw=.5); a[1].axhline(-13.3, c="g", lw=.5); a[1].set_ylim(-60, 3); a[1].set_xlim(-4, 4)
    a[1].set_title(f"Lateral profile: FWHM {f_lat:.2f} mm, PSL {psl:.1f} dB"); a[1].set_xlabel("x [mm]"); a[1].legend()
    a[2].plot(zg * 1e3, ax_prof); a[2].axhline(-6, c="r", lw=.5); a[2].set_ylim(-60, 3)
    a[2].set_title(f"Axial profile (envelope): FWHM {f_ax:.2f} mm"); a[2].set_xlabel("z [mm]")
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig05_point_psf.png", dpi=140); plt.close(fig)

# ============================================================ WIRES
if "wires" in what:
    rf, ang = load("wires")
    fig, a = plt.subplots(1, 1, figsize=(5, 6))
    a.imshow(rf[0, :4200], aspect="auto", cmap="gray", vmin=-np.abs(rf).max() / 4, vmax=np.abs(rf).max() / 4,
             extent=[0, cfg.N_EL, 4200 / cfg.FS * 1e6, 0])
    a.set_xlabel("element"); a.set_ylabel("time [us]"); a.set_title("Channel data (wire phantom, 0 deg)")
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig04_channel_data.png", dpi=140); plt.close(fig)

    xl = bf.lines_for(-10e-3, 10e-3, cfg.LATERAL_UPSAMPLE)
    zg = np.arange(6e-3, 34e-3, cfg.DZ)
    env = bf.envelope(bf.beamform(rf, ang, xl, zg, bf.rect())[0])
    db = bf.to_db(env, cfg.DR_DB)
    xz = np.arange(-3e-3, 3e-3, cfg.PITCH / 16)
    zz = np.arange(16.5e-3, 19.5e-3, cfg.DZ)
    envz = bf.envelope(bf.beamform(rf, ang, xz, zz, bf.rect())[0])
    dbz = bf.to_db(envz, cfg.DR_DB_WIRES_CLOSE)
    # wire depths: local maximum of the envelope on the centre line near each true depth
    res = {}
    ic = np.argmin(np.abs(xl))
    for zt in cfg.WIRE_Z:
        w = (zg > zt - 1e-3) & (zg < zt + 1e-3)
        res[f"wire_{zt*1e3:.0f}mm_est_z_mm"] = float(zg[w][np.argmax(env[w, ic])] * 1e3)
    # close pair: profile at the depth of the pair
    iz = np.argmax(envz.max(axis=1))
    prof = dbz[iz]
    xs = xz * 1e3
    cen = np.argmin(np.abs(xs))                       # x = 0, mid-point between the two wires
    valley = float(prof[cen]); peaks = float(prof.max())
    res["pair_valley_dB"] = valley - peaks
    res["pair_separated_(valley<-3dB)"] = bool(valley - peaks < -3)
    k20 = np.argmin(np.abs(zg - 20e-3))
    p20 = db[k20 - 20:k20 + 20].max(axis=0)
    res["fwhm_lat_wire20mm_mm"] = float(mt.fwhm(xl * 1e3, p20 - p20.max()))
    M["wires"] = res
    print("WIRES", json.dumps(res, indent=1))
    fig, a = plt.subplots(1, 3, figsize=(17, 6))
    show(a[0], db, xl, zg, "Wire phantom, DAS (60 dB)", cfg.DR_DB)
    show(a[1], dbz, xz, zz, "Close pair, 0.8 mm apart (40 dB)", cfg.DR_DB_WIRES_CLOSE)
    a[2].plot(xs, prof); a[2].set_title(f"Pair lateral profile (valley {valley-peaks:.1f} dB)"); a[2].set_xlabel("x [mm]"); a[2].set_ylim(-40, 2)
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig06_wires.png", dpi=140); plt.close(fig)

# ============================================================ CONTRAST
if "contrast" in what:
    rf, ang = load("contrast")
    xl = bf.lines_for(-14e-3, 14e-3, cfg.LATERAL_UPSAMPLE_CONTRAST)
    zg = np.arange(10e-3, 40e-3, cfg.DZ)
    X, Z = np.meshgrid(xl, zg)
    i0 = int(np.argmin(np.abs(ang)))
    variants = {"1 angle (0 deg)": bf.envelope(bf.beamform(rf[i0:i0 + 1], ang[i0:i0 + 1], xl, zg, bf.rect())[0]),
                f"{len(ang)} angles compounded": bf.envelope(bf.beamform(rf, ang, xl, zg, bf.rect())[0])}
    bg_patch = ((np.abs(X) < 12e-3) & (Z > 13e-3) & (Z < 37e-3)
                & (np.hypot(X - cfg.ANECHOIC[0], Z - cfg.ANECHOIC[1]) > 1.6 * cfg.ANECHOIC[2])
                & (np.hypot(X - cfg.HYPER[0], Z - cfg.HYPER[1]) > 1.6 * cfg.HYPER[2]))   # all cyst-free speckle
    res = {}
    fig, a = plt.subplots(1, 2, figsize=(12, 7))
    for ax_, (name, env) in zip(a, variants.items()):
        db = bf.to_db(env, cfg.DR_DB)
        show(ax_, db, xl, zg, f"DAS, {name}", cfg.DR_DB)
        for (cx, cz, r), col in ((cfg.ANECHOIC, "c"), (cfg.HYPER, "r")):
            ax_.add_patch(plt.Circle((cx * 1e3, cz * 1e3), r * 1e3, fill=False, ec=col, lw=1))
        r_ = {}
        for lab, cyst in (("anechoic", cfg.ANECHOIC), ("hyperechoic", cfg.HYPER)):
            bgm = mt.disc_mask(X, Z, cyst[0], cyst[1], 1.3 * cyst[2], 1.9 * cyst[2])
            r_[lab] = mt.contrast_metrics(env, X, Z, cyst, bgm)
        r_["speckle_SNR"] = float(mt.speckle_snr_blocks(env, bg_patch, X, Z))
        res[name] = r_
    M["contrast"] = res
    print("CONTRAST", json.dumps(res, indent=1))
    fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig07_contrast.png", dpi=140); plt.close(fig)

json.dump(M, open(METF, "w"), indent=1)
