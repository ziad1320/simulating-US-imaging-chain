"""06_validate.py - automatic test suite. Run after 03/04/05:  python 06_validate.py
Each test prints PASS/FAIL with the measured value and the accepted range."""
import json, os
import numpy as np
import pymust
import config as cfg
import beamformer as bf

res = []


def check(name, value, lo, hi, unit=""):
    ok = (value >= lo) and (value <= hi)
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name:55s} {value:9.3f} {unit:3s} (expected {lo} .. {hi})")


def flag(name, cond, info=""):
    res.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name:55s} {info}")


M4 = json.load(open(f"{cfg.DATA}/metrics_04.json"))
M5 = json.load(open(f"{cfg.DATA}/metrics_05.json")) if os.path.exists(f"{cfg.DATA}/metrics_05.json") else {}
p = cfg.make_param()
xe = cfg.element_x()

print("--- T1 geometry")
check("array length (127 pitches) [mm]", (xe[-1] - xe[0]) * 1e3, 38.70, 38.72, "mm")
check("32-element aperture [mm]", cfg.APERTURE * 1e3, 9.74, 9.76, "mm")
check("wavelength [mm]", cfg.LAMBDA * 1e3, 0.307, 0.309, "mm")

print("--- T2 transmit-time calibration (own beamformer vs pymust.txdelay) for tilted plane waves")
for deg in (-20, -10, 0, 10, 20):
    th = np.deg2rad(deg)
    d = pymust.txdelay(p, th)[0]
    flag(f"t_off({deg:+d} deg) equals txdelay offset",
         abs((d - xe * np.sin(th) / cfg.C).mean() - (-np.min(xe * np.sin(th)) / cfg.C)) < 1e-9)
    x0, z0 = 3e-3, 19e-3
    rf = pymust.simus(np.array([x0]), np.array([z0]), np.array([1.0]), pymust.txdelay(p, th), p)
    rf = np.asarray(rf[0] if isinstance(rf, tuple) else rf, dtype=np.float32)[None]
    xl = np.arange(1.5e-3, 4.5e-3, cfg.PITCH / 8); zg = np.arange(17e-3, 21e-3, cfg.DZ)
    env = bf.envelope(bf.beamform(rf, np.array([th]), xl, zg, bf.rect())[0])
    iz, ix = np.unravel_index(np.argmax(env), env.shape)
    check(f"point depth @ {deg:+d} deg [mm]", zg[iz] * 1e3, 18.95, 19.05, "mm")
    check(f"point x     @ {deg:+d} deg [mm]", xl[ix] * 1e3, 2.9, 3.1, "mm")

print("--- T3 single point (0 deg, rectangular 32-element DAS)")
P = M4["point"]
check("peak depth [mm]", P["peak_z_mm"], 18.95, 19.05, "mm")
check("peak lateral position [mm]", P["peak_x_mm"], -0.05, 0.05, "mm")
check("lateral FWHM (-6 dB) [mm]", P["fwhm_lat_mm"], 0.65, 0.95, "mm")
check("axial FWHM (envelope) [mm]", P["fwhm_ax_mm"], 0.15, 0.35, "mm")
check("peak sidelobe [dB]", P["peak_sidelobe_dB"], -60, -13, "dB")
check("RMS diff of profile vs pymust.dasmtx [dB]", P["rms_diff_vs_pymust_dB"], 0, 1.0, "dB")

print("--- T4 wires")
W = M4["wires"]
for z in (10, 20, 30):
    check(f"wire at {z} mm: estimated depth [mm]", W[f"wire_{z}mm_est_z_mm"], z - 0.05, z + 0.05, "mm")
flag("0.8 mm pair NOT resolved by conventional DAS (expected)", not W["pair_separated_(valley<-3dB)"],
     f"valley {W['pair_valley_dB']:.2f} dB")

print("--- T5 noisy wire data still localise the 20 mm wire")
for snr in cfg.NOISE_SNR_DB:
    f = f"{cfg.DATA}/wires_snr{snr}.npz"
    if not os.path.exists(f):
        print(f"[SKIP] wires_snr{snr}.npz missing (run 03_simulate.py --noise)"); continue
    d = np.load(f)
    xl = np.array([0.0]); zg = np.arange(18e-3, 22e-3, cfg.DZ)
    env = bf.envelope(bf.beamform(d["rf"], d["angles"], xl, zg, bf.rect())[0])
    check(f"SNR {snr} dB: depth of the 20 mm wire [mm]", zg[np.argmax(env[:, 0])] * 1e3, 19.9, 20.1, "mm")

print("--- T6 contrast phantom (DAS)")
C = M4["contrast"]
keys = list(C.keys()); single, comp = C[keys[0]], C[keys[1]]
check("hyperechoic CR, compounded [dB] (true +12.0)", comp["hyperechoic"]["CR_dB"], 8, 14, "dB")
check("anechoic CR, compounded [dB]", comp["anechoic"]["CR_dB"], -60, -10, "dB")
check("speckle SNR, compounded", comp["speckle_SNR"], 1.3, 2.6)
check("speckle SNR, single angle", single["speckle_SNR"], 1.5, 2.2)

print("--- T7 NSI (optional)")
if M5:
    N = M5["point"]
    flag("NSI lateral FWHM >= 5x narrower than DAS", N["NSI_c=0.03"]["fwhm_lat_mm"] < N["conventional"]["fwhm_lat_mm"] / 5,
         f"{N['NSI_c=0.03']['fwhm_lat_mm']*1e3:.0f} um vs {N['conventional']['fwhm_lat_mm']*1e3:.0f} um")
    flag("NSI FWHM grows with c (0.01 < 0.03 < 0.05)",
         N["NSI_c=0.01"]["fwhm_lat_mm"] < N["NSI_c=0.03"]["fwhm_lat_mm"] < N["NSI_c=0.05"]["fwhm_lat_mm"])
    flag("NSI peak sidelobe lower than DAS", N["NSI_c=0.03"]["psl_dB"] < N["conventional"]["psl_dB"])
    if "contrast" in M5:
        k = list(M5["contrast"].keys())
        flag("NSI lowers speckle SNR (under-developed speckle)", M5["contrast"][k[1]]["speckle_SNR"] < M5["contrast"][k[0]]["speckle_SNR"])
        flag("NSI lowers CNR of the hyperechoic cyst", M5["contrast"][k[1]]["hyperechoic"]["CNR"] < M5["contrast"][k[0]]["hyperechoic"]["CNR"])
else:
    print("[SKIP] run 05_nsi.py first")

print(f"\n{sum(res)}/{len(res)} checks passed")
