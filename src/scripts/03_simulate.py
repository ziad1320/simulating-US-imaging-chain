"""Step 3 - RF channel-data simulation (MUST simus). Saves float32 arrays rf[angle, time, channel].
Usage: python src/scripts/03_simulate.py [--quick] [--noise] [--only point|wires|contrast]
  --quick : 5 angles and half the scatterer density for the contrast phantom (minutes instead of ~40 min)
  --medium: all 21 angles but half the scatterer density (~15 min); good compromise on a laptop
  --noise : also save SNR 35 dB and 10 dB noisy copies of the wire and contrast data"""
import sys, time
import numpy as np
import pymust
from lib import config as cfg
from lib import phantoms as ph

quick = "--quick" in sys.argv
medium = "--medium" in sys.argv
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
noise = "--noise" in sys.argv
p = cfg.make_param()


def simulate(x, z, rc, angles):
    out = []
    for i, th in enumerate(angles):
        t0 = time.time()
        d = pymust.txdelay(p, float(th))
        if cfg.USE_3D:
            rf = pymust.simus(x, np.zeros_like(x), z, rc, d, p)
        else:
            rf = pymust.simus(x, z, rc, d, p)
        rf = np.asarray(rf[0] if isinstance(rf, tuple) else rf, dtype=np.float32)
        out.append(rf)
        print(f"   angle {i+1}/{len(angles)} ({np.rad2deg(th):+.1f} deg) done in {time.time()-t0:.1f} s, rf {rf.shape}", flush=True)
    # keep only the samples that can contain echoes (simus returns a longer record than needed)
    half = cfg.N_EL * cfg.PITCH / 2
    zmax, xmax = np.max(z), np.max(np.abs(x))
    tmax = (zmax + np.hypot(zmax, xmax + half)) / cfg.C + 2 * half * np.sin(np.max(np.abs(angles))) / cfg.C + 2e-6
    n = min(min(r.shape[0] for r in out), int(tmax * cfg.FS))
    return np.stack([r[:n] for r in out])


def add_noise(rf, snr_db, seed):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.mean(rf.astype(np.float64) ** 2) / 10 ** (snr_db / 10))
    return (rf + rng.normal(0, sigma, rf.shape)).astype(np.float32)


def save(name, rf, angles, x, z, rc):
    np.savez_compressed(f"{cfg.DATA}/{name}.npz", rf=rf, angles=angles, fs=cfg.FS, fc=cfg.FC, c=cfg.C,
                        pitch=cfg.PITCH, n_el=cfg.N_EL, scat_x=x, scat_z=z, scat_rc=rc)
    print(f"saved data/{name}.npz  rf{rf.shape}  ({rf.nbytes/1e6:.1f} MB)")


jobs = [("point", ph.point_phantom(), cfg.ANGLES_SINGLE),
        ("wires", ph.wire_phantom(), cfg.ANGLES_SINGLE)]
ang_c = cfg.ANGLES_COMPOUND[::5] if quick else cfg.ANGLES_COMPOUND   # --medium keeps all 21
jobs.append(("contrast", ph.contrast_phantom(cfg.SCAT_PER_MM2 / 2 if (quick or medium) else None), ang_c))

for name, (x, z, rc), ang in jobs:
    if only and name != only:
        continue
    print(f"== {name}: {len(x)} scatterers, {len(ang)} angle(s)", flush=True)
    rf = simulate(x, z, rc, ang)
    save(name, rf, ang, x, z, rc)
    if noise and name != "point":
        for snr in cfg.NOISE_SNR_DB:
            save(f"{name}_snr{snr}", add_noise(rf, snr, cfg.SEED + snr), ang, x, z, rc)
