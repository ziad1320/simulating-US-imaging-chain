"""Step 1 - transducer geometry, excitation pulse (Eq. 5) and simulated pulse-echo spectrum."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymust
import config as cfg

xe = cfg.element_x()
p = cfg.make_param()

# ---- analytic Gaussian-modulated pulse (Eq. 5): choose sigma for a -6 dB band of fc*BW ----
bw = cfg.FC * cfg.BANDWIDTH_PCT / 100
sigma_f = (bw / 2) / np.sqrt(2 * np.log(2))     # amplitude spectrum exp(-(f-f0)^2/2sf^2)=0.5 at +-bw/2
sigma_t = 1 / (2 * np.pi * sigma_f)
t = np.arange(-1.5e-6, 1.5e-6, 1 / cfg.FS)
s = np.exp(-t ** 2 / (2 * sigma_t ** 2)) * np.cos(2 * np.pi * cfg.FC * t)


def spectrum(sig, fs, nfft=2 ** 16):
    S = np.abs(np.fft.rfft(sig, nfft))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, 20 * np.log10(S / S.max() + 1e-12)


def band6(f, SdB):
    idx = np.where(SdB >= -6)[0]
    return f[idx[0]], f[idx[-1]]


f1, S1 = spectrum(s, cfg.FS)
lo1, hi1 = band6(f1, S1)

# ---- simulated pulse-echo from one scatterer (what the receive chain really sees) ----
x0, z0 = cfg.POINT_POS
d = pymust.txdelay(p, 0.0)
rf = pymust.simus(np.array([x0]), np.array([z0]), np.array([1.0]), d, p)
rf = np.asarray(rf[0] if isinstance(rf, tuple) else rf)
ch = rf[:, cfg.N_EL // 2]
k = np.argmax(np.abs(ch))
w = ch[k - 200:k + 200]
tw = (np.arange(-200, 200)) / cfg.FS
f2, S2 = spectrum(ch, cfg.FS)
lo2, hi2 = band6(f2, S2)
fpk = f2[np.argmax(S2)]

print(f"Eq.5 pulse   : sigma = {sigma_t*1e9:.1f} ns, -6 dB band = {lo1/1e6:.2f}-{hi1/1e6:.2f} MHz")
print(f"Simulated    : peak freq = {fpk/1e6:.2f} MHz, -6 dB band = {lo2/1e6:.2f}-{hi2/1e6:.2f} MHz "
      f"(fractional {100*(hi2-lo2)/((hi2+lo2)/2):.0f} %)")

# ---- figures ----
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(xe * 1e3, np.zeros_like(xe), "k.", ms=3)
sub = xe[48:80]
ax.plot(sub * 1e3, np.zeros_like(sub), "r.", ms=5, label=f"{cfg.N_SUB}-element sub-aperture ({cfg.APERTURE*1e3:.2f} mm)")
ax.set_xlabel("x [mm]"); ax.set_yticks([]); ax.set_title(f"Linear array: {cfg.N_EL} elements, pitch {cfg.PITCH*1e3:.4f} mm")
ax.legend(); fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig01_array.png", dpi=150); plt.close(fig)

fig, a = plt.subplots(2, 2, figsize=(11, 6))
a[0, 0].plot(t * 1e6, s); a[0, 0].set_title("Eq. 5 excitation pulse"); a[0, 0].set_xlabel("t [us]")
a[0, 1].plot(f1 / 1e6, S1); a[0, 1].set_xlim(0, 12); a[0, 1].set_ylim(-60, 3)
a[0, 1].axhline(-6, ls="--", c="r"); a[0, 1].set_title(f"Eq. 5 spectrum, -6 dB: {lo1/1e6:.1f}-{hi1/1e6:.1f} MHz"); a[0, 1].set_xlabel("f [MHz]")
a[1, 0].plot(tw * 1e6, w); a[1, 0].set_title("Simulated pulse-echo (centre element)"); a[1, 0].set_xlabel("t - t_peak [us]")
a[1, 1].plot(f2 / 1e6, S2); a[1, 1].set_xlim(0, 12); a[1, 1].set_ylim(-60, 3)
a[1, 1].axhline(-6, ls="--", c="r"); a[1, 1].set_title(f"Simulated spectrum, -6 dB: {lo2/1e6:.1f}-{hi2/1e6:.1f} MHz"); a[1, 1].set_xlabel("f [MHz]")
fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig02_pulse_spectrum.png", dpi=150); plt.close(fig)
