"""Step 2 - build and plot the three phantoms."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import phantoms as ph

P = {"point": ph.point_phantom(), "wires": ph.wire_phantom(), "contrast": ph.contrast_phantom()}
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, (name, (x, z, rc)) in zip(ax, P.items()):
    a.scatter(x * 1e3, z * 1e3, s=np.clip(np.abs(rc) * 4, 0.5, 60), c="k", alpha=0.6)
    a.set_title(f"{name}: {len(x)} scatterers"); a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")
    a.invert_yaxis(); a.set_aspect("equal")
for (cx, cz, r), col in ((cfg.ANECHOIC, "b"), (cfg.HYPER, "r")):
    ax[2].add_patch(plt.Circle((cx * 1e3, cz * 1e3), r * 1e3, fill=False, ec=col, lw=2))
fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig03_phantoms.png", dpi=130); plt.close(fig)
for n, (x, z, rc) in P.items():
    print(f"{n:9s}: {len(x):6d} scatterers")
x, z, rc = P["contrast"]
cell = (1.2e-3 * 0.3e-3) * 1e6  # approx 2-D resolution cell in mm^2
print(f"contrast density {cfg.SCAT_PER_MM2}/mm^2 -> ~{cfg.SCAT_PER_MM2*cell:.0f} scatterers per resolution cell")
