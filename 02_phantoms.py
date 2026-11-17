"""Step 2 - build and plot the three phantoms."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as cfg
import phantoms as ph

P = {"point": ph.point_phantom(), "wires": ph.wire_phantom(), "contrast": ph.contrast_phantom()}
fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1, 1.3]})

# ---- point phantom: zoomed window around the single scatterer, equal axes ----
x, z, rc = P["point"]
a = ax[0]
a.scatter(x * 1e3, z * 1e3, s=80, c="r", edgecolors="k", zorder=3, label="point (rc = 1)")
a.set_xlim(-2, 2); a.set_ylim(z[0] * 1e3 + 2, z[0] * 1e3 - 2)
a.set_xticks([-2, -1, 0, 1, 2]); a.set_aspect("equal"); a.legend(loc="lower right", fontsize=8)
a.set_title(f"point: {len(x)} scatterer (zoomed, 4 x 4 mm)")
a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")

# ---- wire phantom: wires (rc = 1) drawn big and red, weak background drawn small and grey ----
x, z, rc = P["wires"]
is_wire = np.abs(rc) >= 0.5
a = ax[1]
a.scatter(x[~is_wire] * 1e3, z[~is_wire] * 1e3, s=1, c="0.55", alpha=0.6, label=f"weak background ({(~is_wire).sum()})")
a.scatter(x[is_wire] * 1e3, z[is_wire] * 1e3, s=45, c="r", edgecolors="k", zorder=3, label=f"wires ({is_wire.sum()})")
a.legend(loc="lower right", fontsize=8)
a.set_title(f"wires: {len(x)} scatterers"); a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")
a.invert_yaxis(); a.set_aspect("equal")

# ---- contrast phantom: dot size follows strength; zero-strength scatterers are not drawn ----
x, z, rc = P["contrast"]
nz = rc != 0
a = ax[2]
a.scatter(x[nz] * 1e3, z[nz] * 1e3, s=np.clip(np.abs(rc[nz]) * 4, 0.5, 60), c="k", alpha=0.6)
a.set_title(f"contrast: {len(x)} scatterers"); a.set_xlabel("x [mm]"); a.set_ylabel("z [mm]")
a.invert_yaxis(); a.set_aspect("equal")
for (cx, cz, r), col in ((cfg.ANECHOIC, "b"), (cfg.HYPER, "r")):
    a.add_patch(plt.Circle((cx * 1e3, cz * 1e3), r * 1e3, fill=False, ec=col, lw=2))

fig.tight_layout(); fig.savefig(f"{cfg.FIG}/fig03_phantoms.png", dpi=130); plt.close(fig)
for n, (x, z, rc) in P.items():
    print(f"{n:9s}: {len(x):6d} scatterers")
x, z, rc = P["contrast"]
cell = (1.2e-3 * 0.3e-3) * 1e6  # approx 2-D resolution cell in mm^2
print(f"contrast density {cfg.SCAT_PER_MM2}/mm^2 -> ~{cfg.SCAT_PER_MM2*cell:.0f} scatterers per resolution cell")
