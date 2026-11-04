"""Figures for the verification cases and the mesh convergence study."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# V2: Kt convergence
v2 = pd.read_csv(R / "verification/v2_plate_hole.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(10 / v2.h_hole, v2.Kt, "o-", label="CalculiX C3D10")
ax.axhline(v2.Kt_ref.iloc[0], color="k", ls="--", label=f"Heywood formula, Kt = {v2.Kt_ref.iloc[0]:.3f}")
ax.set_xscale("log")
ax.set_xlabel("hole radius / element size at the hole")
ax.set_ylabel("stress concentration factor Kt (net section)")
ax.set_title("V2: plate with a hole, d/W = 0.2")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "v2_kt_convergence.png", dpi=150)
plt.close(fig)

# V1 table as a bar chart of errors (finest mesh)
v1 = pd.read_csv(R / "verification/v1_cantilever.csv")
last = v1.iloc[-1]
labels = {"delta": "tip deflection", "sigma_mid": "mid-span stress", "f1_z": "f1 (bending, thin dir.)",
          "f2_z": "f2 (bending, thin dir.)", "f1_y": "f1 (bending, wide dir.)", "f1_tip": "f1 with tip mass"}
fig, ax = plt.subplots(figsize=(6.5, 3.4))
vals = [last[f"{k}_err_%"] for k in labels]
ax.barh(list(labels.values()), vals, color=["C0" if abs(v) < 1 else "C3" for v in vals])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("error vs beam theory (%)")
ax.set_title(f"V1: cantilever beam, {int(last.n_nodes)} nodes")
ax.set_xlim(-1.5, 1.5)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "v1_cantilever_errors.png", dpi=150)
plt.close(fig)

# mesh convergence
mc = pd.read_csv(R / "mesh_convergence/mesh_convergence.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (name, g) in zip(axes, mc.groupby("design")):
    ref = g.iloc[-1]
    for col, lab in (("vm_max", "static peak von Mises"), ("f1", "f1"), ("vm_rms", "RMS stress (worst axis)"),
                     ("life_h", "fatigue life (worst axis)"), ("vm_bolt_zone", "peak stress at fixed bolt hole")):
        ax.plot(g.n_nodes, g[col] / ref[col], "o-", label=lab)
    ax.axvline(g[g.scale == 1.0].n_nodes.iloc[0], color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_title(f"{name} (dotted: mesh used in the design study)")
    ax.set_xlabel("number of nodes")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("value / value on the finest mesh")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "mesh_convergence.png", dpi=150)
plt.close(fig)
print("saved to", FIG)
