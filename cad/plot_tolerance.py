"""Tornado chart: change of f1 and fatigue life when each dimension sits at its tolerance limit.

    .venv/bin/python cad/plot_tolerance.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LO, HI = "#eb6834", "#2a78d6"          # categorical slots 2 and 1 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"t_b": "base thickness t_b = 10.0 ± 0.2", "t_w": "upright thickness t_w = 7.0 ± 0.2",
         "r_f": "fillet radius r_f = R8 ± 0.5", "d_h": "lightening hole d_h = 10.0 ± 0.2"}


def main():
    d = json.loads((HERE / "results/tolerance_study.json").read_text())
    sens, tol = d["sensitivity"], d["tolerance"]
    keys = sorted(tol, key=lambda k: abs(sens[k]["log10_life"] * tol[k]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    panels = [("f1", "change of f1 (Hz)", lambda k, s: s * sens[k]["f1"] * tol[k]),
              ("life", "change of fatigue life (%)",
               lambda k, s: 100 * (10 ** (s * sens[k]["log10_life"] * tol[k]) - 1))]
    for ax, (name, xlabel, fn) in zip(axes, panels):
        for i, k in enumerate(keys):
            lo, hi = fn(k, -1), fn(k, +1)
            ax.barh(i + 0.19, hi, height=0.36, color=HI, label="dimension at upper limit" if i == 0 else None)
            ax.barh(i - 0.19, lo, height=0.36, color=LO, label="dimension at lower limit" if i == 0 else None)
            for v, y in ((hi, i + 0.19), (lo, i - 0.19)):
                if abs(v) > 0.05 * max(abs(fn(kk, 1)) for kk in keys):
                    ax.text(v + (0.6 if name == "life" else 0.04) * (1 if v > 0 else -1), y,
                            f"{v:+.1f}", va="center", ha="left" if v > 0 else "right", fontsize=8.5, color=MUTED)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel(xlabel, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        lim = max(abs(fn(k, s)) for k in keys for s in (-1, 1)) * 1.35
        ax.set_xlim(-lim, lim)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([NAMES[k] for k in keys], color=INK)
    axes[0].set_title(f"first natural frequency (nominal {d['worst_case']['f1']['linear_nominal']:.1f} Hz)",
                      fontsize=10, color=INK, loc="left")
    axes[1].set_title(f"fatigue life, worst axis (nominal {10 ** d['worst_case']['log10_life']['linear_nominal']:.0f} h)",
                      fontsize=10, color=INK, loc="left")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.savefig(HERE / "results/figures/tolerance_tornado.png", dpi=170, bbox_inches="tight")
    print("wrote tolerance_tornado.png")


if __name__ == "__main__":
    main()
