"""Analyse the DOE: feasibility, trade-offs, hot-spot locations, lightest feasible design.

Also runs two reference studies with CalculiX:
  * plain bracket (no gussets, no hole, t_b = t_w = t, r_f = 5): thinnest t that meets the requirements
  * the lightest feasible DOE design re-analysed on a finer mesh (scale 0.7), because the design-study
    mesh over-predicts fatigue life by roughly 20-30 % (see mesh_convergence.csv)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bracketfea.analysis import run_design, design_mesh_settings
from bracketfea.geometry import BracketParams, MeshSettings
from bracketfea.problem import Requirements

PARAMS = ["t_b", "t_w", "r_f", "L_g", "t_g", "d_h"]
LIFE_MARGIN = 1.5      # life knock-down for the design-study mesh (fine-mesh life is ~20-30 % lower)


def classify_hot_spot(row) -> str:
    p = BracketParams(**{k: row[k] for k in PARAMS})
    x, y, z = row.hot_x, row.hot_y, row.hot_z
    tol = 1.0
    if p.has_hole and np.hypot(y, z - p.z_hole) < p.d_h / 2 + 2.0 and x < p.t_w + tol:
        return "lightening hole"
    if p.has_gussets and abs(abs(y) - p.y_gusset) < p.t_g / 2 + p.r_g + tol:
        if abs(x - (p.t_w + p.L_g)) < 4 and z < p.t_b + p.r_g + tol:
            return "gusset toe (on base)"
        if abs(z - (p.t_b + p.L_g)) < 4 and x < p.t_w + p.r_g + tol:
            return "gusset tip (on upright)"
        return "gusset edge"
    if p.t_w - tol < x < p.t_w + p.r_f + tol and p.t_b - tol < z < p.t_b + p.r_f + tol:
        return "root fillet"
    if z < p.t_b + tol and x > p.t_w + p.r_f:
        return "base plate (bolt side)"
    if x < p.t_w + tol:
        return "upright"
    return "other"


def plain_reference(ts, threads, req):
    rows = []
    for t in ts:
        r = run_design(BracketParams(t_b=t, t_w=t, r_f=5.0), threads=threads)
        s = r.summary(req)
        s["design"] = f"plain t={t}"
        rows.append(s)
        print(f"plain t={t}: mass {s['mass_kg']:.3f} kg, f1 {s['f1']:.0f} Hz, life {s['life_h']:.3g} h, "
              f"vm {s['vm_max']:.0f} MPa, feasible {s['feasible']}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doe", default="results/doe/doe_results.csv")
    ap.add_argument("--out", default="results/optimization")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-fe", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    req = Requirements()

    d = pd.read_csv(args.doe)
    d = d[(d.status == "ok") & (d.split == "doe")].copy()
    d["ok_stress"] = d.vm_max <= req.stress_allow
    d["ok_freq"] = d.f1 >= req.f1_min
    d["ok_life"] = d.life_h >= req.life_min_h
    d["ok_life_margin"] = d.life_h >= LIFE_MARGIN * req.life_min_h
    d["feasible"] = d.ok_stress & d.ok_freq & d.ok_life
    d["feasible_margin"] = d.ok_stress & d.ok_freq & d.ok_life_margin
    d["gussets"] = np.where(d.L_g > 0, "with gussets", "no gussets")
    d["hot_spot"] = d.apply(classify_hot_spot, axis=1)
    d.to_csv(out / "doe_analysis.csv", index=False)

    stats = dict(n=len(d), ok_stress=int(d.ok_stress.sum()), ok_freq=int(d.ok_freq.sum()),
                 ok_life=int(d.ok_life.sum()), feasible=int(d.feasible.sum()),
                 feasible_with_margin=int(d.feasible_margin.sum()),
                 hot_spot_counts=d.hot_spot.value_counts().to_dict(),
                 by_gussets=d.groupby("gussets")[["f1", "life_h", "mass_kg", "vm_max"]].median().to_dict(),
                 feasible_by_gussets=d.groupby("gussets").feasible_margin.mean().to_dict(),
                 corr_log_life=d[PARAMS].corrwith(np.log10(d.life_h), method="spearman").to_dict(),
                 corr_f1=d[PARAMS].corrwith(d.f1, method="spearman").to_dict())

    best = d[d.feasible_margin].sort_values("mass_kg").head(5)
    best.to_csv(out / "lightest_feasible.csv", index=False)
    stats["lightest_feasible"] = best.iloc[0][PARAMS + ["mass_kg", "vm_max", "f1", "life_h", "hot_spot"]].to_dict() \
        if len(best) else None

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sc = ax[0].scatter(d.mass_kg, d.f1, c=np.log10(d.life_h), cmap="viridis", s=14, vmin=-1, vmax=4.5)
    ax[0].scatter(d[d.feasible_margin].mass_kg, d[d.feasible_margin].f1, facecolors="none",
                  edgecolors="red", s=40, label="feasible (with life margin)")
    ax[0].axhline(req.f1_min, color="k", ls="--", lw=1)
    ax[0].set_xlabel("bracket mass (kg)")
    ax[0].set_ylabel("first natural frequency f1 (Hz)")
    ax[0].legend(loc="lower right", fontsize=8)
    plt.colorbar(sc, ax=ax[0], label="log10 fatigue life (h)")
    for g, mk in (("no gussets", "o"), ("with gussets", "^")):
        s = d[d.gussets == g]
        ax[1].scatter(s.mass_kg, s.life_h, marker=mk, s=14, alpha=0.7, label=g)
    ax[1].axhline(req.life_min_h, color="k", ls="--", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("bracket mass (kg)")
    ax[1].set_ylabel("random-vibration fatigue life (h)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoffs.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharey="row")
    for j, k in enumerate(PARAMS):
        for i, (col, lab, log) in enumerate((("f1", "f1 (Hz)", False), ("life_h", "life (h)", True))):
            a = axes[i, j]
            mask = d[k] > 0 if k in ("L_g", "d_h") else np.ones(len(d), bool)
            a.scatter(d[k][mask], d[col][mask], s=6, alpha=0.5)
            if log:
                a.set_yscale("log")
            a.set_xlabel(k + (" (>0 only)" if k in ("L_g", "d_h") else "") + " (mm)")
            if j == 0:
                a.set_ylabel(lab)
            a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "parameter_effects.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    hs = d.hot_spot.value_counts()
    ax.barh(hs.index, hs.values)
    ax.set_xlabel("number of DOE designs")
    ax.set_title("Location of the peak static stress (design region)")
    fig.tight_layout()
    fig.savefig(out / "hot_spot_locations.png", dpi=150)
    plt.close(fig)

    if not args.skip_fe:
        ref = plain_reference([6, 7, 8, 9, 10, 11, 12], args.threads, req)
        ref.to_csv(out / "plain_reference.csv", index=False)
        ok = ref[ref.feasible]
        stats["plain_reference_thinnest_feasible"] = ok.iloc[0][["design", "mass_kg", "f1", "life_h"]].to_dict() \
            if len(ok) else None
        if len(best):
            b = best.iloc[0]
            p = BracketParams(**{k: float(b[k]) for k in PARAMS})
            base = design_mesh_settings(p)
            ms = MeshSettings(h_max=base.h_max * 0.7, h_min=base.h_min * 0.7, h_hole=base.h_hole * 0.7,
                              dist_min=base.dist_min, dist_max=base.dist_max)
            r = run_design(p, ms=ms, threads=args.threads)
            s = r.summary(req)
            stats["lightest_feasible_fine_mesh"] = {k: s[k] for k in ("mass_kg", "vm_max", "f1", "life_h",
                                                                      "feasible", "n_nodes")}
            print("fine-mesh check of the lightest feasible design:", stats["lightest_feasible_fine_mesh"])
    (out / "summary.json").write_text(json.dumps(stats, indent=2, default=float))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("by_gussets",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
