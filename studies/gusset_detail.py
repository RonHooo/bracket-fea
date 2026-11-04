"""Detail design of the gusset: effect of the gusset edge/toe fillet radius r_g on fatigue life.

In the DOE every gusset has r_g = 2 mm, and no gusseted design met the fatigue requirement: the peak
stress sits at the gusset toe. This study keeps the rest of the geometry fixed and increases r_g.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bracketfea.analysis import run_design
from bracketfea.geometry import BracketParams
from bracketfea.problem import Requirements

BASES = {
    "G1 (t_b=8, t_w=6, L_g=40, t_g=6)": dict(t_b=8.0, t_w=6.0, r_f=5.0, L_g=40.0, t_g=6.0),
    "G2 (t_b=6, t_w=5, L_g=35, t_g=5)": dict(t_b=6.0, t_w=5.0, r_f=5.0, L_g=35.0, t_g=5.0),
}
RADII = [2.0, 3.0, 4.0, 5.0, 6.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gusset_detail")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    req = Requirements()
    rows = []
    for name, base in BASES.items():
        for rg in RADII:
            p = BracketParams(**base, r_g=rg)
            r = run_design(p, threads=args.threads)
            s = r.summary(req)
            worst = min(r.random, key=lambda a: r.random[a]["life_dirlik_h"])
            hot = r.mesh.nodes[r.random[worst]["hot_node"]]
            rows.append(dict(design=name, r_g=rg, mass_kg=s["mass_kg"], f1=s["f1"], vm_max=s["vm_max"],
                             life_h=s["life_h"], worst_axis=worst, vm_rms=r.random[worst]["vm_rms_max"],
                             fat_hot_x=hot[0], fat_hot_y=hot[1], fat_hot_z=hot[2], n_nodes=s["n_nodes"]))
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}),
                  flush=True)
            pd.DataFrame(rows).to_csv(out / "gusset_detail.csv", index=False)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in df.groupby("design"):
        ax[0].semilogy(g.r_g, g.life_h, "o-", label=name)
        ax[1].plot(g.r_g, g.vm_rms, "o-", label=name)
    ax[0].axhline(req.life_min_h, color="k", ls="--", lw=1, label="requirement")
    ax[0].set_ylabel("fatigue life, worst axis (h)")
    ax[1].set_ylabel("hot-spot RMS von Mises (MPa)")
    for a in ax:
        a.set_xlabel("gusset edge fillet radius r_g (mm)")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "gusset_detail.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
