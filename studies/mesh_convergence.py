"""Mesh convergence of the bracket outputs (static hot spot, f1, random-vibration RMS stress, fatigue life).

All element sizes are scaled by the same factor s; s = 1 is the setting used in the design study.
The peak stress inside the bolt-hole zone (fixed support) is tracked as well, to show that it does not
converge (stress singularity at the idealised support), which is why that zone is excluded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bracketfea.analysis import run_design, design_mesh_settings
from bracketfea.geometry import BracketParams, MeshSettings

DESIGNS = {
    "A_no_gusset": BracketParams(t_b=6, t_w=6, r_f=5),
    "B_gusset": BracketParams(t_b=8, t_w=8, r_f=5, L_g=40, t_g=6),
}
LEVELS = {"A_no_gusset": [2.0, 1.4, 1.0, 0.7, 0.5], "B_gusset": [2.0, 1.4, 1.0, 0.7]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mesh_convergence")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, p in DESIGNS.items():
        base = design_mesh_settings(p)
        for s in LEVELS[name]:
            ms = MeshSettings(h_max=base.h_max * s, h_min=base.h_min * s, h_hole=base.h_hole * s,
                              dist_min=base.dist_min, dist_max=base.dist_max)
            r = run_design(p, ms=ms, threads=args.threads)
            summ = r.summary()
            worst = min(r.random, key=lambda a: r.random[a]["life_dirlik_h"])
            bolt_zone = ~r.region & (r.mesh.nodes[:, 2] < p.t_b + 1e-6) & (r.mesh.nodes[:, 0] > 50)
            row = dict(design=name, scale=s, h_min=ms.h_min, h_max=ms.h_max, n_nodes=r.mesh.n_nodes,
                       vm_max=summ["vm_max"], f1=summ["f1"], f2=summ["f2"],
                       worst_axis=worst, vm_rms=r.random[worst]["vm_rms_max"],
                       life_h=r.random[worst]["life_dirlik_h"],
                       vm_bolt_zone=float(r.vm_static[bolt_zone].max()),
                       hot_x=summ["hot_x"], hot_y=summ["hot_y"], hot_z=summ["hot_z"],
                       t_solve=r.timing["solve"])
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
                  flush=True)
            pd.DataFrame(rows).to_csv(out / "mesh_convergence.csv", index=False)


if __name__ == "__main__":
    main()
