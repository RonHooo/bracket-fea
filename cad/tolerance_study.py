"""How much do machining tolerances change f1 and fatigue life of the released bracket?

1. Central finite differences with FEA for t_b, t_w, r_f, d_h (step larger than the tolerance so that
   the change is well above mesh noise; all runs use the same mesh settings).
2. Linear model in f1 and log10(life): worst case (all tolerances at the bad side) and a Monte Carlo
   estimate with each dimension normal, tolerance = 3 sigma.
3. FEA check of the two worst-case corners (lowest f1, shortest life) to test the linear model.

    .venv/bin/python cad/tolerance_study.py --threads 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bracketfea.analysis import design_mesh_settings, run_design  # noqa: E402
from bracketfea.geometry import BracketParams  # noqa: E402
from bracketfea.problem import Requirements  # noqa: E402

# drawing tolerances (+/-, mm): ISO 2768-m linear for 6-30 mm; the fillet radius is toleranced on the
# drawing (R8 +/-0.5) because ISO 2768-m would allow +/-1 for radii above 6 mm
TOL = {"t_b": 0.2, "t_w": 0.2, "r_f": 0.5, "d_h": 0.2}
STEP = {"t_b": 0.5, "t_w": 0.5, "r_f": 1.0, "d_h": 1.0}
OUTS = ("f1", "log10_life", "vm_max", "mass_kg")


def released():
    d = json.loads((ROOT / "cad/released_design.json").read_text())
    return {k: d[k] for k in ("t_b", "t_w", "r_f", "d_h")}


def evaluate(prm, ms, threads, req):
    r = run_design(BracketParams(**prm), ms=ms, threads=threads)
    s = r.summary(req)
    return dict(**prm, f1=float(s["f1"]), life_h=float(s["life_h"]), log10_life=float(np.log10(s["life_h"])),
                vm_max=float(s["vm_max"]), mass_kg=float(s["mass_kg"]), n_nodes=int(s["n_nodes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mc", type=int, default=200_000)
    args = ap.parse_args()
    req = Requirements()
    out = ROOT / "cad/results"
    out.mkdir(parents=True, exist_ok=True)
    nom = released()
    ms = design_mesh_settings(BracketParams(**nom))   # same mesh settings for every run
    rows = [dict(run="nominal", **evaluate(nom, ms, args.threads, req))]
    print(rows[-1], flush=True)
    for k in TOL:
        for sgn in (-1, 1):
            prm = dict(nom)
            prm[k] += sgn * STEP[k]
            rows.append(dict(run=f"{k}{'+' if sgn > 0 else '-'}", **evaluate(prm, ms, args.threads, req)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    n0 = df.iloc[0]

    sens, lin = {}, {}
    for k in TOL:
        hi = df[df.run == f"{k}+"].iloc[0]
        lo = df[df.run == f"{k}-"].iloc[0]
        sens[k] = {o: float((hi[o] - lo[o]) / (2 * STEP[k])) for o in OUTS}
        # linearity check: forward vs backward difference
        lin[k] = {o: [float((hi[o] - n0[o]) / STEP[k]), float((n0[o] - lo[o]) / STEP[k])] for o in OUTS}

    worst = {}
    for o, bad in (("f1", -1), ("log10_life", -1), ("vm_max", +1)):
        delta = sum(abs(sens[k][o]) * TOL[k] for k in TOL)
        corner = {k: nom[k] + bad * np.sign(sens[k][o]) * TOL[k] for k in TOL}
        worst[o] = dict(linear_nominal=float(n0[o]), linear_worst=float(n0[o] + bad * delta),
                        rss_3sigma=float(np.sqrt(sum((sens[k][o] * TOL[k]) ** 2 for k in TOL))), corner=corner)

    rng = np.random.default_rng(0)
    X = {k: rng.normal(0, TOL[k] / 3, args.mc) for k in TOL}
    f1 = n0["f1"] + sum(sens[k]["f1"] * X[k] for k in TOL)
    ll = n0["log10_life"] + sum(sens[k]["log10_life"] * X[k] for k in TOL)
    mc = dict(n=args.mc, f1_p001=float(np.quantile(f1, 0.001)), f1_mean=float(f1.mean()), f1_std=float(f1.std()),
              life_p001_h=float(10 ** np.quantile(ll, 0.001)), life_median_h=float(10 ** np.median(ll)),
              p_f1_below_min=float((f1 < req.f1_min).mean()),
              p_life_below_min_design_mesh=float((10 ** ll < req.life_min_h).mean()),
              p_life_below_margin=float((10 ** ll < 1.5 * req.life_min_h).mean()))

    # FEA at the worst-case corners
    corner_rows = []
    for o in ("f1", "log10_life"):
        c = worst[o]["corner"]
        rr = evaluate(c, ms, args.threads, req)
        corner_rows.append(dict(run=f"worst_{o}", **rr))
        worst[o]["fea_at_corner"] = float(rr[o])
        print(corner_rows[-1], flush=True)
    df = pd.concat([df, pd.DataFrame(corner_rows)], ignore_index=True)
    df.to_csv(out / "tolerance_runs.csv", index=False)
    res = dict(nominal=nom, tolerance=TOL, fd_step=STEP, mesh=ms.__dict__, sensitivity=sens,
               linearity_fwd_bwd=lin, worst_case=worst, monte_carlo=mc,
               note="design-study mesh; fine-mesh life is 20-35 % lower (see fea_roundtrip.json)")
    (out / "tolerance_study.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({"sens": sens, "worst": worst, "mc": mc}, indent=1, default=float))


if __name__ == "__main__":
    main()
