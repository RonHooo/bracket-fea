"""Design of experiments: Latin hypercube over the 6 design variables, run in parallel.

Writes one row per design to results/doe/doe_results.csv and, for the GNN surrogate, one compressed
.npz per design (corner-node mesh + static fields + scalars) to --npz-dir.

Sampling (u ~ LHS on [0, 1]^6; 600-point LHS, the first 400 rows are run):
    t_b in [4, 12], t_w in [4, 10], r_f in [2, 10]
    gussets: none if u_4 < 0.3, else L_g in [20, 45]; t_g in [3, 8]
    lightening hole: none if u_6 < 0.5, else d_h in [8, 20]
The "ood" set pushes one or two variables outside these ranges (extrapolation test for the surrogate).
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

PARAM_NAMES = ["t_b", "t_w", "r_f", "L_g", "t_g", "d_h"]


def map_unit(u: np.ndarray) -> dict:
    p = dict(t_b=4 + 8 * u[0], t_w=4 + 6 * u[1], r_f=2 + 8 * u[2],
             L_g=0.0 if u[3] < 0.3 else 20 + 25 * (u[3] - 0.3) / 0.7,
             t_g=3 + 5 * u[4],
             d_h=0.0 if u[5] < 0.5 else 8 + 12 * (u[5] - 0.5) / 0.5)
    return {k: round(float(v), 3) for k, v in p.items()}


def sample_designs(n: int, seed: int) -> list[dict]:
    u = qmc.LatinHypercube(d=6, seed=seed).random(n)
    return [map_unit(row) for row in u]


def sample_ood(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    pushes = [("t_b", 12, 14), ("t_w", 10, 12), ("t_w", 3, 4), ("t_b", 3, 4),
              ("L_g", 45, 50), ("t_g", 8, 10), ("r_f", 10, 12), ("d_h", 20, 24)]
    out = []
    for i in range(n):
        p = map_unit(rng.uniform(size=6))
        k = rng.choice(len(pushes), size=1 + (i % 2), replace=False)
        for j in k:
            name, lo, hi = pushes[j]
            if name == "L_g" and p["L_g"] == 0:
                p["t_g"] = round(float(rng.uniform(3, 8)), 3)
            if name == "d_h" and p["L_g"] > 0:
                p["t_g"] = min(p["t_g"], 6.0)
            p[name] = round(float(rng.uniform(lo, hi)), 3)
        if p["L_g"] > 0:
            p["L_g"] = max(p["L_g"], p["r_f"] + 10)
        out.append(p)
    return out


def sample_shift(n: int, seed: int) -> list[dict]:
    """Designs from a shifted geometry family: width, height, mounting-hole height and bolt
    position vary (they are fixed in the training DOE and are not among the 6 design variables)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = map_unit(rng.uniform(size=6))
        H = float(rng.uniform(90, 110))
        x_bolt = float(rng.uniform(74, 86))
        p.update(W=round(float(rng.uniform(64, 76)), 3), H=round(H, 3), z_mount=round(H - 22, 3),
                 x_bolt=round(x_bolt, 3), L_b=round(x_bolt + 20, 3))
        out.append(p)
    return out


def corner_mesh(mesh):
    """Corner-node sub-mesh (linear tets) for the GNN; returns (corner indices, tets remapped)."""
    corners = mesh.corner_nodes
    remap = -np.ones(mesh.n_nodes, dtype=np.int64)
    remap[corners] = np.arange(len(corners))
    return corners, remap[mesh.tets[:, :4]]


def run_one(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from bracketfea.analysis import run_design
    from bracketfea.geometry import BracketParams
    from bracketfea.ccx import von_mises

    row = dict(id=job["id"], split=job["split"], **job["params"])
    t0 = time.perf_counter()
    try:
        p = BracketParams(**job["params"])
        r = run_design(p, threads=1, workdir=job.get("workdir"))
        row.update(r.summary())
        row["status"] = "ok"
        row["vm_max_corner"] = float(r.vm_static[r.mesh.corner_nodes][r.region[r.mesh.corner_nodes]].max())
        if job.get("npz_dir"):
            corners, tets4 = corner_mesh(r.mesh)
            m = r.mesh
            flags = np.zeros((m.n_nodes, 4), dtype=np.uint8)
            flags[m.sets["FIX"], 0] = 1
            flags[m.sets["MOUNT"], 1] = 1
            flags[m.surface_nodes(), 2] = 1
            flags[r.region, 3] = 1
            np.savez_compressed(
                Path(job["npz_dir"]) / f"{job['split']}_{job['id']:04d}.npz",
                pos=m.nodes[corners].astype(np.float32),
                tets=tets4.astype(np.int32),
                flags=flags[corners],
                u=r.u_static[corners].astype(np.float32),
                s=r.s_static[corners].astype(np.float32),
                vm=r.vm_static[corners].astype(np.float32),
                mode1=r.modal_u[0][corners].astype(np.float32),
                params=np.array([job["params"][k] for k in PARAM_NAMES], dtype=np.float32),
                freq=r.freq.astype(np.float32),
                mass=np.float32(r.mass_kg),
                vm_max=np.float32(r.vm_max),
                life=np.array([r.random[a]["life_dirlik_h"] for a in "xyz"], dtype=np.float64),
                rms=np.array([r.random[a]["vm_rms_max"] for a in "xyz"], dtype=np.float32),
                n_nodes_full=np.int32(m.n_nodes),
                geom=np.str_(json.dumps(p.to_dict())))
    except Exception as e:  # keep going, record the failure
        row["status"] = f"failed: {type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-800:]
    row["t_total"] = time.perf_counter() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of LHS designs to run")
    ap.add_argument("--n-lhs", type=int, default=600, help="size of the LHS the designs are taken from")
    ap.add_argument("--n-ood", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="results/doe")
    ap.add_argument("--npz-dir", default="../gnn-fea-surrogate/data/raw")
    ap.add_argument("--workdir", default=None, help="scratch directory for CalculiX runs")
    ap.add_argument("--n-shift", type=int, default=0, help="also run designs from a shifted family")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    csv = out / "doe_results.csv"

    # the LHS rows come in random order, so the first n rows are a random subset of the n_lhs design
    designs = sample_designs(args.n_lhs, args.seed)[:args.n]
    # OOD designs first, then the DOE in LHS order: if the run is stopped early, the finished DOE
    # designs are still the first rows of the LHS (a random subset of it)
    jobs = [dict(id=i, split="ood", params=p) for i, p in enumerate(sample_ood(args.n_ood, args.seed + 1))]
    jobs += [dict(id=i, split="doe", params=p) for i, p in enumerate(designs)]
    jobs += [dict(id=i, split="shift", params=p) for i, p in enumerate(sample_shift(args.n_shift, args.seed + 2))]
    done = set()
    rows = []
    if csv.exists():
        prev = pd.read_csv(csv)
        rows = prev.to_dict("records")
        done = {(r["split"], int(r["id"])) for r in rows if r["status"] == "ok"}
        rows = [r for r in rows if (r["split"], int(r["id"])) in done]
    todo = [j for j in jobs if (j["split"], j["id"]) not in done]
    for j in todo:
        j["npz_dir"] = str(npz_dir)
        j["workdir"] = args.workdir
    print(f"{len(jobs)} designs, {len(done)} already done, running {len(todo)}", flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in todo]
        for k, fut in enumerate(as_completed(futs), start=1):
            row = fut.result()
            rows.append(row)
            el = time.time() - t_start
            print(f"[{k}/{len(todo)}] {row['split']} {row['id']} {row['status'][:60]} "
                  f"t={row['t_total']:.0f}s elapsed={el / 60:.1f} min", flush=True)
            if k % 10 == 0 or k == len(todo):
                pd.DataFrame(rows).to_csv(csv, index=False)
    pd.DataFrame(rows).sort_values(["split", "id"]).to_csv(csv, index=False)


if __name__ == "__main__":
    main()
