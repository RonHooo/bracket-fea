"""CAD -> FEA round trip: analyse the released bracket from the STEP file exported by FreeCAD.

1. Import cad/out/bracket.step into gmsh, find the bolt holes, the mounting hole and the fillet faces,
   mesh with the design-study settings and run the full analysis (static, modal, random vibration,
   fatigue).
2. Run the same design with the native gmsh geometry (what the design study used) and compare.
3. Re-check the STEP geometry on the finer mesh (element sizes x 0.7) used for the final check in
   studies/optimize.py, because the design-study mesh over-predicts fatigue life by 20-35 %.

    .venv/bin/python cad/fea_from_step.py --threads 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gmsh
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bracketfea import viz  # noqa: E402
from bracketfea.analysis import design_mesh_settings, run_design  # noqa: E402
from bracketfea.geometry import BracketParams, MeshSettings, mesh_volume  # noqa: E402
from bracketfea.problem import Requirements  # noqa: E402

KEYS = ("mass_kg", "vm_max", "f1", "f2", "life_x_h", "life_y_h", "life_z_h", "life_h", "n_nodes", "feasible")


def released_params() -> BracketParams:
    d = json.loads((ROOT / "cad/released_design.json").read_text())
    return BracketParams(**{k: d[k] for k in ("t_b", "t_w", "r_f", "d_h", "L_g")})


def classify_step_surfaces(p: BracketParams, vol: int) -> dict:
    """Surface groups of an imported bracket, found by surface type and position.

    Works whether or not the CAD kernel split a cylinder into several faces.
    """
    tol = 0.05
    out = {"bolt": [], "mount": [], "lhole": [], "refine": []}
    surfs = [s[1] for s in gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False)]
    for s in surfs:
        if gmsh.model.getType(2, s) == "Plane":
            continue
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, s)
        rb, rm, rh = p.d_bolt / 2, p.d_mount / 2, p.d_h / 2
        for sgn in (-1, 1):
            if (x0 > p.x_bolt - rb - tol and x1 < p.x_bolt + rb + tol and
                    y0 > sgn * p.y_bolt - rb - tol and y1 < sgn * p.y_bolt + rb + tol and
                    z0 > -tol and z1 < p.t_b + tol):
                out["bolt"].append(s)
                break
        else:
            in_upright = x0 > -tol and x1 < p.t_w + tol
            if in_upright and y0 > -rm - tol and y1 < rm + tol and z0 > p.z_mount - rm - tol and z1 < p.z_mount + rm + tol:
                out["mount"].append(s)
            elif (p.has_hole and in_upright and y0 > -rh - tol and y1 < rh + tol and
                  z0 > p.z_hole - rh - tol and z1 < p.z_hole + rh + tol):
                out["lhole"].append(s)
            elif x0 > p.t_w - tol and x1 < p.t_w + p.r_f + tol and z0 > p.t_b - tol and z1 < p.t_b + p.r_f + tol:
                out["refine"].append(s)
    if len(out["bolt"]) < 2 or not out["mount"] or not out["refine"]:
        raise RuntimeError(f"surface classification of the STEP file failed: {out}")
    return out


def mesh_step(step: Path, p: BracketParams, ms: MeshSettings):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("step")
        ents = gmsh.model.occ.importShapes(str(step))
        gmsh.model.occ.synchronize()
        vols = [t for d, t in ents if d == 3]
        if len(vols) != 1:
            raise RuntimeError(f"expected one solid in {step}, got {len(vols)}")
        groups = classify_step_surfaces(p, vols[0])
        counts = {k: len(v) for k, v in groups.items()}
        return mesh_volume(p, vols[0], groups, ms), counts
    finally:
        gmsh.finalize()


def summary(r, req):
    s = r.summary(req)
    return {k: (float(s[k]) if k != "feasible" else bool(s[k])) for k in KEYS}


def plots(r, out: Path, tag: str):
    nodes, tets = r.mesh.nodes, r.mesh.tets
    vm = np.where(r.region, r.vm_static, np.nan)
    viz.plot_field(nodes, tets, np.nan_to_num(vm, nan=0.0), str(out / f"{tag}_static_von_mises.png"),
                   title="Static von Mises, 5/5/10 g (MPa), mesh from the FreeCAD STEP file",
                   label="von Mises MPa", clim=(0, np.nanmax(vm)))
    worst = min(r.random, key=lambda a: r.random[a]["life_dirlik_h"])
    rr = r.random[worst]
    rms = np.zeros(r.mesh.n_nodes)
    rms[rr["nodes_idx"]] = rr["rms_map"]
    viz.plot_field(nodes, tets, rms, str(out / f"{tag}_rms_von_mises_{worst}.png"),
                   title=f"RMS von Mises, random vibration in {worst} (MPa)", label=f"RMS vM {worst} MPa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    req = Requirements()
    p = released_params()
    step = ROOT / "cad/out/bracket.step"
    out = ROOT / "cad/results"
    out.mkdir(parents=True, exist_ok=True)
    res = {"params": {k: getattr(p, k) for k in ("t_b", "t_w", "r_f", "d_h")}}

    base = design_mesh_settings(p)
    fine = MeshSettings(h_max=base.h_max * 0.7, h_min=base.h_min * 0.7, h_hole=base.h_hole * 0.7,
                        dist_min=base.dist_min, dist_max=base.dist_max)

    t0 = time.perf_counter()
    r_nat = run_design(p, ms=base, threads=args.threads)
    res["native_design_mesh"] = summary(r_nat, req)
    print("native  ", res["native_design_mesh"], f"{time.perf_counter() - t0:.0f} s", flush=True)

    t0 = time.perf_counter()
    m, counts = mesh_step(step, p, base)
    r_step = run_design(p, ms=base, threads=args.threads, mesh=m)
    res["step_design_mesh"] = summary(r_step, req)
    res["step_surface_groups"] = counts
    print("STEP    ", res["step_design_mesh"], counts, f"{time.perf_counter() - t0:.0f} s", flush=True)
    res["step_vs_native_rel"] = {k: res["step_design_mesh"][k] / res["native_design_mesh"][k] - 1
                                 for k in KEYS if k not in ("feasible", "n_nodes")}
    plots(r_step, out / "figures", "step")

    t0 = time.perf_counter()
    m, _ = mesh_step(step, p, fine)
    r_fine = run_design(p, ms=fine, threads=args.threads, mesh=m)
    res["step_fine_mesh"] = summary(r_fine, req)
    print("STEP fine", res["step_fine_mesh"], f"{time.perf_counter() - t0:.0f} s", flush=True)

    res["requirements"] = {"stress_allow_MPa": req.stress_allow, "f1_min_Hz": req.f1_min,
                           "life_min_h": req.life_min_h}
    (out / "fea_roundtrip.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["step_vs_native_rel"], indent=1))


if __name__ == "__main__":
    main()
