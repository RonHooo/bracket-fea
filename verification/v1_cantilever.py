"""V1: clamped cantilever (L = 200, b = 20, h = 10 mm, steel) against beam theory.

Checks C3D10 node ordering, units (mm, N, MPa, tonne, s), *RIGID BODY load introduction,
*MASS elements and the *FREQUENCY step.

  static   tip load P through a rigid end face: tip deflection (Timoshenko) and
           bending stress at mid-span
  modal    first bending frequencies in both directions (Euler-Bernoulli)
  tip mass point mass on the rigid end face: exact frequency from the frequency equation
"""

import argparse
from pathlib import Path

import gmsh
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from bracketfea.ccx import (Material, write_mesh, write_material, run_ccx, read_frd, frd_field,
                            read_dat_modes, read_dat_node_print)
from bracketfea.geometry import extract_tet10

L, B, H = 200.0, 20.0, 10.0
P = 100.0
M_TIP = 0.5e-3                      # tonne (0.5 kg)
MAT = Material()


def mesh_beam(h: float):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("beam")
    gmsh.model.occ.addBox(0, -B / 2, -H / 2, L, B, H)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(3)
    nodes, tets, _ = extract_tet10()
    gmsh.finalize()
    return nodes, tets


def analytic():
    I_z = B * H ** 3 / 12          # bending in z (thin direction)
    I_y = H * B ** 3 / 12
    A = B * H
    G_mod = MAT.E / (2 * (1 + MAT.nu))
    kappa = 10 * (1 + MAT.nu) / (12 + 11 * MAT.nu)   # Cowper shear coefficient, rectangle
    delta = P * L ** 3 / (3 * MAT.E * I_z) + P * L / (kappa * G_mod * A)
    sigma_mid = P * (L / 2) * (H / 2) / I_z
    m_line = MAT.rho * A

    def f_bend(beta_L, I):
        return beta_L ** 2 / (2 * np.pi * L ** 2) * np.sqrt(MAT.E * I / m_line)

    # cantilever with tip mass (no rotary inertia): frequency equation in x = beta L
    mu = M_TIP / (m_line * L)
    def eq(x):
        return 1 + np.cos(x) * np.cosh(x) + mu * x * (np.cos(x) * np.sinh(x) - np.sin(x) * np.cosh(x))
    x1 = brentq(eq, 0.3, 1.875)
    return dict(delta=delta, sigma_mid=sigma_mid,
                f1_z=f_bend(1.87510, I_z), f2_z=f_bend(4.69409, I_z), f1_y=f_bend(1.87510, I_y),
                f1_tip=f_bend(x1, I_z))


def run(h: float, work: Path) -> dict:
    nodes, tets = mesh_beam(h)
    n = len(nodes)
    clamp = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    end = np.where(np.isclose(nodes[:, 0], L))[0]
    ref, rot = n + 1, n + 2
    out = {"h": h, "n_nodes": n}

    for case in ("static", "tipmass"):
        inp = work / f"{case}.inp"
        with open(inp, "w") as f:
            write_mesh(f, nodes, tets, {"CLAMP": clamp, "END": end},
                       extra_nodes={ref: (L, 0, 0), rot: (L, 0, 0)})
            f.write(f"*NSET, NSET=REF\n{ref}\n")
            write_material(f, MAT)
            f.write(f"*RIGID BODY, NSET=END, REF NODE={ref}, ROT NODE={rot}\n")
            f.write("*BOUNDARY\nCLAMP, 1, 3\n")
            if case == "static":
                f.write(f"*STEP\n*STATIC\n*CLOAD\nREF, 3, {-P}\n")
                f.write("*NODE FILE\nU\n*EL FILE\nS\n*NODE PRINT, NSET=REF\nU\n*END STEP\n")
                # free vibration of the bare beam (rigid end face only, no mass)
                f.write("*STEP\n*FREQUENCY\n6\n*END STEP\n")
            else:
                f.write(f"*ELEMENT, TYPE=MASS, ELSET=EMASS\n{len(tets) + 1}, {ref}\n")
                f.write(f"*MASS, ELSET=EMASS\n{M_TIP:.6e}\n")
                f.write("*STEP\n*FREQUENCY\n4\n*END STEP\n")
        run_ccx(work, case, threads=4)

        md = read_dat_modes(work / f"{case}.dat")
        part = md["participation"]
        if case == "static":
            out["delta"] = -read_dat_node_print(work / f"{case}.dat", "REF")[0][2]
            S = frd_field([b for b in read_frd(work / f"{case}.frd") if b.name == "STRESS"][0], n)
            top = (np.abs(nodes[:, 0] - L / 2) < 1e-6) & np.isclose(nodes[:, 2], H / 2)
            if not top.any():
                top = (np.abs(nodes[:, 0] - L / 2) < h) & np.isclose(nodes[:, 2], H / 2) & (np.abs(nodes[:, 1]) < B / 4)
            out["sigma_mid"] = float(S[top, 0].mean())
            # bending modes: dominant participation in z or y
            fz = md["freq"][np.abs(part[:, 2]) > np.abs(part[:, 1])]
            fy = md["freq"][np.abs(part[:, 1]) > np.abs(part[:, 2])]
            out["f1_z"], out["f2_z"], out["f1_y"] = fz[0], fz[1], fy[0]
        else:
            out["f1_tip"] = md["freq"][0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification")
    args = ap.parse_args()
    out = Path(args.out)
    work = out / "work_v1"
    work.mkdir(parents=True, exist_ok=True)
    ref = analytic()
    rows = [run(h, work) for h in (5.0, 2.5, 1.25)]
    df = pd.DataFrame(rows)
    for k in ("delta", "sigma_mid", "f1_z", "f2_z", "f1_y", "f1_tip"):
        df[f"{k}_ref"] = ref[k]
        df[f"{k}_err_%"] = 100 * (df[k] - ref[k]) / ref[k]
    df.to_csv(out / "v1_cantilever.csv", index=False)
    cols = ["h", "n_nodes"] + [c for c in df.columns if c.endswith("err_%")]
    print({k: round(v, 4) for k, v in ref.items()})
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
