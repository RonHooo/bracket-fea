"""V2: finite-width plate with a central hole in tension, stress concentration factor vs Heywood.

Plate 300 x 100 x 2 mm (thin, close to plane stress), hole d = 20 mm (d/W = 0.2).
One end has u_x = 0, the other a prescribed u_x; the net-section stress comes from the reaction force.
Reference (Peterson's / Pilkey, Heywood fit, net section):
    K_tn = 2 + 0.284 (1 - d/W) - 0.600 (1 - d/W)^2 + 1.32 (1 - d/W)^3
"""

import argparse
from pathlib import Path

import gmsh
import numpy as np
import pandas as pd

from bracketfea.ccx import Material, write_mesh, write_material, run_ccx, read_frd, frd_field
from bracketfea.geometry import extract_tet10

LP, W, T, D = 300.0, 100.0, 2.0, 20.0
MAT = Material()


def kt_heywood(d_over_w: float) -> float:
    r = 1 - d_over_w
    return 2 + 0.284 * r - 0.600 * r ** 2 + 1.32 * r ** 3


def mesh_plate(h_hole: float, h_far: float = 8.0):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")
    occ = gmsh.model.occ
    box = occ.addBox(0, -W / 2, 0, LP, W, T)
    cyl = occ.addCylinder(LP / 2, 0, -1, 0, 0, T + 2, D / 2)
    occ.cut([(3, box)], [(3, cyl)])
    occ.synchronize()
    hole_surf = [s for _, s in gmsh.model.getEntities(2)
                 if abs(gmsh.model.getBoundingBox(2, s)[3] - gmsh.model.getBoundingBox(2, s)[0] - D) < 1e-3]
    fld = gmsh.model.mesh.field
    fd = fld.add("Distance")
    fld.setNumbers(fd, "SurfacesList", hole_surf)
    ft = fld.add("Threshold")
    fld.setNumber(ft, "InField", fd)
    fld.setNumber(ft, "SizeMin", h_hole)
    fld.setNumber(ft, "SizeMax", h_far)
    fld.setNumber(ft, "DistMin", 0.5)
    fld.setNumber(ft, "DistMax", 30)
    fld.setAsBackgroundMesh(ft)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(3)
    nodes, tets, _ = extract_tet10()
    gmsh.finalize()
    return nodes, tets


def run(h_hole: float, work: Path) -> dict:
    nodes, tets = mesh_plate(h_hole)
    n = len(nodes)
    left = np.where(np.isclose(nodes[:, 0], 0))[0]
    right = np.where(np.isclose(nodes[:, 0], LP))[0]
    corner = left[np.argmin(np.hypot(nodes[left, 1], nodes[left, 2]))]
    side = left[np.argmax(nodes[left, 1])]
    with open(work / "plate.inp", "w") as f:
        write_mesh(f, nodes, tets, {"LEFT": left, "RIGHT": right, "P0": [corner], "P1": [side]})
        write_material(f, MAT)
        f.write("*BOUNDARY\nLEFT, 1, 1\nP0, 2, 3\nP1, 3, 3\n")
        f.write("*STEP\n*STATIC\n*BOUNDARY\nRIGHT, 1, 1, 0.1\n")
        f.write("*NODE FILE\nU\n*EL FILE\nS\n*NODE PRINT, NSET=RIGHT, TOTALS=ONLY\nRF\n*END STEP\n")
    run_ccx(work, "plate", threads=4)
    txt = (work / "plate.dat").read_text().splitlines()
    k = [i for i, l in enumerate(txt) if "total force" in l][0]
    j = k + 1
    while not txt[j].strip():
        j += 1
    F = float(txt[j].split()[0])
    S = frd_field([b for b in read_frd(work / "plate.frd") if b.name == "STRESS"][0], n)
    # hole edge at the net section (y = +/- d/2), mid-thickness and surface
    r = np.hypot(nodes[:, 0] - LP / 2, nodes[:, 1])
    edge = (np.abs(r - D / 2) < 1e-6) & (np.abs(nodes[:, 0] - LP / 2) < 0.25 * h_hole + 1e-6)
    sigma_net = F / ((W - D) * T)
    return dict(h_hole=h_hole, n_nodes=n, sigma_net=sigma_net, sxx_peak=float(S[edge, 0].max()),
                Kt=float(S[edge, 0].max() / sigma_net))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification")
    args = ap.parse_args()
    out = Path(args.out)
    work = out / "work_v2"
    work.mkdir(parents=True, exist_ok=True)
    rows = [run(h, work) for h in (4.0, 2.0, 1.0, 0.5, 0.25)]
    df = pd.DataFrame(rows)
    df["Kt_ref"] = kt_heywood(D / W)
    df["Kt_err_%"] = 100 * (df["Kt"] - df["Kt_ref"]) / df["Kt_ref"]
    df.to_csv(out / "v2_plate_hole.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
