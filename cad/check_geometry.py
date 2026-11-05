"""Check that the FreeCAD model and the gmsh model used by the FE analysis are the same solid.

For several parameter sets, build the part in FreeCAD (cad/build_part.py), then compare its volume and
bounding box with the native gmsh/OpenCASCADE geometry from bracketfea.geometry.build_geometry, and
with the STEP file re-imported into gmsh.

    .venv/bin/python cad/check_geometry.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import gmsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bracketfea.geometry import BracketParams, build_geometry  # noqa: E402

OUT = ROOT / "cad/out/geometry_check"
CASES = {
    "released": dict(t_b=10.0, t_w=7.0, r_f=8.0, d_h=10.0),
    "optimizer": dict(t_b=9.902, t_w=7.071, r_f=8.598, d_h=9.738),
    "baseline": dict(t_b=6.0, t_w=6.0, r_f=5.0, d_h=0.0),
    "screened": dict(t_b=9.143, t_w=7.407, r_f=7.31, d_h=12.716),
}


def freecad_build(name, params):
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / f"{name}_params.json"
    pf.write_text(json.dumps(params))
    env = dict(os.environ, BRACKET_PARAMS=str(pf), BRACKET_OUT=str(OUT), BRACKET_NAME=name)
    subprocess.run([str(ROOT / "cad/fc.sh"), str(ROOT / "cad/build_part.py")], env=env, check=True,
                   capture_output=True)
    return json.loads((OUT / f"{name}_info.json").read_text())


def gmsh_native(params):
    gmsh.clear()
    vol = build_geometry(BracketParams(**params))
    return gmsh.model.occ.getMass(3, vol), gmsh.model.getBoundingBox(3, vol)


def gmsh_step(path):
    gmsh.clear()
    ents = gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    vols = [t for d, t in ents if d == 3]
    return sum(gmsh.model.occ.getMass(3, v) for v in vols), gmsh.model.getBoundingBox(3, vols[0]), len(vols)


def main():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    rows = []
    for name, prm in CASES.items():
        info = freecad_build(name, prm)
        v_n, bb_n = gmsh_native(prm)
        v_s, bb_s, n_s = gmsh_step(OUT / f"{name}.step")
        bb_fc = info["bbox"]
        rows.append(dict(case=name, **prm, mass_kg=info["mass_kg"], v_freecad=info["volume_mm3"],
                         v_gmsh_native=v_n, v_step_in_gmsh=v_s, n_step_volumes=n_s,
                         rel_diff_native=info["volume_mm3"] / v_n - 1, rel_diff_step=v_s / v_n - 1,
                         bbox_max_diff=max(abs(a - b) for a, b in zip(bb_fc, bb_n)),
                         bbox_step_max_diff=max(abs(a - b) for a, b in zip(bb_s, bb_n)),
                         sketch_fully_constrained=info["sketch_fully_constrained"], features=[f[1] for f in info["features"]]))
        r = rows[-1]
        print(f"{name:10s} V_freecad={r['v_freecad']:.3f}  V_gmsh={v_n:.3f}  rel={r['rel_diff_native']:+.2e}  "
              f"step rel={r['rel_diff_step']:+.2e}  bbox diff={r['bbox_max_diff']:.1e}  {r['features']}")
    gmsh.finalize()
    (ROOT / "cad/results").mkdir(exist_ok=True)
    (ROOT / "cad/results/geometry_check.json").write_text(json.dumps(rows, indent=2))
    worst = max(abs(r["rel_diff_native"]) for r in rows)
    print(f"worst relative volume difference: {worst:.2e}")
    assert worst < 1e-6, "FreeCAD and gmsh geometries differ"
    assert all(all(r["sketch_fully_constrained"].values()) for r in rows), "a sketch is not fully constrained"


if __name__ == "__main__":
    main()
