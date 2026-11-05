"""Parametric FreeCAD model of the bracket (PartDesign feature tree driven by a spreadsheet).

Run with FreeCAD's command-line interpreter, not the project venv (runpy avoids freecadcmd running a
script file twice):

    freecadcmd -c "import runpy; runpy.run_path('cad/build_part.py', run_name='__main__')"

Environment variables (all optional):
    BRACKET_PARAMS   JSON file with t_b, t_w, r_f, d_h (default cad/released_design.json)
    BRACKET_OUT      output folder (default cad/out)
    BRACKET_NAME     file stem (default "bracket")

Writes <name>.FCStd (editable model), <name>.step, <name>.stl and <name>_info.json.
Gussets are not modelled here: the released design has none.
"""

import json
import math
import os
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part
import Sketcher

HERE = Path(__file__).resolve().parent
V = App.Vector

FIXED = dict(W=70.0, H=100.0, L_b=100.0, x_bolt=80.0, y_bolt=20.0, d_bolt=9.0, z_mount=78.0, d_mount=11.0)
RHO = 7.85e-6  # kg/mm^3, AISI 1020
ALIAS_KEY = {"width": "W", "height": "H"}   # "W" and "H" are unit symbols, not allowed as aliases


def add_params(doc, params):
    sheet = doc.addObject("Spreadsheet::Sheet", "Params")
    rows = [("t_b", "base plate thickness"), ("t_w", "upright thickness"), ("r_f", "inner fillet radius"),
            ("d_h", "lightening hole diameter (0 = none)"), ("width", "width W"), ("height", "overall height H"),
            ("L_b", "base length"), ("x_bolt", "bolt hole distance from back face"),
            ("y_bolt", "bolt hole offset from centre plane"), ("d_bolt", "M8 clearance hole, ISO 273 medium"),
            ("z_mount", "mounting hole height"), ("d_mount", "M10 clearance hole, ISO 273 medium")]
    sheet.set("A1", "name")
    sheet.set("B1", "value")
    sheet.set("C1", "description")
    for i, (name, desc) in enumerate(rows, start=2):
        sheet.set(f"A{i}", name)
        sheet.set(f"B{i}", f"{params[ALIAS_KEY.get(name, name)]} mm")
        sheet.set(f"C{i}", desc)
        sheet.setAlias(f"B{i}", name)
    i = len(rows) + 2
    sheet.set(f"A{i}", "z_hole")
    sheet.set(f"B{i}", "=(t_b + r_f + z_mount - d_mount / 2) / 2")
    sheet.set(f"C{i}", "lightening hole height: midway between fillet and mounting hole")
    sheet.setAlias(f"B{i}", "z_hole")
    return sheet


def named(sk, constraint, name, expr=None):
    idx = sk.addConstraint(constraint)
    sk.renameConstraint(idx, name)
    if expr:
        sk.setExpression(f"Constraints.{name}", expr)
    return idx


def profile_sketch(body, origin, p):
    """L-shaped side profile in the XZ plane (sketch x = global X, sketch y = global Z)."""
    sk = body.newObject("Sketcher::SketchObject", "ProfileSketch")
    sk.AttachmentSupport = ([f for f in origin.OriginFeatures if f.Role == "XZ_Plane"][0], [""])
    sk.MapMode = "FlatFace"
    tb, tw, rf, L, H = p["t_b"], p["t_w"], p["r_f"], p["L_b"], p["H"]
    c = V(tw + rf, tb + rf, 0)
    g = []
    g.append(sk.addGeometry(Part.LineSegment(V(0, 0, 0), V(L, 0, 0))))             # 0 bottom
    g.append(sk.addGeometry(Part.LineSegment(V(L, 0, 0), V(L, tb, 0))))            # 1 base end
    g.append(sk.addGeometry(Part.LineSegment(V(L, tb, 0), V(tw + rf, tb, 0))))     # 2 base top
    arc = Part.ArcOfCircle(Part.Circle(c, V(0, 0, 1), rf), math.pi, 1.5 * math.pi)
    g.append(sk.addGeometry(arc))                                                    # 3 fillet (start at upright)
    g.append(sk.addGeometry(Part.LineSegment(V(tw, tb + rf, 0), V(tw, H, 0))))     # 4 upright front
    g.append(sk.addGeometry(Part.LineSegment(V(tw, H, 0), V(0, H, 0))))            # 5 top
    g.append(sk.addGeometry(Part.LineSegment(V(0, H, 0), V(0, 0, 0))))             # 6 back
    C = Sketcher.Constraint
    sk.addConstraint(C("Coincident", 0, 2, 1, 1))
    sk.addConstraint(C("Coincident", 1, 2, 2, 1))
    sk.addConstraint(C("Coincident", 4, 2, 5, 1))
    sk.addConstraint(C("Coincident", 5, 2, 6, 1))
    sk.addConstraint(C("Coincident", 6, 2, 0, 1))
    sk.addConstraint(C("Coincident", 0, 1, -1, 1))  # back-bottom corner on the origin
    for i in (0, 2, 5):
        sk.addConstraint(C("Horizontal", i))
    for i in (1, 4, 6):
        sk.addConstraint(C("Vertical", i))
    sk.addConstraint(C("Tangent", 2, 2, 3, 2))      # endpoint tangency also makes the points coincide
    sk.addConstraint(C("Tangent", 3, 1, 4, 1))
    named(sk, C("DistanceX", 0, 1, 0, 2, L), "L_b", "Params.L_b")
    named(sk, C("DistanceY", 1, 1, 1, 2, tb), "t_b", "Params.t_b")
    named(sk, C("DistanceX", 5, 2, 5, 1, tw), "t_w", "Params.t_w")
    named(sk, C("DistanceY", 6, 2, 6, 1, H), "height", "Params.height")
    named(sk, C("Radius", 3, rf), "r_f", "Params.r_f")
    return sk


def hole_sketch(body, origin, plane, name, circles):
    """circles: list of (u, v, d, name_prefix, expr_u, expr_v, expr_d) in sketch coordinates."""
    sk = body.newObject("Sketcher::SketchObject", name)
    sk.AttachmentSupport = ([f for f in origin.OriginFeatures if f.Role == plane][0], [""])
    sk.MapMode = "FlatFace"
    C = Sketcher.Constraint
    for (u, v, d, nm, eu, ev, ed) in circles:
        i = sk.addGeometry(Part.Circle(V(u, v, 0), V(0, 0, 1), d / 2))
        named(sk, C("Diameter", i, d), f"{nm}_d", ed)
        named(sk, C("DistanceX", -1, 1, i, 3, u), f"{nm}_u", eu)
        named(sk, C("DistanceY", -1, 1, i, 3, v), f"{nm}_v", ev)
    return sk


def symmetric(feat):
    if "SideType" in feat.PropertiesList:
        feat.SideType = [v for v in feat.getEnumerationsOfProperty("SideType") if "ymmetric" in v][0]
    else:  # FreeCAD < 1.1
        feat.Midplane = True


def through_pocket(body, sk, name):
    pk = body.newObject("PartDesign::Pocket", name)
    pk.Profile = sk
    pk.Type = "ThroughAll"
    symmetric(pk)
    sk.Visibility = False
    return pk


def clearance_hole(body, sk, name, size):
    """ISO 273 medium clearance hole (PartDesign Hole feature); falls back to a pocket."""
    try:
        h = body.newObject("PartDesign::Hole", name)
        h.Profile = sk
        h.ThreadType = "ISOMetricProfile"
        h.ThreadSize = size
        h.Threaded = False
        h.ThreadFit = "Medium"
        h.DepthType = "ThroughAll"
        h.HoleCutType = "None"
        h.DrillPoint = "Flat"
        h.Refine = True
        sk.Visibility = False
        return h
    except Exception as e:  # pragma: no cover - depends on FreeCAD version
        print("Hole feature failed, using a pocket:", e)
        return through_pocket(body, sk, name)


def make_hole(doc, body, sk, name, size, expected):
    """Hole feature (drilled away from the sketch plane); check the removed volume, else use a pocket."""
    before = body.Tip.Shape.Volume
    h = clearance_hole(body, sk, name, size)
    doc.recompute()
    for rev in (False, True):
        if h.TypeId != "PartDesign::Hole":
            break
        h.Reversed = rev
        doc.recompute()
        if h.Shape.isValid() and abs((before - h.Shape.Volume) - expected) < 0.01 * expected:
            return h
    print(f"{name}: Hole feature did not cut as expected, using a pocket")
    body.removeObject(h)
    doc.removeObject(h.Name)
    pk = through_pocket(body, sk, name)
    doc.recompute()
    return pk


def build(params: dict, out_dir: Path, stem: str) -> dict:
    p = dict(FIXED)
    p.update({k: float(params[k]) for k in ("t_b", "t_w", "r_f", "d_h")})
    doc = App.newDocument(stem)
    add_params(doc, p)
    doc.recompute()
    body = doc.addObject("PartDesign::Body", "Bracket")
    origin = body.Origin

    prof = profile_sketch(body, origin, p)
    pad = body.newObject("PartDesign::Pad", "Plate")
    pad.Profile = prof
    symmetric(pad)
    pad.setExpression("Length", "Params.width")
    prof.Visibility = False
    doc.recompute()

    # bolt holes: sketch on XY (sketch x = X, sketch y = Y)
    bolt_sk = hole_sketch(body, origin, "XY_Plane", "BoltHoleSketch", [
        (p["x_bolt"], p["y_bolt"], p["d_bolt"], "bolt1", "Params.x_bolt", "Params.y_bolt", "Params.d_bolt"),
        (p["x_bolt"], -p["y_bolt"], p["d_bolt"], "bolt2", "Params.x_bolt", "-Params.y_bolt", "Params.d_bolt")])
    bolt = make_hole(doc, body, bolt_sk, "BoltHoles", "M8x1.25",
                     expected=2 * math.pi * (p["d_bolt"] / 2) ** 2 * p["t_b"])

    # mounting hole: sketch on YZ (sketch x = Y, sketch y = Z)
    mount_sk = hole_sketch(body, origin, "YZ_Plane", "MountHoleSketch", [
        (0.0, p["z_mount"], p["d_mount"], "mount", None, "Params.z_mount", "Params.d_mount")])
    mount = make_hole(doc, body, mount_sk, "MountHole", "M10x1.5",
                      expected=math.pi * (p["d_mount"] / 2) ** 2 * p["t_w"])

    if p["d_h"] > 0:
        z_hole = (p["t_b"] + p["r_f"] + p["z_mount"] - p["d_mount"] / 2) / 2
        lh_sk = hole_sketch(body, origin, "YZ_Plane", "LighteningHoleSketch", [
            (0.0, z_hole, p["d_h"], "lhole", None, "Params.z_hole", "Params.d_h")])
        through_pocket(body, lh_sk, "LighteningHole")
    doc.recompute()

    shape = body.Shape
    sketches = [o for o in body.Group if o.TypeId == "Sketcher::SketchObject"]
    constrained = {s.Name: bool(s.FullyConstrained) for s in sketches}
    features = [(o.Name, o.TypeId.split("::")[-1]) for o in body.Group
                if o.TypeId.startswith("PartDesign::") and o.TypeId != "PartDesign::Body"]
    info = dict(params=p, volume_mm3=shape.Volume, mass_kg=shape.Volume * RHO,
                bbox=[shape.BoundBox.XMin, shape.BoundBox.YMin, shape.BoundBox.ZMin,
                      shape.BoundBox.XMax, shape.BoundBox.YMax, shape.BoundBox.ZMax],
                valid=shape.isValid(), n_solids=len(shape.Solids), n_faces=len(shape.Faces),
                hole_diameters={"bolt": getattr(bolt, "Diameter", None) and float(bolt.Diameter),
                                "mount": getattr(mount, "Diameter", None) and float(mount.Diameter)},
                features=features, sketch_fully_constrained=constrained, freecad=App.Version()[:3])

    out_dir.mkdir(parents=True, exist_ok=True)
    doc.saveAs(str(out_dir / f"{stem}.FCStd"))
    shape.exportStep(str(out_dir / f"{stem}.step"))
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.02, AngularDeflection=0.1, Relative=False)
    mesh.write(str(out_dir / f"{stem}.stl"))
    (out_dir / f"{stem}_info.json").write_text(json.dumps(info, indent=2))
    App.closeDocument(doc.Name)
    return info


def main():
    params_file = Path(os.environ.get("BRACKET_PARAMS", HERE / "released_design.json"))
    out_dir = Path(os.environ.get("BRACKET_OUT", HERE / "out"))
    stem = os.environ.get("BRACKET_NAME", "bracket")
    info = build(json.loads(params_file.read_text()), out_dir, stem)
    print(json.dumps({k: info[k] for k in ("volume_mm3", "mass_kg", "valid", "features", "hole_diameters")}))


if __name__ == "__main__":
    main()
