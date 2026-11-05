"""Bracket assembly: mounting plate + bracket + 2 kg component + fasteners, with an interference check.

Each part is its own FreeCAD file in cad/out/assembly/ and the assembly file links to them
(App::Link), the same structure as a SolidWorks .sldasm that references .sldprt files. Parts are
positioned with placements; no assembly constraint solver is used.

Run after build_part.py, with FreeCAD's Python:

    cad/fc.sh cad/build_assembly.py

Coordinates are the bracket's: x along the base plate, y across, z up, bottom of the base plate at z = 0,
back face of the upright at x = 0. The component hangs on the back face, its CG 30 mm behind it at the
mounting-hole height (as in the FE model).
"""

import json
import math
import shutil
from pathlib import Path

import FreeCAD as App
import Import
import MeshPart
import Part
import Sketcher

HERE = Path(__file__).resolve().parent
OUT = HERE / "out/assembly"
V = App.Vector
P = App.Placement
R = App.Rotation

RHO_STEEL = 7.85e-6   # kg/mm^3
COMP_MASS = 2.0       # kg

# ISO 4017 hex head screws, ISO 7089 washers (simplified: no thread geometry, cosmetic only)
SCREWS = {"M8": dict(d=8.0, s=13.0, k=5.3, washer=(8.4, 16.0, 1.6), tap=6.8),
          "M10": dict(d=10.0, s=16.0, k=6.4, washer=(10.5, 20.0, 2.0), tap=8.5)}


def plane(body, role):
    return ([f for f in body.Origin.OriginFeatures if f.Role == role][0], [""])


def rect_sketch(body, role, name, x0, y0, x1, y1):
    sk = body.newObject("Sketcher::SketchObject", name)
    sk.AttachmentSupport = plane(body, role)
    sk.MapMode = "FlatFace"
    pts = [V(x0, y0, 0), V(x1, y0, 0), V(x1, y1, 0), V(x0, y1, 0)]
    C = Sketcher.Constraint
    for i in range(4):
        sk.addGeometry(Part.LineSegment(pts[i], pts[(i + 1) % 4]))
    for i in range(4):
        sk.addConstraint(C("Coincident", i, 2, (i + 1) % 4, 1))
    sk.addConstraint(C("Horizontal", 0))
    sk.addConstraint(C("Horizontal", 2))
    sk.addConstraint(C("Vertical", 1))
    sk.addConstraint(C("Vertical", 3))
    sk.addConstraint(C("DistanceX", 0, 1, 0, 2, x1 - x0))
    sk.addConstraint(C("DistanceY", 1, 1, 1, 2, y1 - y0))
    sk.addConstraint(C("DistanceX", -1, 1, 0, 1, x0))
    sk.addConstraint(C("DistanceY", -1, 1, 0, 1, y0))
    return sk


def circles_sketch(body, role, name, centres, d):
    sk = body.newObject("Sketcher::SketchObject", name)
    sk.AttachmentSupport = plane(body, role)
    sk.MapMode = "FlatFace"
    C = Sketcher.Constraint
    for (u, v) in centres:
        i = sk.addGeometry(Part.Circle(V(u, v, 0), V(0, 0, 1), d / 2))
        sk.addConstraint(C("Diameter", i, d))
        sk.addConstraint(C("DistanceX", -1, 1, i, 3, u))
        sk.addConstraint(C("DistanceY", -1, 1, i, 3, v))
    return sk


def symmetric(feat):
    if "SideType" in feat.PropertiesList:
        feat.SideType = [v for v in feat.getEnumerationsOfProperty("SideType") if "ymmetric" in v][0]
    else:
        feat.Midplane = True


def new_part_doc(name):
    doc = App.newDocument(name)
    body = doc.addObject("PartDesign::Body", "Body")
    return doc, body


def mounting_plate():
    """Steel plate standing in for the engine-side structure, 2 x M8 tapped through holes."""
    doc, body = new_part_doc("mounting_plate")
    sk = rect_sketch(body, "XY_Plane", "OutlineSketch", -80.0, -60.0, 130.0, 60.0)
    pad = body.newObject("PartDesign::Pad", "Plate")
    pad.Profile = sk
    pad.Length = 12.0
    pad.Reversed = True          # top face on z = 0
    sk.Visibility = False
    doc.recompute()
    hs = circles_sketch(body, "XY_Plane", "TapHoleSketch", [(80.0, 20.0), (80.0, -20.0)], SCREWS["M8"]["tap"])
    h = body.newObject("PartDesign::Hole", "TappedHoles")
    h.Profile = hs
    h.ThreadType = "ISOMetricProfile"
    h.ThreadSize = "M8x1.25"
    h.Threaded = True
    h.ModelThread = False        # cosmetic thread: modelled at the tap drill diameter
    h.DepthType = "ThroughAll"
    hs.Visibility = False
    doc.recompute()
    before = pad.Shape.Volume
    if abs(before - h.Shape.Volume) < 1.0:
        h.Reversed = True
        doc.recompute()
    return doc, body, dict(part_no="MNT-001", description="Mounting plate 210 x 120 x 12 (engine-side structure)",
                           material="Steel S355", density=RHO_STEEL)


def component():
    """2 kg electronics module envelope: aluminium housing with an M10 tapped boss on its mounting face."""
    doc, body = new_part_doc("component")
    sk = rect_sketch(body, "YZ_Plane", "OutlineSketch", -40.0, 48.0, 40.0, 108.0)
    pad = body.newObject("PartDesign::Pad", "Housing")
    pad.Profile = sk
    pad.Length = 60.0
    pad.Reversed = True          # from x = 0 towards -x
    sk.Visibility = False
    doc.recompute()
    if pad.Shape.BoundBox.XMax > 1.0:
        pad.Reversed = False
        doc.recompute()
    edges = [f"Edge{i + 1}" for i, e in enumerate(pad.Shape.Edges)
             if abs(e.BoundBox.XLength - 60.0) < 1e-6]  # the four edges along x
    fil = body.newObject("PartDesign::Fillet", "EdgeFillets")
    fil.Base = (pad, edges)
    fil.Radius = 5.0
    doc.recompute()
    # connector on the far face
    cs = rect_sketch(body, "YZ_Plane", "ConnectorSketch", -18.0, 64.0, 18.0, 82.0)
    cs.AttachmentOffset = P(V(0, 0, -60.0), R())
    con = body.newObject("PartDesign::Pad", "Connector")
    con.Profile = cs
    con.Length = 12.0
    cs.Visibility = False
    doc.recompute()
    if con.Shape.BoundBox.XMin > -61.0:
        con.Reversed = True
        doc.recompute()
    hs = circles_sketch(body, "YZ_Plane", "TapHoleSketch", [(0.0, 78.0)], SCREWS["M10"]["tap"])
    h = body.newObject("PartDesign::Hole", "TappedBoss")
    h.Profile = hs
    h.ThreadType = "ISOMetricProfile"
    h.ThreadSize = "M10x1.5"
    h.Threaded = True
    h.ModelThread = False
    h.DepthType = "Dimension"
    h.Depth = 20.0
    hs.Visibility = False
    doc.recompute()
    if abs(con.Shape.Volume - h.Shape.Volume) < 1.0:
        h.Reversed = True
        doc.recompute()
    vol = body.Shape.Volume
    return doc, body, dict(part_no="ECM-ENV-01", description="Component envelope, 2 kg (supplied part)",
                           material="Al housing + electronics", density=COMP_MASS / vol)


def screw(size, length):
    s = SCREWS[size]
    doc = App.newDocument(f"screw_{size}x{int(length)}")
    r_hex = s["s"] / math.sqrt(3)      # across-corners radius
    hexagon = Part.makePolygon([V(r_hex * math.cos(math.pi / 6 + i * math.pi / 3),
                                  r_hex * math.sin(math.pi / 6 + i * math.pi / 3), 0) for i in range(7)])
    head = Part.Face(hexagon).extrude(V(0, 0, s["k"]))
    head = head.common(Part.makeCylinder(0.95 * s["s"] / 2 / math.cos(math.pi / 6) , s["k"]))  # chamfered look
    shank = Part.makeCylinder(s["d"] / 2, length, V(0, 0, -length))
    shape = head.fuse(shank).removeSplitter()
    obj = doc.addObject("Part::Feature", "Screw")
    obj.Shape = shape
    return doc, obj, dict(part_no=f"ISO 4017 {size}x{int(length)}-8.8", description=f"Hex head screw {size} x {int(length)}",
                          material="Steel 8.8, zinc plated", density=RHO_STEEL)


def washer(size):
    d1, d2, h = SCREWS[size]["washer"]
    doc = App.newDocument(f"washer_{size}")
    shape = Part.makeCylinder(d2 / 2, h).cut(Part.makeCylinder(d1 / 2, h))
    obj = doc.addObject("Part::Feature", "Washer")
    obj.Shape = shape
    return doc, obj, dict(part_no=f"ISO 7089 {size[1:]}-200HV", description=f"Plain washer {size}",
                          material="Steel 200HV", density=RHO_STEEL)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    released = json.loads((HERE / "released_design.json").read_text())
    t_b, t_w = released["t_b"], released["t_w"]

    parts = {}
    shutil.copy(HERE / "out/bracket.FCStd", OUT / "bracket.FCStd")
    bdoc = App.openDocument(str(OUT / "bracket.FCStd"))
    parts["bracket"] = (bdoc, bdoc.getObject("Bracket"),
                        dict(part_no="BRK-001 rev A", description="L-bracket, machined", material="AISI 1020 HR",
                             density=RHO_STEEL))
    for key, fn in (("plate", mounting_plate), ("component", component), ("screw_m8", lambda: screw("M8", 20)),
                    ("washer_m8", lambda: washer("M8")), ("screw_m10", lambda: screw("M10", 25)),
                    ("washer_m10", lambda: washer("M10"))):
        parts[key] = fn()
        parts[key][0].recompute()
        parts[key][0].saveAs(str(OUT / f"{parts[key][0].Name}.FCStd"))

    # instances: (part key, instance name, placement)
    w8, w10 = SCREWS["M8"]["washer"][2], SCREWS["M10"]["washer"][2]
    inst = [("plate", "MountingPlate", P()),
            ("bracket", "Bracket", P()),
            ("component", "Component", P())]
    for i, y in enumerate((20.0, -20.0), start=1):
        inst.append(("washer_m8", f"WasherM8_{i}", P(V(80.0, y, t_b), R())))
        inst.append(("screw_m8", f"ScrewM8_{i}", P(V(80.0, y, t_b + w8), R())))
    # M10 screw from the front of the upright (+x) into the tapped boss of the component
    # R(y, 90) turns the screw's local +z (head side) to global +x
    inst.append(("washer_m10", "WasherM10", P(V(t_w, 0.0, 78.0), R(V(0, 1, 0), 90))))
    inst.append(("screw_m10", "ScrewM10", P(V(t_w + w10, 0.0, 78.0), R(V(0, 1, 0), 90))))

    asm = App.newDocument("bracket_assembly")
    asm.saveAs(str(OUT / "bracket_assembly.FCStd"))   # links to other files need a saved owner document
    try:
        container = asm.addObject("Assembly::AssemblyObject", "Assembly")
        kind = "Assembly::AssemblyObject"
    except Exception:
        container = asm.addObject("App::Part", "Assembly")
        kind = "App::Part"
    shapes = {}
    for key, name, plc in inst:
        doc, obj, meta = parts[key]
        link = asm.addObject("App::Link", name)
        link.setLink(obj)
        link.Placement = plc
        container.addObject(link)
        sh = obj.Shape.copy()
        sh.Placement = plc.multiply(sh.Placement)
        shapes[name] = (key, sh)
    asm.recompute()
    asm.save()

    # interference and clearance checks
    names = list(shapes)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = shapes[names[i]][1], shapes[names[j]][1]
            bb = App.BoundBox(a.BoundBox)
            bb.enlarge(0.01)               # so that touching parts are checked too
            if not bb.intersect(b.BoundBox):
                continue
            v = a.common(b).Volume
            d = a.distToShape(b)[0]
            pairs.append(dict(a=names[i], b=names[j], overlap_mm3=v, min_distance_mm=d))
    # screw shank in a tapped hole modelled at the tap drill diameter overlaps by the thread depth
    tap_d = {"M8": float(parts["plate"][0].getObject("TappedHoles").Diameter),
             "M10": float(parts["component"][0].getObject("TappedBoss").Diameter)}

    def thread_overlap(size, engagement):
        return math.pi / 4 * (SCREWS[size]["d"] ** 2 - tap_d[size] ** 2) * engagement
    expected = {("MountingPlate", "ScrewM8_1"): thread_overlap("M8", 20 - w8 - t_b),
                ("MountingPlate", "ScrewM8_2"): thread_overlap("M8", 20 - w8 - t_b),
                ("Component", "ScrewM10"): thread_overlap("M10", 25 - w10 - t_w)}
    for pr in pairs:
        key = (pr["a"], pr["b"]) if (pr["a"], pr["b"]) in expected else (pr["b"], pr["a"])
        if key in expected:
            pr["expected_thread_overlap_mm3"] = expected[key]
            pr["status"] = "thread engagement (expected)" if abs(pr["overlap_mm3"] - expected[key]) < 0.02 * expected[key] \
                else "CHECK"
        else:
            pr["status"] = "interference" if pr["overlap_mm3"] > 1e-3 else (
                "contact" if pr["min_distance_mm"] < 1e-6 else "clear")
    # clearance between M8 shank and the bracket's 9.0 mm hole, M10 shank and the 11.0 mm hole
    br = shapes["Bracket"][1]
    clear = {n: shapes[n][1].distToShape(br)[0] for n in ("ScrewM8_1", "ScrewM8_2")}
    sh_m10 = Part.makeCylinder(5.0, t_w, V(0, 0, 78.0), V(1, 0, 0))
    clear["ScrewM10_shank_in_upright"] = sh_m10.distToShape(br)[0]

    # BOM and mass properties
    bom, total_mass = {}, 0.0
    for key, name, _ in inst:
        doc, obj, meta = parts[key]
        m = obj.Shape.Volume * meta["density"]
        row = bom.setdefault(meta["part_no"], dict(part_no=meta["part_no"], description=meta["description"],
                                                   material=meta["material"], qty=0, unit_mass_kg=m))
        row["qty"] += 1
        if key != "plate":
            total_mass += m
    cg = V(0, 0, 0)
    for key, name, _ in inst:
        if key == "plate":
            continue
        doc, obj, meta = parts[key]
        sh = shapes[name][1]
        cg += sh.Solids[0].CenterOfMass * (sh.Volume * meta["density"]) if sh.Solids else V()
    cg = cg * (1.0 / total_mass)

    # exports: one STEP for the assembly, one STL per instance (assembly coordinates) for rendering
    objs = []
    for key, name, plc in inst:
        f = asm.addObject("Part::Feature", f"exp_{name}")
        f.Shape = shapes[name][1]
        objs.append(f)
    Import.export(objs, str(OUT / "bracket_assembly.step"))
    (OUT / "stl").mkdir(exist_ok=True)
    for f, (key, name, plc) in zip(objs, inst):
        MeshPart.meshFromShape(Shape=f.Shape, LinearDeflection=0.05, AngularDeflection=0.15,
                               Relative=False).write(str(OUT / "stl" / f"{name}.stl"))
    for f in objs:
        asm.removeObject(f.Name)
    asm.save()

    bracket_obj = parts["bracket"][1]
    info = dict(container=kind, tapped_hole_diameters=tap_d, instances=[(k, n, list(p.Base)) for k, n, p in inst], interference=pairs,
                clearance_mm=clear, bom=list(bom.values()),
                moving_mass_kg=total_mass, moving_cg_mm=list(cg),
                bracket_cg_mm=list(bracket_obj.Shape.Solids[0].CenterOfMass),
                bracket_mass_kg=bracket_obj.Shape.Volume * RHO_STEEL)
    (OUT / "assembly_info.json").write_text(json.dumps(info, indent=2, default=float))
    for pr in pairs:
        print(f"{pr['a']:14s} {pr['b']:14s} overlap {pr['overlap_mm3']:9.2f} mm3  dist {pr['min_distance_mm']:.3f}  {pr['status']}")
    print("clearance", clear)
    print("moving mass", total_mass, "cg", cg)


if __name__ == "__main__":
    main()
