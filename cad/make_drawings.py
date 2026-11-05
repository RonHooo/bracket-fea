"""Engineering drawings with FreeCAD TechDraw: BRK-001 part drawing (sheet 1) and assembly drawing with
BOM and balloons (sheet 2). TechDraw needs the GUI to lay out and export pages, so run it with:

    cad/fc_gui.sh cad/make_drawings.py

Writes cad/out/drawings/BRK-001_drawing.FCStd (model + both pages, editable in FreeCAD) and PDF files.
GD&T frames and datum symbols are drawn as SVG symbols (TechDraw has no native feature control frame).
"""

import json
import math
import os
import time
import traceback
from pathlib import Path

import FreeCAD as App

HERE = Path(os.environ["BRACKET_CAD_DIR"])
OUT = HERE / "out/drawings"
LOG = OUT / "make_drawings.log"
TEMPLATES = Path(App.getResourceDir()) / "Mod/TechDraw/Templates/ASME"
V = App.Vector
FONT = "osifont, DejaVu Sans, sans-serif"


def log(*a):
    with open(LOG, "a") as f:
        f.write(" ".join(str(x) for x in a) + "\n")


def wait(cond=lambda: False, t=10.0):
    from PySide import QtWidgets
    t0 = time.time()
    while time.time() - t0 < t:
        QtWidgets.QApplication.processEvents()
        time.sleep(0.05)
        if cond():
            return True
    return False


# ----------------------------------------------------------------------------------------- views
class View:
    """A TechDraw part view plus helpers that find edges/vertices from 3D model coordinates."""

    def __init__(self, doc, page, name, source, direction, xdir, x, y, scale=1.0, hidden=True):
        self.obj = doc.addObject("TechDraw::DrawViewPart", name)
        page.addView(self.obj)
        self.obj.Source = source
        self.obj.Direction = direction
        self.obj.XDirection = xdir
        self.obj.ScaleType = "Custom"
        self.obj.Scale = scale
        self.obj.X, self.obj.Y = x, y
        self.obj.HardHidden = hidden
        bb = source[0].Shape.BoundBox
        for s in source[1:]:
            bb.add(s.Shape.BoundBox)
        self.c = bb.Center
        self.D = V(direction).normalize()
        self.X = V(xdir).normalize()
        self.U = self.D.cross(self.X).normalize()        # paper "up"
        self.s = scale
        self.off = (0.0, 0.0)
        self.source = source

    def calibrate(self):
        """TechDraw centres a view on the 2D box of the projected geometry, which for a pictorial view of
        an irregular assembly differs from the projected centre of the 3D box. Measure the offset."""
        self.off = (0.0, 0.0)
        mine = [self.uv(v.Point) for o in self.source for v in o.Shape.Vertexes]
        td = [(v.X, v.Y) for e in self.obj.getVisibleEdges() for v in e.Vertexes]
        cm = [(min(q[i] for q in mine) + max(q[i] for q in mine)) / 2 for i in (0, 1)]
        ct = [(min(q[i] for q in td) + max(q[i] for q in td)) / 2 for i in (0, 1)]
        self.off = (ct[0] - cm[0], ct[1] - cm[1])
        log(self.obj.Name, "centre offset", [round(x, 3) for x in self.off])

    def uv(self, p):
        """Model point -> TechDraw geometry coordinates (scaled, y pointing down)."""
        d = V(*p) - self.c
        return d.dot(self.X) * self.s + self.off[0], -d.dot(self.U) * self.s + self.off[1]

    def vertex(self, p, tol=0.05):
        u, v = self.uv(p)
        best, bi = 1e9, None
        for i, q in enumerate(self.obj.getVisibleVertexes()):
            dd = math.hypot(q.x - u, q.y - v)
            if dd < best:
                best, bi = dd, i
        if best > tol * max(1.0, self.s):
            raise RuntimeError(f"{self.obj.Name}: no vertex at {p} (closest {best:.3f})")
        return f"Vertex{bi}"

    def edge(self, a, b=None, radius=None, tol=0.05):
        """Line edge from a to b, or circle/arc with centre a and the given radius."""
        ua = self.uv(a)
        ub = self.uv(b) if b is not None else None
        for i, e in enumerate(self.obj.getVisibleEdges()):
            c = e.Curve
            if radius is not None:
                if hasattr(c, "Radius") and abs(c.Radius - radius * self.s) < tol and \
                        math.hypot(c.Center.x - ua[0], c.Center.y - ua[1]) < tol:
                    return f"Edge{i}"
            else:
                p0 = (e.Vertexes[0].X, e.Vertexes[0].Y)
                p1 = (e.Vertexes[-1].X, e.Vertexes[-1].Y)
                for q0, q1 in ((p0, p1), (p1, p0)):
                    if math.hypot(q0[0] - ua[0], q0[1] - ua[1]) < tol and math.hypot(q1[0] - ub[0], q1[1] - ub[1]) < tol:
                        return f"Edge{i}"
        raise RuntimeError(f"{self.obj.Name}: no edge {a} {b} r={radius}")

    def centre(self, p, radius):
        """Reference for a hole centre: the centre-mark vertex if there is one, else the circle edge."""
        try:
            return self.vertex(p, tol=0.02)
        except RuntimeError:
            return self.edge(p, radius=radius)

    def page_xy(self, p):
        """Model point -> page coordinates (mm, y up)."""
        u, v = self.uv(p)
        return self.obj.X.Value + u, self.obj.Y.Value - v


def dim(doc, page, view, kind, refs, pos, fmt=None, over=None, under=None, exact=False, name=None):
    """pos = (x, y) on the page; refs = list of 'EdgeN'/'VertexN' of `view`."""
    d = doc.addObject("TechDraw::DrawViewDimension", name or "Dim")
    d.Type = kind
    d.References2D = [(view.obj, r) for r in refs]
    page.addView(d)
    if fmt:
        d.FormatSpec = fmt
    if over is not None:
        d.OverTolerance = over
        d.UnderTolerance = under if under is not None else -over
        d.EqualTolerance = under is None
        if under == 0:
            d.FormatSpecUnderTolerance = "%.0f"
    if exact:
        d.TheoreticalExact = True
    d.X = pos[0] - view.obj.X.Value
    d.Y = pos[1] - view.obj.Y.Value
    return d


# ------------------------------------------------------------------------------- SVG annotations
def _svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">'
            f'<g fill="none" stroke="#000" stroke-width="0.35" font-family="{FONT}">{body}</g></svg>')


def _text(x, y, s, size=3.5, anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="#000" stroke="none">'
            f'{s}</text>')


def _sym_position(cx, cy, r=2.2):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}"/><line x1="{cx - r - 1}" y1="{cy}" x2="{cx + r + 1}" y2="{cy}"/>'
            f'<line x1="{cx}" y1="{cy - r - 1}" x2="{cx}" y2="{cy + r + 1}"/>')


def _sym_perp(cx, cy, r=2.5):
    return (f'<line x1="{cx - r}" y1="{cy + r}" x2="{cx + r}" y2="{cy + r}"/>'
            f'<line x1="{cx}" y1="{cy + r}" x2="{cx}" y2="{cy - r}"/>')


def _sym_flat(cx, cy, r=2.6):
    return f'<polygon points="{cx - r},{cy + 1.3} {cx + r - 1.2},{cy + 1.3} {cx + r},{cy - 1.3} {cx - r + 1.2},{cy - 1.3}"/>'


def _sym_dia(cx, cy, r=1.4):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}"/><line x1="{cx - 1.9}" y1="{cy + 1.9}" x2="{cx + 1.9}" y2="{cy - 1.9}"/>'


def _sym_mmc(cx, cy, r=1.7):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}"/>' + _text(cx, cy + 1.05, "M", 2.6)


def fcf(kind, tol, dia=False, mmc=False, datums=()):
    """Feature control frame as SVG. kind: position | perpendicularity | flatness."""
    h = 7.0
    cells = []
    sym = {"position": _sym_position, "perpendicularity": _sym_perp, "flatness": _sym_flat}[kind]
    tol_w = 4 + 2.2 * len(tol) + (4 if dia else 0) + (4.5 if mmc else 0)
    widths = [8.0, tol_w] + [7.0] * len(datums)
    w = sum(widths)
    body = f'<rect x="0.2" y="0.2" width="{w - 0.4}" height="{h - 0.4}"/>'
    x = 0.0
    for i, cw in enumerate(widths):
        if i:
            body += f'<line x1="{x}" y1="0.2" x2="{x}" y2="{h - 0.2}"/>'
        if i == 0:
            body += sym(x + cw / 2, h / 2)
        elif i == 1:
            xx = x + 2
            if dia:
                body += _sym_dia(xx + 1.5, h / 2)
                xx += 4
            body += _text(xx, h / 2 + 1.25, tol, 3.5, "start")
            if mmc:
                body += _sym_mmc(x + cw - 2.8, h / 2)
        else:
            body += _text(x + cw / 2, h / 2 + 1.25, datums[i - 2], 3.5)
        x += cw
    return _svg(w, h, body), w, h


def datum_symbol(letter, direction="down", leader=6.0):
    """Datum feature symbol: letter in a box, leader and filled triangle (pointing `direction`)."""
    b = 6.0
    if direction == "down":
        w, h = b, b + leader + 3
        body = (f'<rect x="0.2" y="0.2" width="{b - 0.4}" height="{b - 0.4}"/>' + _text(b / 2, b / 2 + 1.25, letter) +
                f'<line x1="{b / 2}" y1="{b}" x2="{b / 2}" y2="{b + leader}"/>'
                f'<polygon points="{b / 2 - 1.6},{b + leader} {b / 2 + 1.6},{b + leader} {b / 2},{h - 0.1}" fill="#000"/>')
    elif direction == "up":
        w, h = b, b + leader + 3
        body = (f'<polygon points="{b / 2},0.1 {b / 2 - 1.6},3 {b / 2 + 1.6},3" fill="#000"/>'
                f'<line x1="{b / 2}" y1="3" x2="{b / 2}" y2="{3 + leader}"/>'
                f'<rect x="0.2" y="{3 + leader}" width="{b - 0.4}" height="{b - 0.4}"/>' +
                _text(b / 2, 3 + leader + b / 2 + 1.25, letter))
    elif direction == "left":
        w, h = b + leader + 3, b
        body = (f'<polygon points="0.1,{b / 2} 3,{b / 2 - 1.6} 3,{b / 2 + 1.6}" fill="#000"/>'
                f'<line x1="3" y1="{b / 2}" x2="{3 + leader}" y2="{b / 2}"/>'
                f'<rect x="{3 + leader}" y="0.2" width="{b - 0.4}" height="{b - 0.4}"/>' +
                _text(3 + leader + b / 2, b / 2 + 1.25, letter))
    else:  # right
        w, h = b + leader + 3, b
        body = (f'<rect x="0.2" y="0.2" width="{b - 0.4}" height="{b - 0.4}"/>' + _text(b / 2, b / 2 + 1.25, letter) +
                f'<line x1="{b}" y1="{b / 2}" x2="{b + leader}" y2="{b / 2}"/>'
                f'<polygon points="{b + leader},{b / 2 - 1.6} {b + leader},{b / 2 + 1.6} {w - 0.1},{b / 2}" fill="#000"/>')
    return _svg(w, h, body), w, h


def text_block(lines, size=3.0, width=150, line_h=None, bold_first=False):
    line_h = line_h or size * 1.55
    h = line_h * len(lines) + 1
    body = ""
    for i, s in enumerate(lines):
        weight = ' font-weight="bold"' if (bold_first and i == 0) else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body += (f'<text x="0" y="{(i + 1) * line_h - 0.6}" font-size="{size}"{weight} fill="#000" stroke="none">'
                 f'{s}</text>')
    return _svg(width, h, body), width, h


def place_svg(doc, page, name, svg_wh, x, y, anchor="center"):
    """Place an SVG symbol with its centre (or top-left / bottom-left corner) at page (x, y)."""
    svg, w, h = svg_wh
    s = doc.addObject("TechDraw::DrawViewSymbol", name)
    s.Symbol = svg
    page.addView(s)
    if anchor == "topleft":
        x, y = x + w / 2, y - h / 2
    elif anchor == "bottomleft":
        x, y = x + w / 2, y + h / 2
    s.X, s.Y = x, y
    return s


def new_page(doc, name, title, subtitle, number, scale, weight, sheet):
    page = doc.addObject("TechDraw::DrawPage", name)
    tmpl = doc.addObject("TechDraw::DrawSVGTemplate", name + "Template")
    tmpl.Template = str(TEMPLATES / "ANSIC_Landscape.svg")
    page.Template = tmpl
    t = dict(tmpl.EditableTexts)
    t.update(Title=title, Subtitle=subtitle, AuthorName="H. HO", CreationDate="2026-11-04",
             SupervisorName="-", CheckDate="-", Company_name="PORTFOLIO PROJECT (NOT FOR PRODUCTION)",
             COPYRIGHT="github.com/RonHooo", drawing_number=f"{number}  REV A", scale=scale, Weight=weight,
             SheetNumber=sheet, A3="ANSI C", A__="A  RELEASE")
    for k in ("B__", "C__", "D__", "E__", "F__", "G__", "H__", "I__"):
        t[k] = ""
    tmpl.EditableTexts = t
    page.KeepUpdated = True
    return page


# ------------------------------------------------------------------------------------- sheet 1
def part_sheet(doc, body, prm, mass):
    tb, tw, rf, dh = prm["t_b"], prm["t_w"], prm["r_f"], prm["d_h"]
    W, H, L = 70.0, 100.0, 100.0
    xb, yb, db, zm, dm = 80.0, 20.0, 9.0, 78.0, 11.0
    zh = (tb + rf + zm - dm / 2) / 2
    page = new_page(doc, "Sheet1", "L-BRACKET", "COMPONENT MOUNT, MACHINED",
                    "BRK-001", "1:1", f"{mass:.3f}", "1 / 2")
    front = View(doc, page, "FrontView", [body], V(0, -1, 0), V(1, 0, 0), 150, 190)
    top = View(doc, page, "TopView", [body], V(0, 0, 1), V(1, 0, 0), 150, 330, hidden=False)
    right = View(doc, page, "RightView", [body], V(1, 0, 0), V(0, 1, 0), 330, 190)
    iso = View(doc, page, "IsoView", [body], V(1, -1, 1), V(1, 1, 0), 470, 305, scale=0.8, hidden=False)
    doc.recompute()
    wait(lambda: all(len(v.obj.getVisibleEdges()) > 0 for v in (front, top, right, iso)), 20)
    wait(t=1.0)
    for v in (front, top, right, iso):
        v.calibrate()

    # front view (x right, z up): overall sizes, thicknesses, fillet
    fx = lambda x, z: (x, -W / 2, z)       # point on the near face (y = -35)
    dim(doc, page, front, "DistanceX", [front.vertex(fx(0, 0)), front.vertex(fx(L, 0))],
        front.page_xy(fx(L / 2, -28)), fmt="%.0f")
    dim(doc, page, front, "DistanceY", [front.vertex(fx(0, 0)), front.vertex(fx(0, H))],
        front.page_xy(fx(-32, H / 2)), fmt="%.0f")
    dim(doc, page, front, "DistanceY", [front.vertex(fx(L, 0)), front.vertex(fx(L, tb))],
        front.page_xy(fx(L + 10, tb / 2)), fmt="%.1f")
    dim(doc, page, front, "DistanceX", [front.vertex(fx(0, H)), front.vertex(fx(tw, H))],
        front.page_xy(fx(tw / 2, H + 9)), fmt="%.1f")
    dim(doc, page, front, "Radius", [front.edge(fx(tw + rf, tb + rf), radius=rf)],
        front.page_xy(fx(tw + rf + 14, tb + rf + 14)), fmt="R%.0f", over=0.5)

    # right view (looking at the upright from +x; y right, z up)
    ry = lambda y, z: (tw, y, z)
    dim(doc, page, right, "Diameter", [right.edge(ry(0, zm), radius=dm / 2)],
        right.page_xy(ry(30, zm + 16)), fmt="⌀%.1f", over=0.2, under=0.0)
    dim(doc, page, right, "Diameter", [right.edge(ry(0, zh), radius=dh / 2)],
        right.page_xy(ry(30, zh + 12)), fmt="⌀%.1f")
    dim(doc, page, right, "DistanceX", [right.vertex((tw, -W / 2, H)), right.vertex((tw, W / 2, H))],
        right.page_xy(ry(0, H + 9)), fmt="%.0f")
    # mounting hole height from datum A (basic) and lightening hole height (toleranced)
    dim(doc, page, right, "DistanceY", [right.vertex((L, W / 2, 0)), right.centre(ry(0, zm), dm / 2)],
        right.page_xy(ry(W / 2 + 14, zm / 2)), fmt="%.0f", exact=True)
    dim(doc, page, right, "DistanceY", [right.vertex((L, -W / 2, 0)), right.centre(ry(0, zh), dh / 2)],
        right.page_xy(ry(-W / 2 - 12, zh / 2)), fmt="%.2f", over=0.5)

    # top view (x right, y up): bolt pattern, basic dimensions from datums B (x = 0) and C (y = -35)
    tz = lambda x, y: (x, y, tb)
    dim(doc, page, top, "Diameter", [top.edge(tz(xb, yb), radius=db / 2)],
        top.page_xy(tz(xb + 34, yb + 30)), fmt="2X ⌀%.1f THRU", over=0.2, under=0.0)
    dim(doc, page, top, "DistanceX", [top.vertex(tz(0, W / 2)), top.centre(tz(xb, yb), db / 2)],
        top.page_xy(tz(xb / 2, W / 2 + 12)), fmt="%.0f", exact=True)
    dim(doc, page, top, "DistanceY", [top.vertex(tz(L, -W / 2)), top.centre(tz(xb, -yb), db / 2)],
        top.page_xy(tz(L + 9, -W / 2 + (W / 2 - yb) / 2)), fmt="%.0f", exact=True)
    dim(doc, page, top, "DistanceY", [top.centre(tz(xb, -yb), db / 2), top.centre(tz(xb, yb), db / 2)],
        top.page_xy(tz(L + 9, 0)), fmt="%.0f", exact=True)
    dim(doc, page, top, "DistanceY", [top.vertex(tz(0, -W / 2)), top.vertex(tz(0, W / 2))],
        top.page_xy(tz(-12, 0)), fmt="%.0f")
    doc.recompute()

    # datums: A = bottom of base (front view), B = back face of upright (front view), C = near side (top view)
    xa, ya = front.page_xy(fx(25, 0))
    place_svg(doc, page, "DatumA", datum_symbol("A", "up", 6), xa, ya - 0.1 - (6 + 6 + 3) / 2)
    place_svg(doc, page, "FCF_flatA", fcf("flatness", "0.1"), xa + 5, ya - 9.1, anchor="topleft")
    xb_, yb_ = front.page_xy(fx(0, 60))
    place_svg(doc, page, "DatumB", datum_symbol("B", "left", 6), xb_ - 0.1 - (6 + 6 + 3) / 2, yb_)
    place_svg(doc, page, "FCF_perpB", fcf("perpendicularity", "0.1", datums=("A",)), xb_ - 26, yb_ - 5,
              anchor="topleft")
    xc, yc = top.page_xy(tz(40, -W / 2))
    place_svg(doc, page, "DatumC", datum_symbol("C", "up", 6), xc, yc - 0.1 - (6 + 6 + 3) / 2)

    # feature control frames under the hole callouts
    xd, yd = top.page_xy(tz(xb + 34, yb + 30))
    place_svg(doc, page, "FCF_bolt", fcf("position", "0.5", dia=True, mmc=True, datums=("A", "B", "C")),
              xd + 2, yd - 10, anchor="topleft")
    xm, ym = right.page_xy(ry(30, zm + 16))
    place_svg(doc, page, "FCF_mount", fcf("position", "1.0", dia=True, mmc=True, datums=("B", "A", "C")),
              xm + 12, ym - 3, anchor="topleft")

    centre_marks(front, top, right, prm)
    notes = ["NOTES (UNLESS OTHERWISE SPECIFIED):",
             "1. DIMENSIONS IN MILLIMETERS. TOLERANCES: X ±0.5   X.X ±0.2   X.XX ±0.1",
             "2. INTERPRET DIMENSIONS AND TOLERANCES PER ASME Y14.5-2018.",
             "3. MATERIAL: AISI 1020 HOT-ROLLED STEEL (Sy 210 MPa, Sut 380 MPa).",
             "4. FILLET R8 IS FATIGUE-CRITICAL: Ra 1.6 MAX, NO TOOL MARKS ACROSS THE FILLET.",
             "5. BREAK SHARP EDGES 0.2-0.5 EXCEPT FILLET R8.",
             "6. FINISH: BLACK OXIDE.",
             "7. BOXED DIMENSIONS ARE BASIC (THEORETICALLY EXACT).",
             f"8. MASS {mass:.3f} kg (REF). DESIGN VERIFIED BY FEA, SEE bracket-fea/cad/README."]
    place_svg(doc, page, "Notes", text_block(notes, 3.0, 180, bold_first=True), 25, 98, anchor="topleft")
    v = json.loads((HERE / "results/fea_roundtrip.json").read_text())["step_fine_mesh"]
    ver = ["DESIGN VERIFICATION (CalculiX FEA OF THIS CAD GEOMETRY, FINE MESH):",
           f"FIRST NATURAL FREQUENCY: {v['f1']:.1f} Hz (REQUIRED: MIN 100 Hz)",
           f"PEAK STATIC STRESS AT 5/5/10 g: {v['vm_max']:.1f} MPa (ALLOWED: MAX 140 MPa)",
           f"RANDOM VIBRATION FATIGUE LIFE: {v['life_h']:.0f} h (REQUIRED: MIN 1000 h)",
           "LOADS AND REQUIREMENTS ARE PROJECT ASSUMPTIONS."]
    place_svg(doc, page, "Verification", text_block(ver, 2.8, 170, bold_first=True), 220, 112, anchor="topleft")
    return page, (front, top, right, iso)


def centre_marks(front, top, right, prm):
    """Centre lines of the holes as cosmetic lines (dash-dot)."""
    tb, tw, rf, dh = prm["t_b"], prm["t_w"], prm["r_f"], prm["d_h"]
    zh = (tb + rf + 78.0 - 11.0 / 2) / 2
    marks = [(top, (80.0, 20.0, tb), 4.5), (top, (80.0, -20.0, tb), 4.5),
             (right, (tw, 0.0, 78.0), 5.5), (right, (tw, 0.0, zh), dh / 2)]
    for view, c, r in marks:
        e = r + 3.0
        for a, b in (((-e, 0), (e, 0)), ((0, -e), (0, e))):
            p0 = V(*c) + view.X * a[0] + view.U * a[1]
            p1 = V(*c) + view.X * b[0] + view.U * b[1]
            u0, v0 = view.uv(p0)
            u1, v1 = view.uv(p1)
            try:
                # cosmetic geometry takes y pointing up (unlike getVisibleEdges)
                view.obj.makeCosmeticLine(V(u0, -v0, 0), V(u1, -v1, 0), 4, 0.18)
            except Exception as ex:
                log("cosmetic line failed", ex)


# ------------------------------------------------------------------------------------- sheet 2
def assembly_sheet(doc, info):
    page = new_page(doc, "Sheet2", "BRACKET ASSEMBLY", "WITH BILL OF MATERIALS",
                    "ASM-001", "1:1", f"{info['moving_mass_kg']:.2f}", "2 / 2")
    asm_dir = HERE / "out/assembly"
    # rebuild instance shapes from the part files and placements in assembly_info.json
    adoc = App.openDocument(str(asm_dir / "bracket_assembly.FCStd"))
    objs = []
    for key, name, base in info["instances"]:
        link = adoc.getObject(name)
        sh = link.getLinkedObject(True).Shape.copy()
        sh.Placement = link.Placement.multiply(sh.Placement)
        o = doc.addObject("Part::Feature", "asm_" + name)
        o.Shape = sh
        o.Visibility = False
        objs.append(o)
    App.closeDocument(adoc.Name)
    doc.recompute()
    iso = View(doc, page, "AsmIso", objs, V(0.45, -1.0, 0.65), V(1, 0.45, 0), 185, 265, scale=1.0, hidden=False)
    front = View(doc, page, "AsmFront", objs, V(0, -1, 0), V(1, 0, 0), 450, 215, scale=0.5, hidden=True)
    side = View(doc, page, "AsmTop", objs, V(0, 0, 1), V(1, 0, 0), 450, 315, scale=0.5, hidden=False)
    doc.recompute()
    wait(lambda: all(len(v.obj.getVisibleEdges()) > 0 for v in (iso, front, side)), 30)
    wait(t=1.0)
    for v in (iso, front, side):
        v.calibrate()

    # balloons on the iso view
    items = [(1, "MountingPlate", (110, -45, -6)), (2, "Bracket", (3.5, -25, 95)), (3, "Component", (-40, -40, 100)),
             (4, "ScrewM8_1", (80, -20, 16.9)), (5, "WasherM8_1", (83.3, -27.3, 11.0)), (6, "ScrewM10", (15.4, 0, 84)),
             (7, "WasherM10", (9, -9, 78))]
    offsets = {1: (35, -25), 2: (-25, 45), 3: (-45, 30), 4: (40, 35), 5: (50, 12), 6: (45, 25), 7: (30, 45)}
    for n, inst, p in items:
        b = doc.addObject("TechDraw::DrawViewBalloon", f"Balloon{n}")
        b.SourceView = iso.obj
        page.addView(b)
        u, v = iso.uv(p)
        b.OriginX, b.OriginY = u, -v
        b.Text = str(n)
        b.BubbleShape = "Circular"
        b.X, b.Y = u + offsets[n][0], -v + offsets[n][1]
    # BOM table
    bom_rows = [("ITEM", "PART NUMBER", "DESCRIPTION", "MATERIAL", "QTY")]
    order = ["MNT-001", "BRK-001 rev A", "ECM-ENV-01", "ISO 4017 M8x20-8.8", "ISO 7089 8-200HV",
             "ISO 4017 M10x25-8.8", "ISO 7089 10-200HV"]
    by_pn = {r["part_no"]: r for r in info["bom"]}
    for i, pn in enumerate(order, start=1):
        r = by_pn[pn]
        bom_rows.append((str(i), pn, r["description"], r["material"], str(r["qty"])))
    place_svg(doc, page, "BOM", bom_table(bom_rows), 360, 140, anchor="topleft")
    place_svg(doc, page, "ViewLabels", text_block(["FRONT VIEW  1:2"], 3.0, 40), 425, 180, anchor="topleft")
    place_svg(doc, page, "ViewLabels3", text_block(["ISOMETRIC VIEW  1:1"], 3.0, 50), 150, 145, anchor="topleft")
    place_svg(doc, page, "ViewLabels2", text_block(["TOP VIEW  1:2"], 3.0, 40), 427, 280, anchor="topleft")
    notes = ["ASSEMBLY NOTES:",
             "1. TIGHTEN M8 CLASS 8.8 SCREWS (ITEM 4) TO 25 N·m, M10 (ITEM 6) TO 49 N·m (LIGHTLY OILED).",
             "2. MOUNTING FACE (DATUM A OF BRK-001) CLEAN AND DRY; NO PAINT UNDER THE BRACKET.",
             "3. ITEM 3 IS A 2 kg ENVELOPE OF THE SUPPLIED COMPONENT; ITS CG IS 30 mm BEHIND THE UPRIGHT.",
             "4. THREADS ARE COSMETIC: TAPPED HOLES ARE MODELLED AT THE TAP DRILL DIAMETER."]
    place_svg(doc, page, "AsmNotes", text_block(notes, 3.0, 215, bold_first=True), 25, 80, anchor="topleft")
    return page, (iso, front, side)


def bom_table(rows):
    cols = [12, 42, 70, 46, 12]
    rh = 6.0
    w, h = sum(cols), rh * len(rows)
    body = f'<rect x="0.2" y="0.2" width="{w - 0.4}" height="{h - 0.4}"/>'
    for i in range(1, len(rows)):
        sw = 0.5 if i == 1 else 0.25
        body += f'<line x1="0" y1="{i * rh}" x2="{w}" y2="{i * rh}" stroke-width="{sw}"/>'
    x = 0
    for cw in cols[:-1]:
        x += cw
        body += f'<line x1="{x}" y1="0" x2="{x}" y2="{h}"/>'
    for i, r in enumerate(rows):
        x = 0
        for j, (cw, s) in enumerate(zip(cols, r)):
            anchor, tx = ("middle", x + cw / 2) if j in (0, 4) else ("start", x + 1.5)
            weight = ' font-weight="bold"' if i == 0 else ""
            body += (f'<text x="{tx}" y="{i * rh + 4.2}" font-size="2.7"{weight} text-anchor="{anchor}" '
                     f'fill="#000" stroke="none">{s.replace("&", "&amp;")}</text>')
            x += cw
    return _svg(w, h, body), w, h


def main():
    import FreeCADGui as Gui
    import TechDrawGui
    OUT.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        LOG.unlink()
    prm = json.loads((HERE / "released_design.json").read_text())
    info = json.loads((HERE / "out/assembly/assembly_info.json").read_text())
    src = App.openDocument(str(HERE / "out/bracket.FCStd"))
    src.saveAs(str(OUT / "BRK-001_drawing.FCStd"))
    doc = src
    body = doc.getObject("Bracket")
    mass = body.Shape.Volume * 7.85e-6
    p1, views1 = part_sheet(doc, body, prm, mass)
    p2, views2 = assembly_sheet(doc, info)
    doc.recompute()
    wait(t=3.0)
    for p in (p1, p2):
        p.ViewObject.show()
        wait(t=2.0)
    doc.save()
    TechDrawGui.exportPageAsPdf(p1, str(OUT / "BRK-001_part_drawing.pdf"))
    TechDrawGui.exportPageAsPdf(p2, str(OUT / "ASM-001_assembly_drawing.pdf"))
    TechDrawGui.exportPageAsSvg(p1, str(OUT / "BRK-001_part_drawing.svg"))
    TechDrawGui.exportPageAsSvg(p2, str(OUT / "ASM-001_assembly_drawing.svg"))
    log("done")


try:
    main()
except Exception:
    log(traceback.format_exc())
os._exit(0)
