"""Parametric L-bracket geometry and quadratic tetrahedral meshing with gmsh (OpenCASCADE kernel).

Coordinate system (mm):
    x  along the base plate, the upright sits at 0 <= x <= t_w
    y  across the width, centred on y = 0
    z  vertical, the bottom of the base plate is z = 0
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import gmsh
import numpy as np


@dataclass
class BracketParams:
    # design variables
    t_b: float = 6.0      # base plate thickness
    t_w: float = 6.0      # upright thickness
    r_f: float = 5.0      # inner fillet radius at the base/upright corner
    L_g: float = 0.0      # gusset leg length (0 = no gussets)
    t_g: float = 5.0      # gusset thickness
    d_h: float = 0.0      # lightening hole diameter in the upright (0 = no hole)
    # fixed geometry
    W: float = 70.0       # width
    H: float = 100.0      # overall height
    L_b: float = 100.0    # base length
    x_bolt: float = 80.0  # bolt hole x position
    y_bolt: float = 20.0  # bolt holes at y = +/- y_bolt
    d_bolt: float = 9.0   # M8 clearance hole
    z_mount: float = 78.0  # component mounting hole height
    d_mount: float = 11.0  # M10 clearance hole
    y_gusset: float = 25.0  # gusset mid-planes at y = +/- y_gusset
    r_g: float = 2.0      # fillet radius on the gusset attachment edges

    @property
    def has_gussets(self) -> bool:
        return self.L_g > 0.0

    @property
    def has_hole(self) -> bool:
        return self.d_h > 0.0

    @property
    def z_hole(self) -> float:
        """Lightening hole centre: midway between the fillet and the mounting hole."""
        z_lo = self.t_b + self.r_f
        z_hi = self.z_mount - self.d_mount / 2
        return 0.5 * (z_lo + z_hi)

    def check(self) -> None:
        if self.has_gussets:
            if self.L_g < self.r_f + 2 * self.r_g + 5:
                raise ValueError("gusset too small compared with the fillet")
            if self.t_b + self.L_g > self.z_mount - self.d_mount / 2 - 3:
                raise ValueError("gusset reaches the mounting hole")
            if self.y_gusset + self.t_g / 2 + self.r_g > self.W / 2 - 1:
                raise ValueError("gusset too close to the plate edge")
        if self.has_hole:
            gap = (self.z_mount - self.d_mount / 2) - (self.t_b + self.r_f)
            if self.d_h > gap - 6:
                raise ValueError("lightening hole does not fit")
            if self.has_gussets and self.d_h / 2 > self.y_gusset - self.t_g / 2 - self.r_g - 3:
                raise ValueError("lightening hole hits the gussets")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeshSettings:
    h_max: float = 4.0     # element size far from hot spots
    h_min: float = 1.0     # element size at the fillet and gusset edges
    h_hole: float = 1.5    # element size at the holes
    dist_min: float = 1.0  # size field transition (distance from refined surfaces)
    dist_max: float = 12.0
    order: int = 2


@dataclass
class BracketMesh:
    """Mesh in CalculiX ordering. Node ids are 1..N (row i of `nodes` is node i+1)."""
    nodes: np.ndarray                 # (N, 3)
    tets: np.ndarray                  # (E, 10) 0-based node indices, CalculiX C3D10 order
    sets: dict = field(default_factory=dict)   # name -> 0-based node index arrays
    volume: float = 0.0               # mm^3
    params: BracketParams | None = None

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def corner_nodes(self) -> np.ndarray:
        return np.unique(self.tets[:, :4])

    def surface_nodes(self) -> np.ndarray:
        """Nodes on the boundary faces (faces that belong to exactly one tetrahedron)."""
        # 6-node faces of C3D10 (corners + mid-side nodes); first 3 entries are the corners
        faces = np.array([[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7], [1, 2, 3, 5, 9, 8], [0, 2, 3, 6, 9, 7]])
        if self.tets.shape[1] == 4:
            faces = faces[:, :3]
        f = self.tets[:, faces].reshape(-1, faces.shape[1])
        key = np.sort(f[:, :3], axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        return np.unique(f[cnt[inv].ravel() == 1])


# gmsh 10-node tet: 4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(2,3) 9=(1,3)
# CalculiX C3D10:   4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(1,3) 9=(2,3)
GMSH_TO_CCX_TET10 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])


def _bbox(dim: int, tag: int) -> np.ndarray:
    return np.array(gmsh.model.getBoundingBox(dim, tag))


def _close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


def build_geometry(p: BracketParams) -> int:
    """Build the solid in the current gmsh model and return its volume tag."""
    occ = gmsh.model.occ
    p.check()

    base = occ.addBox(0, -p.W / 2, 0, p.L_b, p.W, p.t_b)
    upright = occ.addBox(0, -p.W / 2, 0, p.t_w, p.W, p.H)
    fil_box = occ.addBox(p.t_w, -p.W / 2, p.t_b, p.r_f, p.W, p.r_f)
    fil_cyl = occ.addCylinder(p.t_w + p.r_f, -p.W / 2 - 1, p.t_b + p.r_f, 0, p.W + 2, 0, p.r_f)
    fillet, _ = occ.cut([(3, fil_box)], [(3, fil_cyl)])

    tools = [(3, upright)] + fillet
    if p.has_gussets:
        for yc in (-p.y_gusset, p.y_gusset):
            y0 = yc - p.t_g / 2
            pts = [occ.addPoint(p.t_w, y0, p.t_b),
                   occ.addPoint(p.t_w + p.L_g, y0, p.t_b),
                   occ.addPoint(p.t_w, y0, p.t_b + p.L_g)]
            lines = [occ.addLine(pts[i], pts[(i + 1) % 3]) for i in range(3)]
            surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
            ext = occ.extrude([(2, surf)], 0, p.t_g, 0)
            tools += [e for e in ext if e[0] == 3]
    solid, _ = occ.fuse([(3, base)], tools)
    occ.synchronize()
    vol = solid[0][1]

    if p.has_gussets:
        vol = _fillet_gussets(p, vol)

    cutters = []
    for yb in (-p.y_bolt, p.y_bolt):
        cutters.append((3, occ.addCylinder(p.x_bolt, yb, -1, 0, 0, p.t_b + 2, p.d_bolt / 2)))
    cutters.append((3, occ.addCylinder(-1, 0, p.z_mount, p.t_w + 2, 0, 0, p.d_mount / 2)))
    if p.has_hole:
        cutters.append((3, occ.addCylinder(-1, 0, p.z_hole, p.t_w + 2, 0, 0, p.d_h / 2)))
    solid, _ = occ.cut([(3, vol)], cutters)
    occ.synchronize()
    if len(solid) != 1:
        raise RuntimeError(f"expected one volume, got {solid}")
    return solid[0][1]


def _fillet_gussets(p: BracketParams, vol: int) -> int:
    """Round the re-entrant edges where the gussets meet the plates (avoids stress singularities)."""
    occ = gmsh.model.occ
    surfs = gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False)
    curves = gmsh.model.getBoundary(surfs, combined=False, oriented=False)
    curves = sorted({abs(c[1]) for c in curves if c[0] == 1})
    tol = 1e-3
    picked = []
    for yc in (-p.y_gusset, p.y_gusset):
        y_planes = (yc - p.t_g / 2, yc + p.t_g / 2)
        for c in curves:
            x0, y0, z0, x1, y1, z1 = _bbox(1, c)
            in_side_plane = any(_close(y0, yp, tol) and _close(y1, yp, tol) for yp in y_planes)
            on_upright = _close(x0, p.t_w, tol) and _close(x1, p.t_w, tol)
            on_base = _close(z0, p.t_b, tol) and _close(z1, p.t_b, tol)
            on_fillet = (x0 > p.t_w - tol and x1 < p.t_w + p.r_f + tol and
                         z0 > p.t_b - tol and z1 < p.t_b + p.r_f + tol)
            if in_side_plane and (on_upright or on_base or on_fillet):
                picked.append(c)
                continue
            spans_gusset = _close(y0, y_planes[0], tol) and _close(y1, y_planes[1], tol)
            tip_up = on_upright and _close(z0, p.t_b + p.L_g, tol) and _close(z1, p.t_b + p.L_g, tol)
            tip_base = on_base and _close(x0, p.t_w + p.L_g, tol) and _close(x1, p.t_w + p.L_g, tol)
            if spans_gusset and (tip_up or tip_base):
                picked.append(c)
    if len(picked) < 6:
        raise RuntimeError(f"found only {len(picked)} gusset edges to fillet")
    out = occ.fillet([vol], picked, [p.r_g], removeVolume=True)
    occ.synchronize()
    vols = [t for d, t in out if d == 3]
    if len(vols) != 1:
        raise RuntimeError("gusset fillet failed")
    return vols[0]


def _classify_surfaces(p: BracketParams, vol: int) -> dict:
    """Find the bolt hole, mounting hole and hot-spot surfaces by their bounding boxes."""
    surfs = [s[1] for s in gmsh.model.getBoundary([(3, vol)], combined=False, oriented=False)]
    tol = 1e-3
    out = {"bolt": [], "mount": [], "lhole": [], "refine": []}
    for s in surfs:
        x0, y0, z0, x1, y1, z1 = _bbox(2, s)
        cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        # bolt holes: cylinders along z through the base
        if (_close(dx, p.d_bolt, 0.05) and _close(dy, p.d_bolt, 0.05) and
                _close(cx, p.x_bolt, 0.05) and abs(abs(cy) - p.y_bolt) < 0.05):
            out["bolt"].append(s)
        # mounting hole: cylinder along x through the upright
        elif (_close(dy, p.d_mount, 0.05) and _close(dz, p.d_mount, 0.05) and
              _close(cz, p.z_mount, 0.05) and _close(cy, 0, 0.05)):
            out["mount"].append(s)
        elif (p.has_hole and _close(dy, p.d_h, 0.05) and _close(dz, p.d_h, 0.05) and
              _close(cz, p.z_hole, 0.05)):
            out["lhole"].append(s)
        # inner fillet (cylindrical, between x = t_w and t_w + r_f)
        elif (x0 > p.t_w - tol and x1 < p.t_w + p.r_f + tol and
              z0 > p.t_b - tol and z1 < p.t_b + p.r_f + tol and dx > tol and dz > tol):
            out["refine"].append(s)
        # gusset edge fillets (small surfaces near the gusset faces)
        elif p.has_gussets and min(dx, dy, dz) < p.r_g + tol and any(
                abs(cy - s_ * p.y_gusset) < p.t_g / 2 + p.r_g + tol for s_ in (-1, 1)):
            out["refine"].append(s)
    if len(out["bolt"]) < 2 or len(out["mount"]) < 1:
        raise RuntimeError(f"surface classification failed: {out}")
    return out


def mesh_bracket(p: BracketParams, ms: MeshSettings | None = None, msh_path: str | None = None,
                 verbose: bool = False) -> BracketMesh:
    """Build, mesh and return the bracket with node sets FIX (bolt holes) and MOUNT (mounting hole)."""
    ms = ms or MeshSettings()
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.clear()
    gmsh.model.add("bracket")
    try:
        vol = build_geometry(p)
        volume = gmsh.model.occ.getMass(3, vol)
        groups = _classify_surfaces(p, vol)

        fields = gmsh.model.mesh.field
        f_d1 = fields.add("Distance")
        fields.setNumbers(f_d1, "SurfacesList", groups["refine"])
        fields.setNumber(f_d1, "Sampling", 40)
        f_t1 = fields.add("Threshold")
        fields.setNumber(f_t1, "InField", f_d1)
        fields.setNumber(f_t1, "SizeMin", ms.h_min)
        fields.setNumber(f_t1, "SizeMax", ms.h_max)
        fields.setNumber(f_t1, "DistMin", ms.dist_min)
        fields.setNumber(f_t1, "DistMax", ms.dist_max)
        f_d2 = fields.add("Distance")
        fields.setNumbers(f_d2, "SurfacesList", groups["bolt"] + groups["mount"] + groups["lhole"])
        fields.setNumber(f_d2, "Sampling", 40)
        f_t2 = fields.add("Threshold")
        fields.setNumber(f_t2, "InField", f_d2)
        fields.setNumber(f_t2, "SizeMin", ms.h_hole)
        fields.setNumber(f_t2, "SizeMax", ms.h_max)
        fields.setNumber(f_t2, "DistMin", ms.dist_min)
        fields.setNumber(f_t2, "DistMax", ms.dist_max)
        f_min = fields.add("Min")
        fields.setNumbers(f_min, "FieldsList", [f_t1, f_t2])
        fields.setAsBackgroundMesh(f_min)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", ms.order)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)   # untangle curved quadratic elements
        gmsh.model.mesh.generate(3)
        _, etags, _ = gmsh.model.mesh.getElements(3)
        min_sj = gmsh.model.mesh.getElementQualities(np.concatenate(etags), "minSJ")
        if (min_sj <= 0).any():
            raise RuntimeError(f"{int((min_sj <= 0).sum())} invalid curved elements after optimisation")
        if msh_path:
            gmsh.write(msh_path)
        mesh = _extract(groups, vol, ms.order)
        mesh.volume = volume
        mesh.params = p
        return mesh
    finally:
        if own:
            gmsh.finalize()


def extract_tet10(vol: int | None = None, nper_want: int = 10):
    """Nodes and 10-node (or 4-node) tets of the current gmsh model, renumbered 0..N-1, CalculiX order.

    Returns (nodes, tets, lookup) where lookup maps gmsh node tags to new indices (-1 = unused).
    """
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)
    etypes, _, enodes = gmsh.model.mesh.getElements(3, -1 if vol is None else vol)
    blocks = []
    for et, en in zip(etypes, enodes):
        nper = gmsh.model.mesh.getElementProperties(et)[3]
        if nper == nper_want:
            blocks.append(en.reshape(-1, nper_want).astype(np.int64))
    if not blocks:
        raise RuntimeError(f"no {nper_want}-node tetrahedra in mesh")
    tet10 = np.vstack(blocks)
    used = np.unique(tet10)
    tmax = int(tags.max()) + 1
    lookup = -np.ones(tmax, dtype=np.int64)
    tag_to_row = -np.ones(tmax, dtype=np.int64)
    tag_to_row[tags.astype(np.int64)] = np.arange(len(tags))
    lookup[used] = np.arange(len(used))
    nodes = coords[tag_to_row[used]]
    tets = lookup[tet10][:, GMSH_TO_CCX_TET10] if nper_want == 10 else lookup[tet10]

    # make sure every element has positive volume
    a, b, c, d = (nodes[tets[:, i]] for i in range(4))
    signed = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    flip = signed < 0
    if flip.any():
        # swap corners 1 and 2 and the matching mid-side nodes
        perm = np.array([0, 2, 1, 3, 6, 5, 4, 7, 9, 8])[:tets.shape[1]]
        tets[flip] = tets[flip][:, perm]
    return nodes, tets, lookup


def _extract(groups: dict, vol: int, order: int = 2) -> BracketMesh:
    nodes, tets, lookup = extract_tet10(vol, 10 if order == 2 else 4)
    sets = {}
    for name, key in (("FIX", "bolt"), ("MOUNT", "mount")):
        idx = []
        for s in groups[key]:
            ntags, _, _ = gmsh.model.mesh.getNodes(2, s, includeBoundary=True)
            idx.append(lookup[ntags.astype(np.int64)])
        idx = np.unique(np.concatenate(idx))
        sets[name] = idx[idx >= 0]
    return BracketMesh(nodes=nodes, tets=tets, sets=sets)
