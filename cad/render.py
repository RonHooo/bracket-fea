"""Renders of the bracket and the assembly (assembled and exploded) from the STL files written by
build_part.py and build_assembly.py.

    .venv/bin/python cad/render.py
"""

from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
STL = HERE / "out/assembly/stl"
FIG = HERE / "results/figures"

COLORS = {"MountingPlate": "#8c8c8c", "Bracket": "#4f7cac", "Component": "#2f2f2f",
          "ScrewM8": "#c9b037", "WasherM8": "#d9d9d9", "ScrewM10": "#c9b037", "WasherM10": "#d9d9d9"}
# exploded offsets (mm) along the assembly directions
EXPLODE = {"MountingPlate": (0, 0, -45), "Bracket": (0, 0, 0), "Component": (-60, 0, 0),
           "ScrewM8_1": (0, 0, 50), "ScrewM8_2": (0, 0, 50), "WasherM8_1": (0, 0, 24), "WasherM8_2": (0, 0, 24),
           "ScrewM10": (150, 0, 0), "WasherM10": (75, 0, 0)}
# label anchor points (assembled coordinates, before the exploded offset) and label offsets
ANCHOR = {"MountingPlate": (125, -60, -6), "Bracket": (3.5, -35, 100), "Component": (-35, -40, 108),
          "ScrewM8_1": (80, -20, 17), "WasherM8_1": (88, -20, 11), "ScrewM10": (15, -8, 78),
          "WasherM10": (8, -10, 78)}
LABELS = {"MountingPlate": "1  mounting plate", "Bracket": "2  bracket BRK-001", "Component": "3  component (2 kg)",
          "ScrewM8_1": "4  M8x20 screw (2)", "WasherM8_1": "5  M8 washer (2)", "ScrewM10": "6  M10x25 screw",
          "WasherM10": "7  M10 washer"}


def color(name):
    return COLORS[name.split("_")[0]]


def plotter(window=(1400, 1000)):
    pl = pv.Plotter(off_screen=True, window_size=window, lighting="three lights")
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    return pl


def camera(pl, direction, zoom=1.0):
    """Look at the scene centre from `direction` (z up), narrow view angle to limit perspective."""
    b = np.array(pl.bounds).reshape(3, 2)
    c = b.mean(axis=1)
    d = np.asarray(direction, float)
    pl.camera.view_angle = 20
    pl.camera_position = [tuple(c + 1000 * d / np.linalg.norm(d)), tuple(c), (0, 0, 1)]
    pl.reset_camera()
    pl.camera.zoom(zoom)


def assembly(path, exploded=False, labels=False):
    pl = plotter()
    for f in sorted(STL.glob("*.stl")):
        name = f.stem
        m = pv.read(f)
        if exploded:
            m = m.translate(EXPLODE[name], inplace=False)
        pl.add_mesh(m, color=color(name), smooth_shading=True, specular=0.3, split_sharp_edges=True)
        if labels and name in LABELS:
            c = np.array(ANCHOR[name]) + (np.array(EXPLODE[name]) if exploded else 0)
            pl.add_point_labels([c], [LABELS[name]], font_size=15, text_color="black", shape_color="white",
                                shape_opacity=0.85, always_visible=True, point_size=6, point_color="black",
                                show_points=True, margin=2)
        if exploded and name.startswith(("Screw", "Washer")):
            c0 = np.array(pv.read(f).center)
            line = pv.Line(c0, c0 + np.array(EXPLODE[name]))
            pl.add_mesh(line, color="#999999", line_width=1)
    camera(pl, (0.45, -1.0, 0.65), 1.15 if exploded else 1.3)
    pl.screenshot(str(path))
    pl.close()


def bracket(path):
    pl = plotter((1200, 1000))
    m = pv.read(HERE / "out/bracket.stl")
    pl.add_mesh(m, color=COLORS["Bracket"], smooth_shading=True, specular=0.4, split_sharp_edges=True)
    edges = m.extract_feature_edges(feature_angle=30, boundary_edges=False, manifold_edges=False)
    pl.add_mesh(edges, color="black", line_width=1.5)
    camera(pl, (0.9, -1.0, 0.8), 0.95)
    pl.screenshot(str(path))
    pl.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    bracket(FIG / "bracket_render.png")
    assembly(FIG / "assembly_render.png")
    assembly(FIG / "assembly_exploded.png", exploded=True, labels=True)
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
