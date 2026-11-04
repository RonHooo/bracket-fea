"""3D field plots of bracket meshes with pyvista (off-screen)."""

from __future__ import annotations

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

# VTK_QUADRATIC_TETRA uses the same node order as CalculiX C3D10
_VTK_QTET = 24


def to_grid(nodes: np.ndarray, tets: np.ndarray, point_data: dict | None = None) -> pv.UnstructuredGrid:
    cells = np.hstack([np.full((len(tets), 1), 10), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), _VTK_QTET, dtype=np.uint8), nodes)
    for k, v in (point_data or {}).items():
        grid.point_data[k] = v
    return grid


def set_camera(pl: pv.Plotter, azimuth: float = 215, elevation: float = 20, zoom: float = 1.0) -> None:
    pl.view_isometric()
    pl.camera.azimuth = azimuth
    pl.camera.elevation = elevation
    pl.camera.zoom(zoom)


def plot_field(nodes, tets, values, path, title="", label="", cmap="turbo", clim=None,
               warp=None, warp_scale=0.0, azimuth=215, elevation=20, show_edges=False,
               window=(1100, 850), extra=None, fmt="%.0f"):
    grid = to_grid(nodes, tets, {label or "f": values})
    if warp is not None and warp_scale:
        grid.point_data["U"] = warp
        grid = grid.warp_by_vector("U", factor=warp_scale)
    surf = grid.extract_surface(algorithm="dataset_surface")
    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("white")
    pl.add_mesh(surf, scalars=label or "f", cmap=cmap, clim=clim, show_edges=show_edges,
                edge_color="gray", line_width=0.2,
                scalar_bar_args=dict(title=label, color="black", vertical=True,
                                     position_x=0.86, position_y=0.15, height=0.7, width=0.05,
                                     title_font_size=16, label_font_size=14, fmt=fmt))
    if extra:
        extra(pl)
    if title:
        pl.add_text(title, font_size=12, color="black")
    set_camera(pl, azimuth, elevation)
    pl.screenshot(path)
    pl.close()


def plot_mesh(nodes, tets, path, title="", azimuth=215, elevation=20, window=(1100, 850)):
    surf = to_grid(nodes, tets).extract_surface(algorithm="dataset_surface")
    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("white")
    pl.add_mesh(surf, color="lightsteelblue", show_edges=True, edge_color="dimgray", line_width=0.3)
    if title:
        pl.add_text(title, font_size=12, color="black")
    set_camera(pl, azimuth, elevation)
    pl.screenshot(path)
    pl.close()
