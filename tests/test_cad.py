"""Tests for the CAD -> FEA helpers in cad/ (no FreeCAD needed)."""

import math
import sys
from pathlib import Path

import gmsh
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cad"))

from bracketfea.geometry import BracketParams, MeshSettings, build_geometry  # noqa: E402
from fea_from_step import classify_step_surfaces, mesh_step  # noqa: E402
from joint_check import K_NUT, preload, tipping_bolt_forces  # noqa: E402


def test_step_round_trip_classification(tmp_path):
    """A STEP file written from the native geometry is found and meshed like the native model."""
    p = BracketParams(t_b=10, t_w=7, r_f=8, d_h=10)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        vol = build_geometry(p)
        v_native = gmsh.model.occ.getMass(3, vol)
        gmsh.write(str(tmp_path / "b.step"))
    finally:
        gmsh.finalize()
    m, counts = mesh_step(tmp_path / "b.step", p, MeshSettings(h_max=6, h_min=3, h_hole=3, order=1))
    assert counts["bolt"] >= 2 and counts["mount"] >= 1 and counts["lhole"] >= 1 and counts["refine"] >= 1
    assert m.volume == pytest.approx(v_native, rel=1e-9)
    assert len(m.sets["FIX"]) > 0 and len(m.sets["MOUNT"]) > 0


def test_tipping_single_bolt():
    """One bolt at distance d from the pivot edge, force F at lever L: T = F L / d."""
    bolts = np.array([[50.0, 0.0, 0.0]])
    edges = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
    F, L = 100.0, 30.0
    # A downward force at x = -L tips the part about x = 0 and lifts the bolt at x = 50.
    T = tipping_bolt_forces(bolts, edges, [((-L, 0.0, 0.0), (0.0, 0.0, -F))], normal=(0, 0, 1))
    assert T == pytest.approx(F * L / 50.0)


def test_tipping_two_bolts_share_by_distance():
    """Bolt forces are proportional to their distance from the pivot edge."""
    bolts = np.array([[20.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
    edges = [((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))]
    M = 1000.0 * 10.0
    T = tipping_bolt_forces(bolts, edges, [((-10.0, 0.0, 0.0), (0.0, 0.0, -1000.0))], normal=(0, 0, 1))
    assert T == pytest.approx(M * 60.0 / (20.0 ** 2 + 60.0 ** 2))


def test_preload_from_torque():
    b = dict(d=10.0, At=58.0, Sp=580.0, torque=49.0)
    pl = preload(b)
    assert pl["F_nom"] == pytest.approx(49.0e3 / (K_NUT * 10.0))
    assert pl["F_min"] < pl["F_nom"] < pl["F_max"]


def test_released_design_is_close_to_optimizer():
    import json
    d = json.loads((ROOT / "cad/released_design.json").read_text())
    for k, v in d["optimizer_values"].items():
        assert abs(d[k] - v) <= 0.61, k
