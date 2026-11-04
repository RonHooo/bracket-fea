"""Fast unit tests (no CalculiX run needed except the small smoke test at the end)."""

import numpy as np
import pytest

from bracketfea import fatigue as fat
from bracketfea.ccx import von_mises, ccx_path
from bracketfea.geometry import BracketParams, GMSH_TO_CCX_TET10
from bracketfea.vibration import modal_frf, miles_grms, PSD, Q_VM


def test_von_mises_uniaxial_and_shear():
    s = np.array([[100.0, 0, 0, 0, 0, 0], [0, 0, 0, 50.0, 0, 0]])
    assert np.allclose(von_mises(s), [100.0, np.sqrt(3) * 50.0])


def test_von_mises_matrix_form():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(20, 6))
    assert np.allclose(np.einsum("ni,ij,nj->n", s, Q_VM, s), von_mises(s) ** 2)


def test_shigley_sn_curve():
    sn = fat.SNCurve.shigley(380.0, fat.surface_factor_hot_rolled(380.0), 0.9)
    assert sn.cycles(0.9 * 380.0) == pytest.approx(1e3, rel=1e-6)
    assert sn.endurance() == pytest.approx(0.811 * 0.9 * 190, rel=1e-3)


def test_psd_grms_flat():
    psd = PSD(freqs=(20.0, 1000.0), levels=(0.02, 0.02))
    assert psd.grms() == pytest.approx(np.sqrt(0.02 * 980), rel=1e-4)


def test_miles_equation_matches_sdof_integral():
    """Absolute acceleration of an SDOF on a flat PSD vs Miles' equation (light damping)."""
    fn, zeta, level = 100.0, 0.02, 0.01
    f = np.linspace(1, 2000, 400000)
    w, wn = 2 * np.pi * f, 2 * np.pi * fn
    T2 = (wn ** 4 + (2 * zeta * wn * w) ** 2) / ((wn ** 2 - w ** 2) ** 2 + (2 * zeta * wn * w) ** 2)
    grms = np.sqrt(np.trapezoid(T2 * level, f))
    assert grms == pytest.approx(miles_grms(fn, 1 / (2 * zeta), level), rel=0.03)


def test_modal_frf_static_limit():
    H = modal_frf(np.array([1e-6]), np.array([50.0]), 0.02)
    assert H[0, 0].real == pytest.approx(1 / (2 * np.pi * 50.0) ** 2, rel=1e-6)


def test_narrowband_damage_scales_with_rms():
    sn = fat.SNCurve(a=800.0, b=-0.13)
    f = np.linspace(90, 110, 2001)
    S = np.ones_like(f)
    d1 = fat.damage_narrowband(f, S, sn, 3600)
    d2 = fat.damage_narrowband(f, 4 * S, sn, 3600)     # rms x 2
    assert d2 / d1 == pytest.approx(2 ** sn.m, rel=1e-6)


def test_dirlik_close_to_narrowband_for_narrow_spectrum():
    sn = fat.SNCurve(a=800.0, b=-0.13)
    f = np.linspace(95, 105, 4001)
    S = 20.0 * np.exp(-0.5 * ((f - 100) / 0.8) ** 2)
    ratio = fat.damage_dirlik(f, S, sn, 3600) / fat.damage_narrowband(f, S, sn, 3600)
    assert 0.85 < ratio < 1.05


def test_synthesized_signal_has_target_variance():
    f = np.linspace(10, 200, 500)
    S = np.full_like(f, 2.0)
    x = fat.synthesize(f, S, duration=200.0, fs=1000.0, rng=np.random.default_rng(1))
    assert np.var(x) == pytest.approx(np.trapezoid(S, f), rel=0.05)


def test_tet10_permutation_is_a_permutation():
    assert sorted(GMSH_TO_CCX_TET10.tolist()) == list(range(10))


def test_params_check_rejects_gusset_hitting_mount_hole():
    with pytest.raises(ValueError):
        BracketParams(t_b=12, L_g=62).check()


def test_mesh_and_surface_nodes():
    from bracketfea.geometry import mesh_bracket, MeshSettings
    m = mesh_bracket(BracketParams(), MeshSettings(h_max=6, h_min=3, h_hole=3))
    surf = m.surface_nodes()
    assert 0 < len(surf) < m.n_nodes
    assert len(m.sets["FIX"]) > 0 and len(m.sets["MOUNT"]) > 0
    assert m.volume == pytest.approx(80522, rel=0.01)


@pytest.mark.skipif(not ccx_path(), reason="CalculiX not installed")
def test_end_to_end_coarse(tmp_path):
    from bracketfea.analysis import run_design
    from bracketfea.geometry import MeshSettings
    r = run_design(BracketParams(), ms=MeshSettings(h_max=6, h_min=3, h_hole=3), workdir=tmp_path,
                   threads=1)
    assert 50 < r.freq[0] < 65                      # ~57 Hz on converged meshes
    assert r.total_mass_t * 1e3 == pytest.approx(r.mass_kg + 2.0, rel=0.02)
    assert np.isfinite(r.life_h)
