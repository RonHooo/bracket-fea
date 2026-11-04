"""End-to-end analysis of one bracket design: mesh -> CalculiX (static + modal) -> random vibration
-> fatigue. Returns scalar results and the fields needed for plots and for the ML dataset."""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fatigue as fat
from .ccx import (read_dat_modes, read_frd, frd_field, run_ccx, von_mises, write_bracket_deck,
                  read_dat_node_print)
from .geometry import BracketParams, MeshSettings, mesh_bracket, BracketMesh
from .problem import STEEL, SN, LoadCase, Requirements
from .vibration import vm_psd, frequency_grid

AXES = ("x", "y", "z")


def design_mesh_settings(p: BracketParams, h_min: float = 1.5, h_max_cap: float = 4.0,
                         t_ratio: float = 0.6, dist_max: float = 8.0) -> MeshSettings:
    """About two quadratic elements through the thinnest plate; refined at fillets and holes."""
    t = min(p.t_b, p.t_w, p.t_g if p.has_gussets else 99)
    h_max = max(min(h_max_cap, t_ratio * t), h_min)
    return MeshSettings(h_max=h_max, h_min=h_min, h_hole=1.5 * h_min, dist_min=1.0, dist_max=dist_max)


def design_region(nodes: np.ndarray, p: BracketParams) -> np.ndarray:
    """Nodes used for stress evaluation: outside 2 d_bolt of the bolt axes (idealised fixed supports)
    and outside 1.5 d_mount of the mounting hole axis (rigid coupling to the component)."""
    x, y, z = nodes.T
    nb = np.min([np.hypot(x - p.x_bolt, y - s * p.y_bolt) for s in (-1, 1)], axis=0)
    nm = np.hypot(y, z - p.z_mount)
    return (nb > 2.0 * p.d_bolt) & (nm > 1.5 * p.d_mount)


@dataclass
class DesignResult:
    params: BracketParams
    mesh: BracketMesh
    mass_kg: float
    region: np.ndarray
    u_static: np.ndarray
    s_static: np.ndarray
    vm_static: np.ndarray
    freq: np.ndarray
    gamma: np.ndarray            # (n_modes, 6) participation factors (x, y, z, rx, ry, rz)
    eff_mass: np.ndarray
    total_mass_t: float
    modal_u: np.ndarray | None = None
    modal_s: np.ndarray | None = None
    random: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    cg_disp: np.ndarray | None = None

    @property
    def vm_max(self) -> float:
        return float(self.vm_static[self.region].max())

    @property
    def hot_spot(self) -> np.ndarray:
        idx = np.where(self.region)[0]
        return self.mesh.nodes[idx[np.argmax(self.vm_static[idx])]]

    @property
    def life_h(self) -> float:
        return min(r["life_dirlik_h"] for r in self.random.values()) if self.random else np.nan

    def summary(self, req: Requirements | None = None) -> dict:
        req = req or Requirements()
        p = self.params
        out = {k: getattr(p, k) for k in ("t_b", "t_w", "r_f", "L_g", "t_g", "d_h")}
        out.update(mass_kg=self.mass_kg, vm_max=self.vm_max, f1=float(self.freq[0]),
                   f2=float(self.freq[1]), n_nodes=self.mesh.n_nodes,
                   hot_x=self.hot_spot[0], hot_y=self.hot_spot[1], hot_z=self.hot_spot[2])
        for ax, r in self.random.items():
            out[f"vm_rms_{ax}"] = r["vm_rms_max"]
            out[f"life_{ax}_h"] = r["life_dirlik_h"]
        out["life_h"] = self.life_h
        out["ok_stress"] = out["vm_max"] <= req.stress_allow
        out["ok_freq"] = out["f1"] >= req.f1_min
        out["ok_life"] = (out["life_h"] >= req.life_min_h) if self.random else np.nan
        out["feasible"] = bool(out["ok_stress"] and out["ok_freq"] and (out["ok_life"] in (True, np.nan)))
        out.update({f"t_{k}": v for k, v in self.timing.items()})
        return out


def random_vibration(res: DesignResult, lc: LoadCase, nodes_idx: np.ndarray | None = None) -> dict:
    """Equivalent von Mises random response and Dirlik fatigue life for base excitation in x, y, z."""
    if nodes_idx is None:
        surf = np.zeros(res.mesh.n_nodes, bool)
        surf[res.mesh.surface_nodes()] = True
        nodes_idx = np.where(res.region & surf)[0]
    idx = nodes_idx
    f_lo, f_hi = lc.psd_band
    fgrid = frequency_grid(f_lo, f_hi, res.freq, lc.zeta)
    psd_in = lc.psd(fgrid)
    out = {}
    for k, ax in enumerate(AXES):
        S = vm_psd(fgrid, res.freq, res.gamma[:, k], res.modal_s[:, idx, :], lc.zeta, psd_in)
        rms = np.sqrt(np.trapezoid(S, fgrid, axis=0))
        j = int(np.argmax(rms))
        Sh = S[:, j]
        T = fat.HOUR
        d_dir = fat.damage_dirlik(fgrid, Sh, SN, T)
        out[ax] = dict(vm_rms_max=float(rms[j]), hot_node=int(idx[j]), rms_map=rms, nodes_idx=idx,
                       f=fgrid, psd_hot=Sh,
                       life_dirlik_h=fat.life_hours(d_dir),
                       life_nb_h=fat.life_hours(fat.damage_narrowband(fgrid, Sh, SN, T)),
                       life_steinberg_h=fat.life_hours(fat.damage_steinberg(fgrid, Sh, SN, T)))
    return out


def run_design(p: BracketParams, lc: LoadCase | None = None, ms: MeshSettings | None = None,
               workdir: str | Path | None = None, n_modes: int = 10, threads: int = 1,
               do_random: bool = True, keep_modal: bool = True, keep_files: bool = False,
               ssd: dict | None = None) -> DesignResult:
    lc = lc or LoadCase()
    ms = ms or design_mesh_settings(p)
    tmp = Path(tempfile.mkdtemp(prefix="bracket_", dir=workdir))
    timing = {}
    try:
        t0 = time.perf_counter()
        mesh = mesh_bracket(p, ms)
        timing["mesh"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        write_bracket_deck(tmp / "job.inp", mesh, STEEL, lc.comp_mass_kg * 1e-3, lc.cg(p.z_mount),
                           lc.load_g, n_modes=n_modes, ssd=ssd)
        run_ccx(tmp, "job", threads=threads)
        timing["solve"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        blocks = read_frd(tmp / "job.frd")
        n = mesh.n_nodes
        disp = [b for b in blocks if b.name == "DISP"]
        strs = [b for b in blocks if b.name == "STRESS"]
        u_st, s_st = frd_field(disp[0], n), frd_field(strs[0], n)
        modal_u = np.stack([frd_field(b, n) for b in disp[1:1 + n_modes]])
        modal_s = np.stack([frd_field(b, n) for b in strs[1:1 + n_modes]])
        md = read_dat_modes(tmp / "job.dat")
        cg = read_dat_node_print(tmp / "job.dat", "CG")
        timing["parse"] = time.perf_counter() - t0

        res = DesignResult(params=p, mesh=mesh, mass_kg=mesh.volume * STEEL.rho * 1e3,
                           region=design_region(mesh.nodes, p), u_static=u_st, s_static=s_st,
                           vm_static=von_mises(s_st), freq=md["freq"][:n_modes],
                           gamma=md["participation"][:n_modes], eff_mass=md["eff_mass"][:n_modes],
                           total_mass_t=float(md["total_eff_mass"][0]),
                           modal_u=modal_u, modal_s=modal_s, timing=timing,
                           cg_disp=cg[0] if cg else None)
        if do_random:
            t0 = time.perf_counter()
            res.random = random_vibration(res, lc)
            timing["random"] = time.perf_counter() - t0
        if not keep_modal:
            res.modal_u = res.modal_s = None
        if keep_files:
            res.workdir = tmp
        return res
    finally:
        if not keep_files:
            shutil.rmtree(tmp, ignore_errors=True)
