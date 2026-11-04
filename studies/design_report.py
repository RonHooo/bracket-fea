"""Detailed analysis and figures for one design (static, modes, random vibration, fatigue).

    python studies/design_report.py --name baseline
    python studies/design_report.py --name optimum --params '{"t_b": 9.1, ...}'
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from bracketfea.analysis import run_design, design_mesh_settings
from bracketfea.ccx import von_mises
from bracketfea.geometry import BracketParams, MeshSettings
from bracketfea.problem import LoadCase, Requirements, SN, S_Y
from bracketfea import viz


def component_actor(lc: LoadCase, p: BracketParams):
    def add(pl):
        cg = np.array(lc.cg(p.z_mount))
        pl.add_mesh(pv.Sphere(radius=6, center=cg), color="dimgray")
        pl.add_mesh(pv.Line(cg, (p.t_w / 2, 0, p.z_mount)), color="dimgray", line_width=4)
    return add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--params", default="{}", help="JSON dict of BracketParams fields")
    ap.add_argument("--scale", type=float, default=0.7, help="mesh scale (1 = design-study mesh)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/designs")
    args = ap.parse_args()

    p = BracketParams(**json.loads(args.params))
    lc, req = LoadCase(), Requirements()
    base = design_mesh_settings(p)
    ms = MeshSettings(h_max=base.h_max * args.scale, h_min=base.h_min * args.scale,
                      h_hole=base.h_hole * args.scale, dist_min=base.dist_min, dist_max=base.dist_max)
    r = run_design(p, lc=lc, ms=ms, threads=args.threads)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    nodes, tets = r.mesh.nodes, r.mesh.tets
    summ = r.summary(req)
    summ.update(scale=args.scale, freq=r.freq.tolist(),
                eff_mass_frac=(r.eff_mass[:, :3] / r.total_mass_t).tolist(),
                lives={a: {k: r.random[a][k] for k in ("life_dirlik_h", "life_nb_h", "life_steinberg_h")}
                       for a in r.random},
                hot_nodes={a: nodes[r.random[a]["hot_node"]].tolist() for a in r.random},
                params=p.to_dict())
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))

    comp = component_actor(lc, p)
    viz.plot_mesh(nodes, tets, str(out / "mesh.png"), title=f"{args.name}: {r.mesh.n_nodes} nodes (C3D10)")

    vm = r.vm_static.copy()
    vm_plot = np.where(r.region, vm, np.nan)
    umax = np.abs(r.u_static).max()
    viz.plot_field(nodes, tets, vm_plot, str(out / "static_von_mises.png"),
                   title=f"Static design load: von Mises (MPa), max {r.vm_max:.0f} MPa\n"
                         f"grey = excluded near supports, deformation x{5 / umax:.0f}",
                   label="von Mises (MPa)", warp=r.u_static, warp_scale=5 / umax, extra=comp)

    for k in range(2):
        u = r.modal_u[k]
        mag = np.linalg.norm(u, axis=1)
        viz.plot_field(nodes, tets, mag / mag.max(), str(out / f"mode{k + 1}.png"),
                       title=f"Mode {k + 1}: {r.freq[k]:.0f} Hz", label="|u| (normalised)",
                       warp=u, warp_scale=8 / np.abs(u).max(), cmap="viridis", extra=comp, fmt="%.1f")

    worst = min(r.random, key=lambda a: r.random[a]["life_dirlik_h"])
    rr = r.random[worst]
    rms = np.full(r.mesh.n_nodes, np.nan)
    rms[rr["nodes_idx"]] = rr["rms_map"]
    viz.plot_field(nodes, tets, rms, str(out / f"rms_von_mises_{worst}.png"),
                   title=f"Random vibration, base excitation in {worst}: RMS von Mises (MPa)\n"
                         f"hot spot {rr['vm_rms_max']:.0f} MPa RMS, life {rr['life_dirlik_h']:.3g} h (Dirlik)",
                   label="RMS stress (MPa)", extra=comp)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    f = rr["f"]
    ax[0].loglog(f, lc.psd(f), "k-")
    ax[0].set_title(f"Input PSD ({lc.psd.grms():.1f} g RMS)")
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("acceleration PSD (g^2/Hz)")
    ax[1].loglog(f, rr["psd_hot"], "r-")
    for fi in r.freq[r.freq < f.max()]:
        ax[1].axvline(fi, color="gray", ls=":", lw=1)
    ax[1].set_title(f"Hot-spot stress PSD, excitation {worst} (dotted: natural frequencies)")
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("equivalent von Mises PSD (MPa^2/Hz)")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "psd.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: summ[k] for k in ("mass_kg", "vm_max", "f1", "f2", "life_h", "feasible",
                                             "n_nodes")}, default=float))


if __name__ == "__main__":
    main()
