"""V3: checks of the random-vibration and fatigue post-processing written for this project.

(a) Modal superposition (my code) vs CalculiX *STEADY STATE DYNAMICS, CG displacement per 1 g, x and z.
(b) Mode truncation: quasi-static (w -> 0) modal sum vs a direct static 1 g analysis (stress at hot spot).
(c) Spectral fatigue (narrow-band, Steinberg, Dirlik) vs rainflow counting of synthesized time
    histories, for a narrow-band, a bimodal and the bracket hot-spot stress PSD.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bracketfea import fatigue as fat
from bracketfea.analysis import run_design, design_mesh_settings, design_region
from bracketfea.ccx import (G, write_bracket_deck, run_ccx, read_dat_modes, read_dat_mode_shapes,
                            von_mises)
from bracketfea.geometry import BracketParams, mesh_bracket
from bracketfea.problem import STEEL, SN, LoadCase
from bracketfea.vibration import modal_frf, static_per_g


def read_ssd_cg(dat: Path):
    """CG displacements from the steady-state step (*NODE PRINT): rows of (freq, ux, uy, uz)."""
    lines = dat.read_text().splitlines()
    start = max(i for i, l in enumerate(lines) if "S T E P" in l)   # last step = steady state
    out = []
    for k, l in enumerate(lines):
        if k < start:
            continue
        if l.strip().startswith("displacements") and "set CG" in l:
            t = float(l.split("time")[-1])
            j = k + 1
            while not lines[j].strip():
                j += 1
            v = [float(x) for x in lines[j].split()[1:4]]
            out.append([t] + v)
    return np.array(out)


def check_ssd(p: BracketParams, lc: LoadCase, work: Path, figdir: Path) -> list:
    mesh = mesh_bracket(p, design_mesh_settings(p))
    n = mesh.n_nodes
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax_i, (name, d) in enumerate((("x", (1.0, 0, 0)), ("z", (0, 0, 1.0)))):
        job = f"ssd_{name}"
        ssd = dict(f_lo=20.0, f_hi=1000.0, n=20, damping=lc.zeta, direction=d)
        write_bracket_deck(work / f"{job}.inp", mesh, STEEL, lc.comp_mass_kg * 1e-3, lc.cg(p.z_mount),
                           load_g=(1.0, 0, 0), n_modes=10, ssd=ssd)
        run_ccx(work, job, threads=8)
        md = read_dat_modes(work / f"{job}.dat")
        shapes = read_dat_mode_shapes(work / f"{job}.dat", "CG")
        phi_cg = np.array([shapes[i] for i in range(1, 11)])                 # (modes, 3)
        k = int(np.argmax(np.abs(d)))
        gamma = md["participation"][:10, k]
        ssd_rows = read_ssd_cg(work / f"{job}.dat")
        # CalculiX prints the steady-state amplitude as (real, imaginary) pairs: keep the first
        # occurrence per frequency as real part and the second as imaginary part
        freqs = np.unique(ssd_rows[:, 0])
        amp_ccx = []
        for fq in freqs:
            r = ssd_rows[ssd_rows[:, 0] == fq]
            re = r[0, 1 + k]
            im = r[1, 1 + k] if len(r) > 1 else 0.0
            amp_ccx.append(abs(re + 1j * im))
        amp_ccx = np.array(amp_ccx)
        H = modal_frf(freqs, md["freq"][:10], lc.zeta)
        u_modal = (-(gamma[None, :] * H) * G) @ phi_cg[:, k]
        amp_mine = np.abs(u_modal)
        err = np.abs(amp_mine - amp_ccx) / amp_ccx.max()
        rows.append(dict(check=f"SSD CG displacement, base excitation {name}",
                         max_err_pct_of_peak=100 * err.max(),
                         peak_ccx=amp_ccx.max(), peak_mine=amp_mine.max(),
                         f_peak_ccx=freqs[np.argmax(amp_ccx)], f_peak_mine=freqs[np.argmax(amp_mine)]))
        a = axes[ax_i]
        a.loglog(freqs, amp_ccx, "k-", lw=2, label="CalculiX steady-state dynamics")
        a.loglog(freqs, amp_mine, "r--", lw=1.5, label="modal superposition (this code)")
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel(f"|u_{name}| at component CG per 1 g (mm)")
        a.set_title(f"base excitation in {name}")
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "v3_ssd_vs_modal.png", dpi=150)
    plt.close(fig)
    return rows


def check_truncation(p: BracketParams, lc: LoadCase) -> list:
    rows = []
    for k, ax in enumerate("xyz"):
        load = [0.0, 0.0, 0.0]
        load[k] = 1.0
        lck = LoadCase(load_g=tuple(load))
        r = run_design(p, lc=lck, threads=8, do_random=False)
        # GRAV +ax is equivalent to a base acceleration of -1 g along ax
        s_modal = -static_per_g(r.freq, r.gamma[:, k], r.modal_s)
        vm_m, vm_d = von_mises(s_modal), r.vm_static
        reg = r.region
        j = np.where(reg)[0][np.argmax(vm_d[reg])]
        rows.append(dict(check=f"static 1 g {ax}: 10-mode sum vs direct (hot spot von Mises)",
                         direct=vm_d[j], modal=vm_m[j], err_pct=100 * (vm_m[j] - vm_d[j]) / vm_d[j],
                         eff_mass_fraction=r.eff_mass[:, k].sum() / r.total_mass_t))
    return rows


def check_spectral_fatigue(cases: dict, figdir: Path, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    fig, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 3.8))
    for a, (name, (f, S)) in zip(np.atleast_1d(axes), cases.items()):
        T = fat.HOUR
        d_nb = fat.damage_narrowband(f, S, SN, T)
        d_st = fat.damage_steinberg(f, S, SN, T)
        d_dk = fat.damage_dirlik(f, S, SN, T)
        dur, fs = 400.0, 8 * f.max()
        d_rf = np.array([fat.damage_rainflow(fat.synthesize(f, S, dur, fs, rng), SN, dur, T)
                         for _ in range(8)])
        m = fat.spectral_moments(f, S)
        rows.append(dict(check=f"spectral fatigue: {name}", rms=np.sqrt(m[0]),
                         irregularity=m[2] / np.sqrt(m[0] * m[4]),
                         life_rainflow_h=1 / d_rf.mean(), life_rainflow_cov=d_rf.std() / d_rf.mean(),
                         life_dirlik_h=1 / d_dk, life_narrowband_h=1 / d_nb, life_steinberg_h=1 / d_st,
                         dirlik_vs_rainflow=d_rf.mean() / d_dk))
        a.semilogy(f, S, "k-")
        a.set_title(f"{name}\nrainflow {1 / d_rf.mean():.3g} h, Dirlik {1 / d_dk:.3g} h,\n"
                    f"narrow-band {1 / d_nb:.3g} h", fontsize=9)
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel("stress PSD (MPa^2/Hz)")
        a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "v3_spectral_fatigue.png", dpi=150)
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification")
    args = ap.parse_args()
    out = Path(args.out)
    work = out / "work_v3"
    work.mkdir(parents=True, exist_ok=True)
    lc = LoadCase()
    p = BracketParams()

    rows = check_ssd(p, lc, work, out)
    rows += check_truncation(p, lc)

    f = np.linspace(1, 600, 6000)
    fn, zeta = 120.0, 0.02
    sdof = 1e3 / ((fn ** 2 - f ** 2) ** 2 + (2 * zeta * fn * f) ** 2) * fn ** 2
    bimodal = (0.3 * np.exp(-0.5 * ((f - 60) / 4) ** 2) + 0.15 * np.exp(-0.5 * ((f - 400) / 15) ** 2))
    r = run_design(p, lc=lc, threads=8)
    worst = min(r.random, key=lambda a: r.random[a]["life_dirlik_h"])
    hot = r.random[worst]
    cases = {"narrow-band (SDOF)": (f, sdof * 20 / sdof.max()),
             "bimodal": (f, bimodal * 60),
             f"bracket hot spot ({worst})": (hot["f"], hot["psd_hot"])}
    rows += check_spectral_fatigue(cases, out)
    df = pd.DataFrame(rows)
    df.to_csv(out / "v3_vibration_fatigue.csv", index=False)
    for big in work.glob("*.frd"):
        big.unlink()
    for big in work.glob("*.eig"):
        big.unlink()
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
