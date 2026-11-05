"""One-page portfolio summary of the bracket project (PDF + PNG).

    .venv/bin/python cad/make_portfolio_page.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "results/figures"
INK, MUTED = "#0b0b0b", "#52514e"


def panel(fig, rect, img, caption):
    ax = fig.add_axes(rect)
    im = mpimg.imread(str(img))
    ax.imshow(im)
    ax.set_axis_off()
    fig.text(rect[0], rect[1] - 0.012, caption, fontsize=8.2, color=MUTED, va="top")
    return ax


def crop_white(path, out, pad=10):
    """Trim the white border of a render so the panels use their space."""
    import numpy as np
    im = mpimg.imread(str(path))
    rgb = im[..., :3]
    mask = (rgb < 0.97).any(axis=2)
    ys, xs = np.where(mask)
    im = im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    plt.imsave(str(out), im)
    return out


def main():
    rt = json.loads((HERE / "results/fea_roundtrip.json").read_text())["step_fine_mesh"]
    tol = json.loads((HERE / "results/tolerance_study.json").read_text())["worst_case"]
    jc = json.loads((HERE / "results/joint_check.json").read_text())
    tmp = HERE / "out/portfolio_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    asm = crop_white(FIG / "assembly_render.png", tmp / "asm.png")
    exp = crop_white(FIG / "assembly_exploded.png", tmp / "exp.png")
    rms = crop_white(FIG / "step_rms_von_mises_y.png", tmp / "rms.png")

    fig = plt.figure(figsize=(16.5, 10.6))
    fig.text(0.03, 0.965, "Engine component bracket: from design study to released part", fontsize=19,
             color=INK, weight="bold", va="top")
    fig.text(0.03, 0.927, "Parametric CAD (FreeCAD, Python API)  ·  CalculiX FEA: static, modal, random vibration, "
             "fatigue  ·  ASME Y14.5 GD&T drawings  ·  tolerance and bolted-joint checks", fontsize=11.5,
             color=MUTED, va="top")

    panel(fig, [0.02, 0.50, 0.30, 0.39], asm, "Assembly: mounting plate, bracket BRK-001, 2 kg component, ISO fasteners")
    panel(fig, [0.33, 0.50, 0.40, 0.39], FIG / "drawing_BRK-001.png",
          "Part drawing (TechDraw, ANSI C): datums A/B/C, position at MMC, basic dimensions")
    panel(fig, [0.75, 0.50, 0.23, 0.39], exp, "Exploded view with item numbers")
    panel(fig, [0.02, 0.07, 0.27, 0.36], rms, "RMS stress, lateral random vibration (mesh from the STEP file)")
    panel(fig, [0.30, 0.07, 0.42, 0.36], FIG / "tolerance_tornado.png",
          "Tolerance sensitivity from FEA finite differences")

    j1, j2 = jc["joint1_M10_component"]["worst"], jc["joint2_M8_base"]["worst"]
    lines = [("Result", ""),
             ("Mass", "0.884 kg (-5 % vs. plain bracket)"),
             ("First natural frequency", f"{rt['f1']:.1f} Hz   (required ≥ 100)"),
             ("Peak static stress", f"{rt['vm_max']:.1f} MPa   (allowed ≤ 140)"),
             ("Fatigue life", f"{rt['life_h']:,.0f} h   (required ≥ 1000)"),
             ("Worst-case tolerances", f"f1 {tol['f1']['fea_at_corner']:.1f} Hz, life {10 ** tol['log10_life']['fea_at_corner']:,.0f} h"),
             ("CAD vs FE geometry", "identical volume (2e-15)"),
             ("Bolted joints, min SF", f"separation {min(j1['sf_separation'], j2['sf_separation']):.0f}, "
                                        f"slip {min(j1['sf_slip'], j2['sf_slip']):.1f}"),
             ("", ""),
             ("Loads and requirements are project", ""),
             ("assumptions, not a real specification.", "")]
    y = 0.42
    for i, (k, v) in enumerate(lines):
        bold = "bold" if i == 0 else "normal"
        fig.text(0.745, y, k, fontsize=10, color=INK if i < 9 else MUTED, weight=bold, va="top")
        fig.text(0.858, y, v, fontsize=10, color=INK, va="top")
        y -= 0.033
    fig.text(0.03, 0.015, "H. Ho · github.com/RonHooo/bracket-fea", fontsize=9, color=MUTED)
    fig.savefig(HERE / "results/portfolio_bracket.pdf")
    fig.savefig(HERE / "results/portfolio_bracket.png", dpi=110)
    print("wrote portfolio_bracket.pdf/png")


if __name__ == "__main__":
    main()
