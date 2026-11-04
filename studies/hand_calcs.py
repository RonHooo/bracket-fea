"""Hand-calculation check of the baseline bracket (static design load).

The base plate between the upright and the bolts acts as a cantilever. At a section x = const away
from the bolts and from the fillet, beam theory gives the bending stress on the top and bottom faces:
    sigma_xx = -/+ M_y / Z,  Z = W t_b^2 / 6
where M_y is the moment about the y axis of all loads on the free side (upright side) of the section.
FE: sigma_xx averaged over the central part of the width (|y| < 25 mm) on the bottom and top faces.
"""

import json
from pathlib import Path

import numpy as np

from bracketfea.analysis import run_design
from bracketfea.ccx import G
from bracketfea.geometry import BracketParams
from bracketfea.problem import LoadCase, STEEL


def main():
    p, lc = BracketParams(), LoadCase()
    r = run_design(p, lc=lc, threads=2, do_random=False)
    nodes, s = r.mesh.nodes, r.s_static
    a = np.array(lc.load_g) * G                       # mm/s^2, same acceleration on everything
    m_c = lc.comp_mass_kg * 1e-3                       # tonne
    cg = np.array(lc.cg(p.z_mount))
    out = {}
    for xs in (40.0, 50.0):
        zc = p.t_b / 2
        # free side of the section: component + upright + fillet + the part of the base plate x < xs
        # bracket self-weight: integrate the volume to the left of the section with the mesh
        tets = r.mesh.tets[:, :4]
        P = nodes[tets]
        vol = np.abs(np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])) / 6
        cen = P.mean(axis=1)
        left = cen[:, 0] < xs
        m_b = STEEL.rho * vol[left].sum()
        cg_b = (cen[left] * vol[left, None]).sum(0) / vol[left].sum()
        M_y = 0.0
        for m, c in ((m_c, cg), (m_b, cg_b)):
            F = m * a
            rvec = c - np.array([xs, 0.0, zc])
            M_y += rvec[2] * F[0] - rvec[0] * F[2]
        Z = p.W * p.t_b ** 2 / 6
        sigma_hand = M_y / Z                            # + on top face (z = t_b) for M_y > 0 (sign check)
        band = (np.abs(nodes[:, 0] - xs) < 1.0) & (np.abs(nodes[:, 1]) < 25)
        top = band & np.isclose(nodes[:, 2], p.t_b)
        bot = band & np.isclose(nodes[:, 2], 0.0)
        fe_top, fe_bot = float(s[top, 0].mean()), float(s[bot, 0].mean())
        out[f"x={xs:.0f}"] = dict(M_y_Nmm=M_y, sigma_beam_MPa=abs(sigma_hand), fe_top_MPa=fe_top,
                                  fe_bottom_MPa=fe_bot, fe_mean_abs=0.5 * (abs(fe_top) + abs(fe_bot)),
                                  diff_pct=100 * (0.5 * (abs(fe_top) + abs(fe_bot)) - abs(sigma_hand)) / abs(sigma_hand))
    Path("results/verification").mkdir(parents=True, exist_ok=True)
    Path("results/verification/hand_calcs.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
