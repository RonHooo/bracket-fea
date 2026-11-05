"""Hand checks of the two bolted joints in the assembly (the FE model treats them as ideal supports).

Joint 1: M10 screw, component -> back face of the upright (one screw, clamped face 70 x 52 mm).
Joint 2: 2 x M8 screws, bracket base -> mounting plate (clamped face 100 x 70 mm).

Loads: the static design load factors (5, 5, 10 g) applied to the component and the bracket at their
centres of gravity, every sign combination (vibration reverses the direction).
Method (Shigley, ch. 8): preload from the torque, T = K F d with K = 0.2 and +/-25 % torque-wrench scatter;
rigid clamped plate tipping about each edge of the contact face, bolt forces proportional to their
distance from that edge; slip resisted by friction from the remaining clamp force. All of this is
conservative: the external load is not shared with the clamped parts (joint stiffness factor C = 1).

    .venv/bin/python cad/joint_check.py
"""

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G = 9.81

BOLTS = {  # tensile stress area (mm^2), proof strength class 8.8 (ISO 898-1, d <= 16 mm), torque (N m)
    "M8": dict(d=8.0, At=36.6, Sp=580.0, torque=25.0),
    "M10": dict(d=10.0, At=58.0, Sp=580.0, torque=49.0),
}
K_NUT = 0.2          # nut factor, as-received steel bolts
SCATTER = 0.25       # torque-wrench preload scatter (+/-)


def preload(b):
    Fi = b["torque"] * 1e3 / (K_NUT * b["d"])      # N (torque in N mm)
    return dict(F_nom=Fi, F_min=(1 - SCATTER) * Fi, F_max=(1 + SCATTER) * Fi,
                stress_max_MPa=(1 + SCATTER) * Fi / b["At"], proof_MPa=b["Sp"])


def load_cases(factors=(5.0, 5.0, 10.0)):
    for s in itertools.product((-1, 1), repeat=3):
        yield np.array(factors) * np.array(s)


def tipping_bolt_forces(bolts, edges, forces, normal):
    """Largest bolt tension from tipping about each contact edge.

    bolts:  (n, 3) bolt positions on the contact plane
    edges:  list of (point, unit direction) of the contact-face edges
    forces: list of (point, force vector) acting on the clamped part
    normal: direction in which the clamped part lifts off the contact plane
    """
    worst = 0.0
    normal = np.asarray(normal, float)
    for p0, a in edges:
        p0, a = np.asarray(p0, float), np.asarray(a, float)
        for sgn in (-1, 1):
            ax = sgn * a
            # lever arm of each bolt = its "lift" for a unit rotation about the edge
            lift = np.array([np.dot(np.cross(ax, b - p0), normal) for b in bolts])
            if (lift <= 0).all():
                continue
            lift = np.clip(lift, 0, None)
            M = sum(np.dot(np.cross(np.asarray(r) - p0, f), ax) for r, f in forces)
            if M <= 0:
                continue
            T = M * lift / np.sum(lift ** 2)
            worst = max(worst, float(T.max()))
    return worst


def joint_component(m_comp, cg):
    """Joint 1: contact plane x = 0, outward normal of the clamped part (component) is -x."""
    b = BOLTS["M10"]
    pl = preload(b)
    bolt = np.array([[0.0, 0.0, 78.0]])
    y0, y1, z0, z1 = -35.0, 35.0, 48.0, 100.0   # overlap of component face and upright
    edges = [((0, 0, z1), (0, 1, 0)), ((0, 0, z0), (0, 1, 0)), ((0, y0, 0), (0, 0, 1)), ((0, y1, 0), (0, 0, 1))]
    rows = []
    for a in load_cases():
        F = m_comp * G * a
        T_tip = tipping_bolt_forces(bolt, edges, [(cg, F)], normal=(-1, 0, 0))   # lifts off towards -x
        T_axial = max(0.0, -F[0])               # pulls the component away from the face (-x)
        T = T_tip + T_axial
        V = float(np.hypot(F[1], F[2]))
        Mx = float(np.cross(np.asarray(cg) - bolt[0], F)[0])  # torque about the screw axis
        friction = 0.15 * (pl["F_min"] - T)
        rows.append(dict(a_g=a.tolist(), bolt_load_N=T, shear_N=V, torque_about_axis_Nmm=Mx,
                         sf_separation=pl["F_min"] / T if T > 0 else np.inf, sf_slip=friction / V))
    worst = dict(sf_separation=min(r["sf_separation"] for r in rows), sf_slip=min(r["sf_slip"] for r in rows),
                 max_bolt_load_N=max(r["bolt_load_N"] for r in rows),
                 max_torque_about_axis_Nmm=max(abs(r["torque_about_axis_Nmm"]) for r in rows))
    # thread engagement in the aluminium housing: internal thread shear area ~ 0.5 pi d L_e (rule of thumb)
    Le, tau_al = 16.0, 207.0                     # mm engaged; 6061-T6 shear strength (MPa), assumed housing alloy
    strip = 0.5 * np.pi * b["d"] * Le * tau_al
    worst.update(thread_strip_N=strip, bolt_proof_load_N=b["Sp"] * b["At"],
                 strip_over_proof=strip / (b["Sp"] * b["At"]), engagement_over_d=Le / b["d"])
    return dict(preload=pl, cases=rows, worst=worst, friction_coefficient=0.15)


def joint_base(m_comp, cg_comp, m_br, cg_br):
    """Joint 2: contact plane z = 0, the bracket lifts off towards +z."""
    b = BOLTS["M8"]
    pl = preload(b)
    bolts = np.array([[80.0, -20.0, 0.0], [80.0, 20.0, 0.0]])
    edges = [((0, 0, 0), (0, 1, 0)), ((100, 0, 0), (0, 1, 0)), ((0, -35, 0), (1, 0, 0)), ((0, 35, 0), (1, 0, 0))]
    rows = []
    for a in load_cases():
        Fc, Fb = m_comp * G * a, m_br * G * a
        forces = [(cg_comp, Fc), (cg_br, Fb)]
        T_tip = tipping_bolt_forces(bolts, edges, forces, normal=(0, 0, 1))
        T_lift = max(0.0, (Fc + Fb)[2]) / 2      # uniform lift-off shared by the two screws
        T = max(T_tip, T_lift)
        Ftot = Fc + Fb
        c = bolts.mean(axis=0)
        Mz = sum(np.cross(np.asarray(r) - c, f)[2] for r, f in forces)
        r_b = 20.0
        V = float(np.hypot(Ftot[0], Ftot[1]) / 2 + abs(Mz) / (2 * r_b))
        friction = 0.2 * (pl["F_min"] - T)
        rows.append(dict(a_g=a.tolist(), bolt_load_N=T, shear_per_bolt_N=V,
                         sf_separation=pl["F_min"] / T if T > 0 else np.inf, sf_slip=friction / V))
    worst = dict(sf_separation=min(r["sf_separation"] for r in rows), sf_slip=min(r["sf_slip"] for r in rows),
                 max_bolt_load_N=max(r["bolt_load_N"] for r in rows))
    return dict(preload=pl, cases=rows, worst=worst, friction_coefficient=0.2)


def main():
    info = json.loads((HERE / "out/assembly/assembly_info.json").read_text())
    m_br, cg_br = info["bracket_mass_kg"], info["bracket_cg_mm"]
    res = {"joint1_M10_component": joint_component(2.0, (-30.0, 0.0, 78.0)),
           "joint2_M8_base": joint_base(2.0, (-30.0, 0.0, 78.0), m_br, cg_br)}
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "joint_check.json").write_text(json.dumps(res, indent=2, default=float))
    for k, r in res.items():
        p, w = r["preload"], r["worst"]
        print(f"{k}: preload {p['F_nom'] / 1e3:.1f} kN (min {p['F_min'] / 1e3:.1f}), bolt stress at max preload "
              f"{p['stress_max_MPa']:.0f} MPa vs proof {p['proof_MPa']:.0f}; worst bolt load {w['max_bolt_load_N']:.0f} N; "
              f"SF separation {w['sf_separation']:.0f}, SF slip {w['sf_slip']:.1f}")
    w = res["joint1_M10_component"]["worst"]
    print(f"thread stripping {w['thread_strip_N'] / 1e3:.0f} kN vs bolt proof load {w['bolt_proof_load_N'] / 1e3:.1f} kN; "
          f"torque about the M10 axis {w['max_torque_about_axis_Nmm']:.1f} N mm")


if __name__ == "__main__":
    main()
