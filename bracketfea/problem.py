"""Load case, material and requirements of the bracket study.

The numbers below are assumptions chosen for a representative engine-mounted bracket; they are not
taken from a specific product specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ccx import Material
from .fatigue import SNCurve, surface_factor_hot_rolled
from .vibration import PSD

# AISI 1020 hot-rolled steel (Shigley, Table A-20): S_ut = 380 MPa, S_y = 210 MPa
S_UT = 380.0
S_Y = 210.0
STEEL = Material(name="STEEL", E=207000.0, nu=0.292, rho=7.85e-9)

KA = surface_factor_hot_rolled(S_UT)   # ~0.81
KB = 0.9                               # size factor, typical for 4-10 mm plate sections
SN = SNCurve.shigley(S_UT, KA, KB, f=0.9)


@dataclass
class LoadCase:
    comp_mass_kg: float = 2.0                      # mounted component
    comp_offset: float = 30.0                      # CG distance from the upright outer face (mm)
    load_g: tuple = (5.0, 5.0, -10.0)              # quasi-static design load factors (x, y, z)
    zeta: float = 0.02                             # modal damping ratio
    psd: PSD = field(default_factory=lambda: PSD(freqs=(20.0, 1000.0), levels=(0.02, 0.02)))
    psd_band: tuple = (20.0, 1000.0)

    def cg(self, z_mount: float) -> tuple:
        return (-self.comp_offset, 0.0, z_mount)


@dataclass
class Requirements:
    safety_factor: float = 1.5     # on yield, static design load
    f1_min: float = 100.0          # Hz, first natural frequency
    life_min_h: float = 1000.0     # h, random vibration fatigue life (worst axis, Dirlik)

    @property
    def stress_allow(self) -> float:
        return S_Y / self.safety_factor
