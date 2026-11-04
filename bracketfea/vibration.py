"""Base-excitation response by modal superposition (harmonic and random).

Mode shapes from CalculiX are mass-normalised, so for a base acceleration a_b(w) along unit vector r
the modal coordinates are  q_i = -Gamma_i a_b / (w_i^2 - w^2 + 2 i zeta w_i w),  Gamma_i = phi_i^T M r,
and any linear response (displacement, stress component) is  y = sum_i y_i q_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ccx import G

# von Mises quadratic form for (xx, yy, zz, xy, yz, zx): s_vm^2 = s^T Q s
Q_VM = np.array([[1.0, -0.5, -0.5, 0, 0, 0],
                 [-0.5, 1.0, -0.5, 0, 0, 0],
                 [-0.5, -0.5, 1.0, 0, 0, 0],
                 [0, 0, 0, 3.0, 0, 0],
                 [0, 0, 0, 0, 3.0, 0],
                 [0, 0, 0, 0, 0, 3.0]])


@dataclass
class PSD:
    """Piecewise log-log acceleration PSD, breakpoints in (Hz, g^2/Hz)."""
    freqs: tuple
    levels: tuple

    def __call__(self, f: np.ndarray) -> np.ndarray:
        lf, ll = np.log(self.freqs), np.log(self.levels)
        out = np.exp(np.interp(np.log(f), lf, ll))
        out[(f < self.freqs[0]) | (f > self.freqs[-1])] = 0.0
        return out

    def grms(self) -> float:
        f = np.geomspace(self.freqs[0], self.freqs[-1], 20000)
        return float(np.sqrt(np.trapezoid(self(f), f)))


def modal_frf(freq_hz: np.ndarray, f_modes: np.ndarray, zeta: float) -> np.ndarray:
    """H_i(w) = 1 / (w_i^2 - w^2 + 2 i zeta w_i w), shape (n_freq, n_modes)."""
    w = 2 * np.pi * freq_hz[:, None]
    wi = 2 * np.pi * f_modes[None, :]
    return 1.0 / (wi ** 2 - w ** 2 + 2j * zeta * wi * w)


def response_per_g(freq_hz, f_modes, gamma, modal_y, zeta):
    """Complex response per 1 g base acceleration.

    gamma    (n_modes,) participation factors for the excitation direction
    modal_y  (n_modes, ...) modal responses (e.g. stresses at nodes, shape (n_modes, n_nodes, 6))
    returns  (n_freq, ...) complex
    """
    H = modal_frf(freq_hz, f_modes, zeta)                 # (nf, nm)
    coef = -(gamma[None, :] * H) * G                       # (nf, nm)
    return np.tensordot(coef, modal_y, axes=([1], [0]))


def static_per_g(f_modes, gamma, modal_y):
    """Quasi-static (w -> 0) response to 1 g, for the mode truncation check."""
    wi = 2 * np.pi * f_modes
    return -np.tensordot(gamma * G / wi ** 2, modal_y, axes=([0], [0]))


def vm_psd(freq_hz, f_modes, gamma, modal_stress, zeta, psd_g2: np.ndarray, chunk: int = 4000):
    """Equivalent von Mises stress PSD (Preumont & Piefort 1994) at every node.

    modal_stress (n_modes, n_nodes, 6); psd_g2 (n_freq,) input PSD in g^2/Hz.
    returns (n_freq, n_nodes) in MPa^2/Hz
    """
    H = modal_frf(freq_hz, f_modes, zeta)
    coef = -(gamma[None, :] * H) * G                      # per (mm/s^2) -> per g
    nn = modal_stress.shape[1]
    out = np.empty((len(freq_hz), nn))
    for k in range(0, nn, chunk):
        s = np.tensordot(coef, modal_stress[:, k:k + chunk, :], axes=([1], [0]))   # (nf, n, 6)
        out[:, k:k + chunk] = np.real(np.einsum("fni,ij,fnj->fn", s.conj(), Q_VM, s))
    return out * psd_g2[:, None]


def frequency_grid(f_lo, f_hi, f_modes, zeta, n_base: int = 400, n_peak: int = 60):
    """Log grid plus dense points around each resonance (half-power bandwidth ~ 2 zeta f_i)."""
    grid = [np.geomspace(f_lo, f_hi, n_base)]
    for fi in f_modes:
        if f_lo < fi < f_hi:
            bw = 2 * zeta * fi
            grid.append(fi + bw * np.linspace(-6, 6, n_peak))
    g = np.unique(np.concatenate(grid))
    return g[(g >= f_lo) & (g <= f_hi)]


def miles_grms(fn: float, Q: float, psd_at_fn: float) -> float:
    """Miles' equation: RMS response acceleration of an SDOF system on a flat PSD (g)."""
    return float(np.sqrt(np.pi / 2 * fn * Q * psd_at_fn))
