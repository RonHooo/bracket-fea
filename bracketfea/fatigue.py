"""S-N curve and spectral (random vibration) fatigue: narrow-band, Steinberg 3-band, Dirlik, rainflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as gamma_fn

HOUR = 3600.0


@dataclass
class SNCurve:
    """Basquin curve S_a = a * N^b (S_a = fully reversed stress amplitude, MPa).

    Built with the Shigley method: S_e = k_a k_b 0.5 S_ut, f S_ut at 1e3 cycles, S_e at 1e6 cycles.
    For random loading the line is extended below S_e (no endurance limit), which is conservative.
    """
    a: float
    b: float

    @classmethod
    def shigley(cls, S_ut: float, ka: float, kb: float, f: float = 0.9) -> "SNCurve":
        S_e = ka * kb * 0.5 * S_ut
        a = (f * S_ut) ** 2 / S_e
        b = -np.log10(f * S_ut / S_e) / 3.0
        return cls(a=a, b=b)

    @property
    def m(self) -> float:
        """Inverse slope: N * S^m = C."""
        return -1.0 / self.b

    @property
    def C(self) -> float:
        return self.a ** self.m

    def cycles(self, S_amp) -> np.ndarray:
        S_amp = np.maximum(np.asarray(S_amp, dtype=float), 1e-12)
        return (S_amp / self.a) ** (1.0 / self.b)

    def endurance(self) -> float:
        return self.a * 1e6 ** self.b


def surface_factor_hot_rolled(S_ut: float) -> float:
    """Shigley surface factor k_a = a S_ut^b for a hot-rolled surface (S_ut in MPa)."""
    return 57.7 * S_ut ** -0.718


def spectral_moments(f: np.ndarray, S: np.ndarray, orders=(0, 1, 2, 4)) -> dict:
    """One-sided PSD S(f) in units^2/Hz on frequency grid f (Hz)."""
    return {n: np.trapezoid(f ** n * S, f) for n in orders}


def damage_narrowband(f, S, sn: SNCurve, T: float) -> float:
    """Rayleigh amplitudes, positive zero-crossing rate (Miles / Bendat)."""
    m = spectral_moments(f, S)
    sigma = np.sqrt(m[0])
    nu0 = np.sqrt(m[2] / m[0])
    return nu0 * T * (np.sqrt(2) * sigma) ** sn.m * gamma_fn(1 + sn.m / 2) / sn.C


def damage_steinberg(f, S, sn: SNCurve, T: float) -> float:
    """Steinberg three-band method: 68.3 % of cycles at 1 sigma, 27.1 % at 2 sigma, 4.33 % at 3 sigma."""
    m = spectral_moments(f, S)
    sigma = np.sqrt(m[0])
    nu0 = np.sqrt(m[2] / m[0])
    frac = np.array([0.6831, 0.2710, 0.0433])
    amps = np.array([1.0, 2.0, 3.0]) * sigma
    return nu0 * T * np.sum(frac / sn.cycles(amps))


def dirlik_pdf(f, S, n_pts: int = 4000):
    """Dirlik (1985) probability density of rainflow stress *ranges*. Returns (ranges, pdf, peak_rate)."""
    m = spectral_moments(f, S)
    m0, m1, m2, m4 = m[0], m[1], m[2], m[4]
    xm = (m1 / m0) * np.sqrt(m2 / m4)
    g = m2 / np.sqrt(m0 * m4)
    D1 = 2 * (xm - g ** 2) / (1 + g ** 2)
    R = (g - xm - D1 ** 2) / (1 - g - D1 + D1 ** 2)
    D2 = (1 - g - D1 + D1 ** 2) / (1 - R)
    D3 = 1 - D1 - D2
    Q = 1.25 * (g - D3 - D2 * R) / D1
    sd = np.sqrt(m0)
    rng = np.linspace(0, 12 * sd, n_pts)
    Z = rng / (2 * sd)
    p = (D1 / Q * np.exp(-Z / Q) + D2 * Z / R ** 2 * np.exp(-Z ** 2 / (2 * R ** 2))
         + D3 * Z * np.exp(-Z ** 2 / 2)) / (2 * sd)
    peak_rate = np.sqrt(m4 / m2)
    return rng, p, peak_rate


def damage_dirlik(f, S, sn: SNCurve, T: float) -> float:
    rng, p, peak_rate = dirlik_pdf(f, S)
    integrand = p / sn.cycles(rng / 2)
    return peak_rate * T * np.trapezoid(integrand, rng)


def synthesize(f, S, duration: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian time history with one-sided PSD S(f) (random phase, inverse FFT)."""
    n = int(duration * fs)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    df = freqs[1] - freqs[0]
    Si = np.interp(freqs, f, S, left=0.0, right=0.0)
    amp = np.sqrt(2 * Si * df) * n / 2
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    return np.fft.irfft(amp * np.exp(1j * phase), n)


def damage_rainflow(x: np.ndarray, sn: SNCurve, duration: float, T: float) -> float:
    """Rainflow-count the time history x (duration in s) and scale the Miner damage to T seconds."""
    import rainflow
    cyc = np.array(rainflow.count_cycles(x))
    if cyc.size == 0:
        return 0.0
    ranges, counts = cyc[:, 0], cyc[:, 1]
    return float(np.sum(counts / sn.cycles(ranges / 2)) * T / duration)


def life_hours(damage_per_hour: float) -> float:
    return np.inf if damage_per_hour <= 0 else 1.0 / damage_per_hour
