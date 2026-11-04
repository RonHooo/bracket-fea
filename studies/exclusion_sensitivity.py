"""Sensitivity of the results to the size of the excluded zone around the bolt holes.

For the plain designs the governing stress sits in the base plate just outside the excluded zone, so
the peak stress and the fatigue life depend on how large that zone is. The FE solution is the same;
only the evaluation region changes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bracketfea.analysis import run_design, random_vibration
from bracketfea.geometry import BracketParams
from bracketfea.problem import LoadCase

DESIGNS = {
    "lightest feasible (DOE id 220)": dict(t_b=9.902, t_w=7.071, r_f=8.598, L_g=0.0, t_g=6.326, d_h=9.738),
    "plain t=9": dict(t_b=9.0, t_w=9.0, r_f=5.0),
}
FACTORS = [1.5, 2.0, 2.5, 3.0]


def region(nodes, p, k_bolt):
    x, y, z = nodes.T
    nb = np.min([np.hypot(x - p.x_bolt, y - s * p.y_bolt) for s in (-1, 1)], axis=0)
    nm = np.hypot(y, z - p.z_mount)
    return (nb > k_bolt * p.d_bolt) & (nm > 1.5 * p.d_mount)


def main():
    lc = LoadCase()
    rows = []
    for name, kw in DESIGNS.items():
        p = BracketParams(**kw)
        r = run_design(p, lc=lc, threads=3)
        for k in FACTORS:
            r.region = region(r.mesh.nodes, p, k)
            rv = random_vibration(r, lc)
            worst = min(rv, key=lambda a: rv[a]["life_dirlik_h"])
            rows.append(dict(design=name, exclusion_radius_over_d=k, vm_max=r.vm_max,
                             life_h=rv[worst]["life_dirlik_h"], worst_axis=worst,
                             vm_rms=rv[worst]["vm_rms_max"]))
            print(json.dumps(rows[-1], default=float), flush=True)
    out = Path("results/optimization")
    pd.DataFrame(rows).to_csv(out / "exclusion_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
