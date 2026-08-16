"""Launch one distance of the corrected soft-GKP toric simulation."""

import argparse
import os
import runpy
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description="Run one of the d=3, 5, 7, 9, or 11 soft-GKP toric simulations."
)
parser.add_argument("--distance", type=int, choices=(3, 5, 7, 9, 11), required=True)
parser.add_argument("--sigma-range", type=float, nargs=2, default=(0.45, 0.50),
                    metavar=("MIN", "MAX"))
parser.add_argument("--num-points", type=int, default=20)
args = parser.parse_args()

distance = args.distance

# Use identical sigma points so the three distances can be compared directly.
os.environ["TORIC_DISTANCES"] = str(distance)
os.environ["TORIC_SIGMA_VALUES"] = ",".join(
    f"{sigma:.8f}" for sigma in np.linspace(*args.sigma_range, args.num_points)
)

# Every array task must have its own outputs.
os.environ["TORIC_CHECKPOINT_FILE"] = f"gkp_toric_d{distance}_results.json"
os.environ["TORIC_SIGMA_PLOT_FILE"] = f"GKP_toric_d{distance}_vs_sigma.png"
os.environ["TORIC_PHYSICAL_PLOT_FILE"] = (
    f"GKP_toric_d{distance}_vs_physical.png"
)

runpy.run_path(
    Path(__file__).with_name("GKP_toric_cluster.py"),
    run_name="__main__",
)
