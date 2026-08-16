"""Run the corrected soft-GKP toric spacetime simulation for distance 15."""

import os
import runpy
from pathlib import Path

os.environ["TORIC_DISTANCES"] = "15"
os.environ["TORIC_SIGMA_VALUES"] = "0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50"
os.environ["TORIC_CHECKPOINT_FILE"] = "gkp_toric_d15_results.json"
os.environ["TORIC_SIGMA_PLOT_FILE"] = "GKP_toric_d15_vs_sigma.png"
os.environ["TORIC_PHYSICAL_PLOT_FILE"] = "GKP_toric_d15_vs_physical.png"

runpy.run_path(
    Path(__file__).with_name("GKP_toric_cluster.py"),
    run_name="__main__",
)
