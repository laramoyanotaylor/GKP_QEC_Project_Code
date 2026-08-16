"""Exact majority-vote performance of classical repetition codes."""

import argparse
import csv
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DISTANCES = (3, 5, 7, 9, 11)


def logical_error_rate(distance, physical_error_rate):
    """Exact binomial failure rate; a tie for even distance fails half the time."""
    p = np.asarray(physical_error_rate, dtype=float)
    result = np.zeros_like(p)
    first_failure_weight = distance // 2 + 1
    for weight in range(first_failure_weight, distance + 1):
        result += comb(distance, weight) * p**weight * (1.0 - p)**(distance - weight)
    if distance % 2 == 0:
        weight = distance // 2
        result += 0.5 * comb(distance, weight) * p**weight * (1.0 - p)**weight
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot exact repetition-code logical error rates for p in [0,1]"
    )
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--output-prefix", default="classical_repetition_full_range")
    args = parser.parse_args()
    if args.points < 3:
        parser.error("--points must be at least 3")

    probabilities = np.linspace(0.0, 1.0, args.points)
    results = {
        str(distance): logical_error_rate(distance, probabilities)
        for distance in DEFAULT_DISTANCES
    }

    json_path = f"{args.output_prefix}_results.json"
    with open(json_path, "w") as file:
        json.dump({
            "physical_error_rate": probabilities.tolist(),
            "logical_error_rate": {
                distance: rates.tolist() for distance, rates in results.items()
            },
            "distances": list(DEFAULT_DISTANCES),
            "method": "exact binomial majority-vote probability",
        }, file, indent=2)

    csv_path = f"{args.output_prefix}_results.csv"
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["physical_error_rate"] + [f"d_{d}" for d in DEFAULT_DISTANCES])
        for index, p in enumerate(probabilities):
            writer.writerow([p] + [results[str(d)][index] for d in DEFAULT_DISTANCES])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for distance in DEFAULT_DISTANCES:
        rates = results[str(distance)]
        label = f"d={distance}"
        axes[0].plot(probabilities, rates, label=label)
        axes[1].plot(probabilities, rates, label=label)

    for axis in axes:
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1.3,
                     label="threshold p=0.5")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Physical bit-flip probability p")
        axis.grid(True, which="both", linestyle="--", alpha=0.55)
    axes[0].set_ylabel("Logical block error probability")
    axes[0].set_title("Full range (linear scale)")
    axes[1].set_yscale("symlog", linthresh=1e-8, linscale=0.5)
    axes[1].set_ylabel("Logical block error probability")
    axes[1].set_title("Full range (symmetric-log scale)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Classical Repetition Code: Exact Majority-Vote Decoding")
    fig.tight_layout()
    plot_path = f"{args.output_prefix}_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Results saved to {json_path} and {csv_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
