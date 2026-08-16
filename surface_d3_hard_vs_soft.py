"""Compare hard and soft GKP decoding for a distance-3 planar surface code.

The simulation uses one perfect surface-code syndrome round.  Each Monte Carlo
shot supplies independent q and p quadratures, so both the X- and Z-check
decoders are sampled.  Results are checkpointed after every sigma value.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching
from scipy.sparse import csc_matrix, eye, hstack, kron
from scipy.special import ndtr


DISTANCE = 3
SQRT_PI = np.sqrt(np.pi)


def repetition_code(d):
    rows = np.repeat(np.arange(d - 1), 2)
    cols = np.column_stack((np.arange(d - 1), np.arange(1, d))).ravel()
    return csc_matrix(
        (np.ones(2 * (d - 1), dtype=np.uint8), (rows, cols)),
        shape=(d - 1, d),
    )


def surface_code(d=DISTANCE):
    """Return X/Z checks and corresponding logical vectors."""
    h = repetition_code(d)
    i_n = eye(d, dtype=np.uint8, format="csc")
    i_r = eye(d - 1, dtype=np.uint8, format="csc")
    hx = hstack((kron(h, i_n), kron(i_r, h.T)), format="csc")
    hz = hstack((kron(i_n, h), kron(h.T, i_r)), format="csc")

    n = d * d + (d - 1) ** 2
    logical_x = np.zeros(n, dtype=np.uint8)
    logical_z = np.zeros(n, dtype=np.uint8)
    logical_x[:d] = 1
    logical_z[np.arange(d) * d] = 1
    return ((hx, logical_x), (hz, logical_z))


def physical_gkp_error_rate(sigma, intervals=100):
    probability = 0.0
    for m in range(intervals):
        lower = (2 * m + 0.5) * SQRT_PI / sigma
        upper = (2 * m + 1.5) * SQRT_PI / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return probability


def sample_gkp(rng, sigma, count):
    displacement = rng.normal(0.0, sigma, count)
    residual = (displacement + SQRT_PI / 2) % SQRT_PI - SQRT_PI / 2
    faults = np.rint((displacement - residual) / SQRT_PI).astype(np.uint8) % 2
    soft_weights = (
        (SQRT_PI - np.abs(residual)) ** 2 - residual**2
    ) / (2 * sigma**2)
    return faults, soft_weights


def logical_failure(checks, logical, faults, decoder):
    syndrome = np.asarray(checks @ faults).ravel() % 2
    correction = decoder.decode(syndrome).astype(np.uint8)
    return int(np.dot(faults ^ correction, logical) % 2)


def standard_error(errors, samples):
    rate = errors / samples
    return np.sqrt(rate * (1.0 - rate) / samples)


def save_outputs(rows, output_prefix):
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    plot_path = output_prefix.with_suffix(".png")

    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n")
    temporary.replace(json_path)

    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axis = plt.subplots(figsize=(7, 5))
    sigmas = np.asarray([row["sigma"] for row in rows])
    styles = {
        "hard": {"marker": "s", "linestyle": "--"},
        "soft": {"marker": "o", "linestyle": "-"},
    }
    for method in ("hard", "soft"):
        rates = np.asarray([row[f"{method}_rate"] for row in rows])
        errors = np.asarray([row[f"{method}_error_bar"] for row in rows])
        axis.errorbar(
            sigmas, rates, yerr=errors, capsize=3, color="tab:blue",
            label=f"{method.title()} decoding", **styles[method],
        )
    axis.set_xlabel(r"Displacement spread $\sigma$")
    axis.set_ylabel("Logical error rate per decoded quadrature")
    axis.set_title("Distance-3 planar surface code")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def simulate(sigmas, shots, seed, output_prefix):
    rng = np.random.default_rng(seed)
    code = surface_code()
    rows = []

    for sigma in sigmas:
        p_gkp = physical_gkp_error_rate(sigma)
        hard_weight = np.log((1.0 - p_gkp) / p_gkp)
        hard_decoders = [
            pymatching.Matching.from_check_matrix(checks, weights=hard_weight)
            for checks, _ in code
        ]
        hard_failures = 0
        soft_failures = 0

        for shot in range(shots):
            for index, (checks, logical) in enumerate(code):
                faults, weights = sample_gkp(rng, sigma, checks.shape[1])
                hard_failures += logical_failure(
                    checks, logical, faults, hard_decoders[index]
                )
                soft_decoder = pymatching.Matching.from_check_matrix(
                    checks, weights=weights
                )
                soft_failures += logical_failure(
                    checks, logical, faults, soft_decoder
                )
            if (shot + 1) % 10_000 == 0:
                print(f"sigma={sigma:.3f}: {shot + 1}/{shots} shots", flush=True)

        samples = 2 * shots
        row = {
            "distance": DISTANCE,
            "sigma": float(sigma),
            "physical_gkp_error_rate": p_gkp,
            "shots": shots,
            "decoded_quadratures": samples,
            "hard_failures": hard_failures,
            "hard_rate": hard_failures / samples,
            "hard_error_bar": standard_error(hard_failures, samples),
            "soft_failures": soft_failures,
            "soft_rate": soft_failures / samples,
            "soft_error_bar": standard_error(soft_failures, samples),
        }
        rows.append(row)
        save_outputs(rows, output_prefix)
        print(
            f"sigma={sigma:.3f}: hard={row['hard_rate']:.6g} +/- "
            f"{row['hard_error_bar']:.2g}, soft={row['soft_rate']:.6g} +/- "
            f"{row['soft_error_bar']:.2g}", flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sigma-min", type=float, default=0.4)
    parser.add_argument("--sigma-max", type=float, default=0.7)
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument(
        "--output-prefix", type=Path,
        default=Path("surface_d3_hard_vs_soft_results"),
    )
    args = parser.parse_args()
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.num_points)
    simulate(sigmas, args.shots, args.seed, args.output_prefix)


if __name__ == "__main__":
    main()
