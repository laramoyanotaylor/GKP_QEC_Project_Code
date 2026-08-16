"""Standard X/Z Bernoulli simulations of triangular 2D colour codes.

Each data qubit independently receives X and Z errors with marginal
probability p_GKP(sigma). Unlike the analogue-GKP simulation, BP+OSD receives
only the same uniform channel probability for every qubit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ldpc import BpOsdDecoder
from ldpc.mod2 import rank
from scipy.special import ndtr
from scipy.optimize import brentq

from colour_code_triangular_2d import is_logical_error, triangular_color_code


SQRT_PI = np.sqrt(np.pi)
DEFAULT_SIGMAS = np.round(np.arange(0.20, 0.701, 0.01), 2)


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    """Marginal inner-GKP logical error = outer-code physical error rate."""
    probability = 0.0
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * SQRT_PI / sigma
        upper = (2 * m + 1.5) * SQRT_PI / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability)


def sigma_for_physical_error_rate(probability):
    return brentq(lambda sigma: gkp_physical_error_rate(sigma) - probability,
                  0.05, 2.0)


def save_result(filename, data):
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2)
    temporary.replace(filename)


def simulate(distance, sigmas, max_trials, min_trials, min_errors, seed, output):
    hx = triangular_color_code(distance)
    hz = hx.copy()
    n = hx.shape[1]
    expected_n = (3 * distance**2 + 1) // 4
    assert n == expected_n
    assert n - rank(hx) - rank(hz) == 1
    rank_hx, rank_hz = rank(hx), rank(hz)
    rng = np.random.default_rng(seed)

    if output.exists():
        with output.open() as file:
            data = json.load(file)
        print(f"Loaded checkpoint from {output}")
    else:
        data = {}
    code_data = data.setdefault(
        f"standard_triangular_color_xz_n{n}_d{distance}", {}
    )

    z_decoder = BpOsdDecoder(
        hx,
        error_rate=0.1,
        bp_method="product_sum",
        max_iter=n,
        schedule="serial",
        osd_method="osd_cs",
        osd_order=2,
    )
    x_decoder = BpOsdDecoder(
        hz,
        error_rate=0.1,
        bp_method="product_sum",
        max_iter=n,
        schedule="serial",
        osd_method="osd_cs",
        osd_order=2,
    )

    for sigma in sigmas:
        sigma = round(float(sigma), 5)
        key = f"{sigma:.5f}"
        probability = gkp_physical_error_rate(sigma)
        channel = np.full(n, probability)
        z_decoder.update_channel_probs(channel)
        x_decoder.update_channel_probs(channel)
        point = code_data.get(key, {})
        trials = int(point.get("trials", 0))
        block_errors = int(point.get("block_errors", 0))
        x_errors = int(point.get("x_errors", 0))
        z_errors = int(point.get("z_errors", 0))
        print(f"d={distance}, sigma={sigma:.5f}, p={probability:.6g}: "
              f"starting at {trials} trials, {block_errors} block errors")

        while (trials < max_trials
               and (trials < min_trials or block_errors < min_errors)):
            batch_size = min(500, max_trials - trials)
            for _ in range(batch_size):
                x_error = (rng.random(n) < probability).astype(np.uint8)
                z_error = (rng.random(n) < probability).astype(np.uint8)

                z_syndrome = np.asarray(hx @ z_error).ravel() % 2
                z_correction = z_decoder.decode(
                    z_syndrome.astype(np.uint8)
                ).astype(np.uint8)
                assert np.array_equal(
                    np.asarray(hx @ z_correction).ravel() % 2, z_syndrome
                )
                z_failed = is_logical_error(
                    z_error ^ z_correction, hz, rank_hz
                )

                x_syndrome = np.asarray(hz @ x_error).ravel() % 2
                x_correction = x_decoder.decode(
                    x_syndrome.astype(np.uint8)
                ).astype(np.uint8)
                assert np.array_equal(
                    np.asarray(hz @ x_correction).ravel() % 2, x_syndrome
                )
                x_failed = is_logical_error(
                    x_error ^ x_correction, hx, rank_hx
                )

                x_errors += int(x_failed)
                z_errors += int(z_failed)
                block_errors += int(x_failed or z_failed)
            trials += batch_size
            block_rate = block_errors / trials
            x_rate = x_errors / trials
            z_rate = z_errors / trials

            def error_bar(rate):
                return np.sqrt(rate * (1 - rate) / trials)

            code_data[key] = {
                "sigma": sigma,
                "physical_error_probability": probability,
                "block_rate": block_rate,
                "x_rate": x_rate,
                "z_rate": z_rate,
                "block_error_bar": error_bar(block_rate),
                "x_error_bar": error_bar(x_rate),
                "z_error_bar": error_bar(z_rate),
                "trials": trials,
                "block_errors": block_errors,
                "x_errors": x_errors,
                "z_errors": z_errors,
                "noise_model": "independent_bernoulli_pauli_x_and_z",
                "uses_gkp_analog_information": False,
            }
            save_result(output, data)

        print(f"standard colour d={distance}: block={block_errors / trials:.6g} "
              f"({block_errors}/{trials})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--probability-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--max-trials", type=int, default=1_000_000)
    parser.add_argument("--min-trials", type=int, default=10_000)
    parser.add_argument("--min-errors", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sigmas = ([sigma_for_physical_error_rate(p) for p in
               np.linspace(*args.probability_range, args.num_points)]
              if args.probability_range else args.sigmas)
    output = args.output or Path(
        f"standard_colour_code_triangular_xz_d{args.distance}_results.json"
    )
    simulate(args.distance, sigmas, args.max_trials, args.min_trials,
             args.min_errors, args.seed, output)


if __name__ == "__main__":
    main()
