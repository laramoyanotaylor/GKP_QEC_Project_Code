"""Explicit hard-GKP validation for surface, toric, and colour codes.

Gaussian shifts are sampled, reduced modulo sqrt(pi), and converted to binary
Pauli faults.  The decoder receives only the uniform marginal error rate, not
the shot-specific modular residual.  This is statistically equivalent to the
ordinary Bernoulli-Pauli comparison at the same p_GKP.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pymatching
from scipy.optimize import brentq
from scipy.special import ndtr

from GKP_surface_2d_array import surface_checks, surface_logicals
from GKP_toric_2d_array import (
    save_result, toric_x_logicals, toric_x_stabilisers,
)


SQRT_PI = math.sqrt(math.pi)
DISTANCES = {
    "surface": (3, 5, 7, 9, 11),
    "toric": (3, 5, 7, 9, 11),
    "colour": (3, 5, 7),
}


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    probability = 0.0
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * SQRT_PI / sigma
        upper = (2 * m + 1.5) * SQRT_PI / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability)


def sigma_for_physical_error_rate(probability):
    if not 0 < probability < 0.5:
        raise ValueError("physical probabilities must lie between zero and 0.5")
    return brentq(lambda sigma: gkp_physical_error_rate(sigma) - probability,
                  1e-3, 5.0)


def probabilities_from_result(path):
    """Read the exact physical-probability grid stored in a result JSON."""
    with path.open() as file:
        root = next(iter(json.load(file).values()))
    if isinstance(root.get("physical_error_probability"), list):
        values = root["physical_error_probability"]
    else:
        values = [
            point["physical_error_probability"]
            for point in root.values()
            if isinstance(point, dict) and "physical_error_probability" in point
        ]
    probabilities = sorted({round(float(value), 12) for value in values})
    if not probabilities:
        raise ValueError(f"no physical-error probabilities found in {path}")
    return np.asarray(probabilities, dtype=float)


def hard_gkp_faults(rng, sigma, count):
    displacement = rng.normal(0.0, sigma, count)
    cells = np.rint(displacement / SQRT_PI).astype(np.int64)
    return (cells & 1).astype(np.uint8)


def simulate_matching(topology, distance, sigmas, args):
    if topology == "toric":
        checks = toric_x_stabilisers(distance)
        logical = toric_x_logicals(distance)
        channels = ((checks, logical),)
    else:
        hx, hz = surface_checks(distance)
        logical_x, logical_z = surface_logicals(distance)
        channels = ((hz, logical_z), (hx, logical_x))

    output = args.output or Path(
        f"hard_gkp_{topology}_2d_d{distance}_results.json"
    )
    if output.exists():
        with output.open() as file:
            data = json.load(file)
    else:
        data = {str(distance): {}}
    saved = data.setdefault(str(distance), {})
    rng = np.random.default_rng(args.seed)

    for sigma in sigmas:
        sigma = float(sigma)
        probability = gkp_physical_error_rate(sigma)
        key = f"{sigma:.5f}"
        point = saved.get(key, {})
        trials = int(point.get("trials", 0))
        errors = int(point.get("errors", 0))
        weight = np.log((1 - probability) / probability)
        decoders = [pymatching.Matching.from_check_matrix(h, weights=weight)
                    for h, _ in channels]
        print(f"hard GKP {topology} d={distance}, sigma={sigma:.5f}, "
              f"p_GKP={probability:.6g}: starting at {trials}")

        while trials < args.max_trials and (
            trials < args.min_trials or errors < args.min_errors
        ):
            for (checks, logical), decoder in zip(channels, decoders):
                fault = hard_gkp_faults(rng, sigma, checks.shape[1])
                syndrome = np.asarray(checks @ fault).ravel() % 2
                correction = decoder.decode(syndrome).astype(np.uint8)
                logical_result = np.asarray(logical @ (fault ^ correction)).ravel() % 2
                errors += int(np.any(logical_result))
            trials += 1
            if trials % 500 == 0:
                saved[key] = {"trials": trials, "errors": errors}
                save_result(output, data)

        samples = len(channels) * trials
        rate = errors / samples
        saved[key] = {
            "sigma": sigma,
            "physical_error_probability": probability,
            "rate": rate,
            "error_bar": math.sqrt(rate * (1 - rate) / samples),
            "trials": trials,
            "decoded_error_channels": samples,
            "errors": errors,
            "noise_model": "explicit_gaussian_hard_gkp",
            "decoder_priors": "uniform marginal p_GKP",
            "uses_gkp_analog_information": False,
            "measurement_noise": False,
        }
        save_result(output, data)
        print(f"hard GKP {topology} d={distance}: {rate:.6g} ({errors}/{samples})")


def save_colour(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2)
    temporary.replace(path)


def simulate_colour(distance, sigmas, args):
    from ldpc import BpOsdDecoder
    from ldpc.mod2 import rank
    from colour_code_triangular_2d import is_logical_error, triangular_color_code

    hx = triangular_color_code(distance)
    hz = hx.copy()
    n = hx.shape[1]
    rank_hx, rank_hz = rank(hx), rank(hz)
    assert n - rank_hx - rank_hz == 1
    output = args.output or Path(
        f"hard_gkp_colour_code_triangular_xz_d{distance}_results.json"
    )
    if output.exists():
        with output.open() as file:
            data = json.load(file)
    else:
        data = {}
    saved = data.setdefault(f"hard_gkp_triangular_color_xz_n{n}_d{distance}", {})
    rng = np.random.default_rng(args.seed)
    z_decoder = BpOsdDecoder(
        hx, error_rate=0.1, bp_method="product_sum", max_iter=n,
        schedule="serial", osd_method="osd_cs", osd_order=2,
    )
    x_decoder = BpOsdDecoder(
        hz, error_rate=0.1, bp_method="product_sum", max_iter=n,
        schedule="serial", osd_method="osd_cs", osd_order=2,
    )

    for sigma in sigmas:
        sigma = float(sigma)
        probability = gkp_physical_error_rate(sigma)
        channel = np.full(n, probability)
        z_decoder.update_channel_probs(channel)
        x_decoder.update_channel_probs(channel)
        key = f"{sigma:.5f}"
        point = saved.get(key, {})
        trials = int(point.get("trials", 0))
        block_errors = int(point.get("block_errors", 0))
        x_errors = int(point.get("x_errors", 0))
        z_errors = int(point.get("z_errors", 0))
        print(f"hard GKP colour d={distance}, sigma={sigma:.5f}, "
              f"p_GKP={probability:.6g}: starting at {trials}")

        while trials < args.max_trials and (
            trials < args.min_trials or block_errors < args.min_errors
        ):
            x_fault = hard_gkp_faults(rng, sigma, n)
            z_fault = hard_gkp_faults(rng, sigma, n)
            z_syndrome = np.asarray(hx @ z_fault).ravel().astype(np.uint8) % 2
            z_correction = z_decoder.decode(z_syndrome).astype(np.uint8)
            z_failed = is_logical_error(z_fault ^ z_correction, hz, rank_hz)
            x_syndrome = np.asarray(hz @ x_fault).ravel().astype(np.uint8) % 2
            x_correction = x_decoder.decode(x_syndrome).astype(np.uint8)
            x_failed = is_logical_error(x_fault ^ x_correction, hx, rank_hx)
            x_errors += int(x_failed)
            z_errors += int(z_failed)
            block_errors += int(x_failed or z_failed)
            trials += 1
            if trials % 500 == 0:
                saved[key] = {
                    "trials": trials, "block_errors": block_errors,
                    "x_errors": x_errors, "z_errors": z_errors,
                }
                save_colour(output, data)

        block_rate = block_errors / trials
        x_rate, z_rate = x_errors / trials, z_errors / trials
        saved[key] = {
            "sigma": sigma,
            "physical_error_probability": probability,
            "block_rate": block_rate, "x_rate": x_rate, "z_rate": z_rate,
            "block_error_bar": math.sqrt(block_rate * (1 - block_rate) / trials),
            "x_error_bar": math.sqrt(x_rate * (1 - x_rate) / trials),
            "z_error_bar": math.sqrt(z_rate * (1 - z_rate) / trials),
            "trials": trials, "block_errors": block_errors,
            "x_errors": x_errors, "z_errors": z_errors,
            "noise_model": "explicit_gaussian_hard_gkp",
            "decoder_priors": "uniform marginal p_GKP",
            "uses_gkp_analog_information": False,
        }
        save_colour(output, data)
        print(f"hard GKP colour d={distance}: block={block_rate:.6g} "
              f"({block_errors}/{trials})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=tuple(DISTANCES), required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--probability-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--probabilities-from", type=Path,
                        help="use the exact probability grid from a result JSON")
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--min-trials", type=int, default=10_000)
    parser.add_argument("--min-errors", type=int, default=500)
    parser.add_argument("--max-trials", type=int, default=200_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.distance not in DISTANCES[args.topology]:
        parser.error(f"unsupported {args.topology} distance {args.distance}")
    if bool(args.probability_range) == bool(args.probabilities_from):
        parser.error("specify exactly one of --probability-range or --probabilities-from")
    probabilities = (
        probabilities_from_result(args.probabilities_from)
        if args.probabilities_from
        else np.linspace(*args.probability_range, args.num_points)
    )
    print(f"Using {len(probabilities)} physical probabilities: "
          f"{probabilities[0]:.6g} to {probabilities[-1]:.6g}")
    sigmas = [sigma_for_physical_error_rate(p) for p in probabilities]
    if args.topology == "colour":
        simulate_colour(args.distance, sigmas, args)
    else:
        simulate_matching(args.topology, args.distance, sigmas, args)


if __name__ == "__main__":
    main()
