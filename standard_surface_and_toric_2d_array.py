"""One-round surface/toric simulation with ordinary binary Pauli errors.

Sigma is retained as the horizontal-axis parameter for direct comparison with
the GKP runs.  Each sigma is converted to the corresponding physical GKP Pauli
error probability, then every data qubit receives an independent yes/no error
with that probability.  No analog GKP information or measurement noise is used.
"""

import argparse
from pathlib import Path

import numpy as np
import pymatching
from scipy.optimize import brentq

from GKP_surface_2d_array import surface_checks, surface_logicals
from GKP_toric_2d_array import (
    DEFAULT_SIGMAS,
    DISTANCES,
    load_result,
    save_result,
    toric_x_logicals,
    toric_x_stabilisers,
)


SQRT_PI = np.sqrt(np.pi)


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    """Physical Pauli-fault probability used for the Bernoulli comparison."""
    from scipy.special import ndtr

    probability = 0.0
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * SQRT_PI / sigma
        upper = (2 * m + 1.5) * SQRT_PI / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability)


def sigma_for_physical_error_rate(probability):
    """Invert the hard-decision GKP channel so grids can be uniform in p."""
    if not 0 < probability < 0.5:
        raise ValueError("physical error probabilities must lie between 0 and 0.5")
    return brentq(lambda sigma: gkp_physical_error_rate(sigma) - probability,
                  1e-3, 5.0)


def binary_failure(rng, probability, checks, logical, decoder):
    error = (rng.random(checks.shape[1]) < probability).astype(np.uint8)
    syndrome = np.asarray(checks @ error).ravel() % 2
    correction = decoder.decode(syndrome).astype(np.uint8)
    residual = error ^ correction
    logical_result = np.asarray(logical @ residual).ravel() % 2
    return int(np.any(logical_result))


def simulate(topology, distance, sigmas, max_trials, min_trials, min_errors, seed, output):
    if topology == "toric":
        checks = toric_x_stabilisers(distance)
        logical = toric_x_logicals(distance)
        channels = ((checks, logical),)
    else:
        hx, hz = surface_checks(distance)
        logical_x, logical_z = surface_logicals(distance)
        # Surface boundaries distinguish X and Z errors, so simulate both.
        channels = ((hz, logical_z), (hx, logical_x))

    rng = np.random.default_rng(seed)
    data = load_result(output, distance)
    distance_data = data[str(distance)]

    for sigma in sigmas:
        sigma = float(sigma)
        key = f"{sigma:.5f}"
        probability = gkp_physical_error_rate(sigma)
        weight = np.log((1 - probability) / probability)
        decoders = [
            pymatching.Matching.from_check_matrix(check, weights=weight)
            for check, _ in channels
        ]

        previous = distance_data.get(key, {})
        trials = int(previous.get("trials", 0))
        errors = int(previous.get("errors", 0))
        print(
            f"standard {topology} d={distance}, sigma={sigma:.5f}, "
            f"p={probability:.6g}: starting at {trials} trials"
        )

        while trials < max_trials and (trials < min_trials or errors < min_errors):
            for (check, logical), decoder in zip(channels, decoders):
                errors += binary_failure(rng, probability, check, logical, decoder)
            trials += 1

            if trials % 500 == 0:
                distance_data[key] = {"trials": trials, "errors": errors}
                save_result(output, data)

        samples = len(channels) * trials
        rate = errors / samples
        error_bar = np.sqrt(rate * (1 - rate) / samples)
        distance_data[key] = {
            "sigma": sigma,
            "physical_error_probability": probability,
            "rate": rate,
            "error_bar": error_bar,
            "trials": trials,
            "decoded_error_channels": samples,
            "errors": errors,
            "noise_model": "bernoulli_pauli",
            "uses_gkp_analog_information": False,
            "measurement_noise": False,
            "correction_rounds": 1,
        }
        save_result(output, data)
        print(f"standard {topology} d={distance}: {rate:.6g} ({errors}/{samples})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("surface", "toric"), required=True)
    parser.add_argument("--distance", type=int, choices=DISTANCES, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--probability-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--max-trials", type=int, default=50_000)
    parser.add_argument("--min-trials", type=int, default=1_000)
    parser.add_argument("--min-errors", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probability_range:
        probabilities = np.linspace(*args.probability_range, args.num_points)
        sigmas = [sigma_for_physical_error_rate(p) for p in probabilities]
    else:
        sigmas = args.sigmas or DEFAULT_SIGMAS
    output = args.output or Path(
        f"standard_{args.topology}_2d_d{args.distance}_results.json"
    )
    simulate(
        args.topology,
        args.distance,
        sigmas,
        args.max_trials,
        args.min_trials,
        args.min_errors,
        args.seed,
        output,
    )


if __name__ == "__main__":
    main()
