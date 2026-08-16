"""Standard binary-noise baseline for the spacetime toric-code tutorial.

This matches ``GKP_toric_cluster.py`` (T=d rounds, noisy syndrome
measurements and a perfect final measurement), but replaces each analogue GKP
channel by an independent 0/1 Bernoulli channel.  The Bernoulli probability is
the hard-decision GKP error probability corresponding to sigma, and the
decoder receives only that uniform probability (no analogue residuals).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, eye, hstack, kron, vstack
from scipy.special import ndtr
from scipy.optimize import brentq

from GKP_toric_2d_array import toric_x_logicals, toric_x_stabilisers


SQRT_PI = np.sqrt(np.pi)
DISTANCES = (3, 5, 7, 9, 11)
# The standard phenomenological threshold is near p=3%, or sigma about 0.41.
DEFAULT_SIGMAS = np.array(
    [0.38, 0.39, 0.395, 0.400, 0.405, 0.410, 0.415, 0.420, 0.430]
)


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    """Probability that ideal GKP hard correction gives a binary fault."""
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


def spacetime_check_matrix(checks, rounds):
    """Detector matrix for data faults and measurement faults."""
    num_checks, num_qubits = checks.shape
    space = vstack([
        kron(eye(rounds, dtype=np.uint8), checks),
        csc_matrix((num_checks, rounds * num_qubits), dtype=np.uint8),
    ])
    identity = eye(num_checks, dtype=np.uint8)
    zero = csc_matrix((num_checks, num_checks), dtype=np.uint8)
    time = vstack([
        hstack([
            identity if detector_round in (fault_round, fault_round + 1) else zero
            for fault_round in range(rounds)
        ])
        for detector_round in range(rounds + 1)
    ])
    matrix = hstack([space, time]).tocsc()
    matrix.data %= 2
    matrix.eliminate_zeros()
    return matrix


def save_checkpoint(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2)
    temporary.replace(path)


def load_checkpoint(path, distance):
    if path.exists():
        with path.open() as file:
            data = json.load(file)
    else:
        data = {}
    data.setdefault(str(distance), {})
    return data


def simulate(distance, sigmas, max_trials, min_trials, min_errors, seed, output):
    rounds = distance
    checks = toric_x_stabilisers(distance)
    logicals = toric_x_logicals(distance)
    num_qubits = checks.shape[1]
    num_checks = checks.shape[0]
    detector_matrix = spacetime_check_matrix(checks, rounds)
    rng = np.random.default_rng(seed)
    data = load_checkpoint(output, distance)
    points = data[str(distance)]

    for sigma_value in sigmas:
        sigma = float(sigma_value)
        key = f"{sigma:.5f}"
        probability = gkp_physical_error_rate(sigma)
        weight = np.log((1.0 - probability) / probability)
        decoder = pymatching.Matching.from_check_matrix(
            detector_matrix, weights=weight
        )
        point = points.get(key, {})
        trials = int(point.get("trials", 0))
        errors = int(point.get("errors", 0))
        print(
            f"standard toric d={distance}, T={rounds}, sigma={sigma:.5f}, "
            f"p={probability:.6g}: starting at {trials} trials, {errors} errors"
        )

        while trials < max_trials and (trials < min_trials or errors < min_errors):
            batch_size = min(500, max_trials - trials)
            for _ in range(batch_size):
                data_faults = (
                    rng.random((rounds, num_qubits)) < probability
                ).astype(np.uint8)
                measurement_faults = (
                    rng.random((rounds, num_checks)) < probability
                ).astype(np.uint8)

                cumulative = np.zeros(num_qubits, dtype=np.uint8)
                previous_noisy = np.zeros(num_checks, dtype=np.uint8)
                detectors = []
                for round_index in range(rounds):
                    cumulative ^= data_faults[round_index]
                    noisy = (
                        (np.asarray(checks @ cumulative).ravel() % 2).astype(np.uint8)
                        ^ measurement_faults[round_index]
                    )
                    detectors.append(noisy ^ previous_noisy)
                    previous_noisy = noisy
                perfect_final = (
                    np.asarray(checks @ cumulative).ravel() % 2
                ).astype(np.uint8)
                detectors.append(perfect_final ^ previous_noisy)

                correction = decoder.decode(np.concatenate(detectors)).astype(np.uint8)
                cumulative_correction = np.bitwise_xor.reduce(
                    correction[: rounds * num_qubits].reshape(rounds, num_qubits),
                    axis=0,
                )
                residual = cumulative ^ cumulative_correction
                errors += int(np.any(np.asarray(logicals @ residual).ravel() % 2))

            trials += batch_size
            rate = errors / trials
            points[key] = {
                "sigma": sigma,
                "physical_error_probability": probability,
                "rate": rate,
                "error_bar": float(np.sqrt(rate * (1 - rate) / trials)),
                "trials": trials,
                "errors": errors,
                "rounds": rounds,
                "noise_model": "independent_bernoulli_data_and_measurement",
                "uses_gkp_analog_information": False,
            }
            save_checkpoint(output, data)

        print(f"standard toric d={distance}: {errors / trials:.6g} ({errors}/{trials})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=DISTANCES, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--probability-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--max-trials", type=int, default=2_000_000)
    parser.add_argument("--min-trials", type=int, default=20_000)
    parser.add_argument("--min-errors", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probability_range:
        probabilities = np.linspace(*args.probability_range, args.num_points)
        sigmas = [sigma_for_physical_error_rate(p) for p in probabilities]
    else:
        sigmas = args.sigmas if args.sigmas is not None else DEFAULT_SIGMAS
    output = args.output or Path(
        f"standard_toric_spacetime_d{args.distance}_results.json"
    )
    simulate(
        args.distance, sigmas, args.max_trials, args.min_trials,
        args.min_errors, args.seed, output,
    )


if __name__ == "__main__":
    main()
