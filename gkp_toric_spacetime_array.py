"""Checkpointed soft-GKP spacetime toric simulation for one Slurm task."""

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, eye, hstack, kron, vstack

from GKP_toric_2d_array import (
    DISTANCES,
    SQRT_PI,
    sigma_for_physical_error_rate,
    toric_x_logicals,
    toric_x_stabilisers,
)


def spacetime_check_matrix(checks, rounds):
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


def gkp_fault_and_weight(rng, sigma, count):
    displacement = rng.normal(0.0, sigma, count)
    residual = (displacement + SQRT_PI / 2) % SQRT_PI - SQRT_PI / 2
    fault = np.rint((displacement - residual) / SQRT_PI).astype(np.uint8) % 2
    weight = ((SQRT_PI - np.abs(residual))**2 - residual**2) / (2 * sigma**2)
    return fault, weight


def save_checkpoint(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temporary.replace(path)


def simulate(distance, sigmas, max_trials, min_trials, min_errors, seed, output):
    rounds = distance
    checks = toric_x_stabilisers(distance)
    logicals = toric_x_logicals(distance)
    num_checks, num_qubits = checks.shape
    detector_matrix = spacetime_check_matrix(checks, rounds)
    rng = np.random.default_rng(seed)

    if output.exists():
        with output.open() as file:
            data = json.load(file)
    else:
        data = {}
    points = data.setdefault(str(distance), {})

    for sigma_value in sigmas:
        sigma = float(sigma_value)
        key = f"{sigma:.5f}"
        previous = points.get(key, {})
        trials = int(previous.get("trials", 0))
        errors = int(previous.get("errors", 0))
        print(f"soft-GKP spacetime toric d={distance}, sigma={sigma:.5f}: "
              f"starting at {trials} trials, {errors} errors")

        while trials < max_trials and (trials < min_trials or errors < min_errors):
            batch_size = min(250, max_trials - trials)
            for _ in range(batch_size):
                data_faults = []
                data_weights = []
                measurement_faults = []
                measurement_weights = []
                cumulative = np.zeros(num_qubits, dtype=np.uint8)
                noisy_syndromes = []

                for _round in range(rounds):
                    fault, weight = gkp_fault_and_weight(rng, sigma, num_qubits)
                    cumulative ^= fault
                    data_faults.append(fault)
                    data_weights.append(weight)

                    measurement_fault, measurement_weight = gkp_fault_and_weight(
                        rng, sigma, num_checks
                    )
                    ideal = np.asarray(checks @ cumulative).ravel() % 2
                    noisy_syndromes.append(ideal.astype(np.uint8) ^ measurement_fault)
                    measurement_faults.append(measurement_fault)
                    measurement_weights.append(measurement_weight)

                detectors = []
                previous_syndrome = np.zeros(num_checks, dtype=np.uint8)
                for noisy in noisy_syndromes:
                    detectors.append(noisy ^ previous_syndrome)
                    previous_syndrome = noisy
                perfect_final = (np.asarray(checks @ cumulative).ravel() % 2).astype(np.uint8)
                detectors.append(perfect_final ^ previous_syndrome)

                weights = np.concatenate(data_weights + measurement_weights)
                decoder = pymatching.Matching.from_check_matrix(
                    detector_matrix, weights=weights
                )
                correction = decoder.decode(np.concatenate(detectors)).astype(np.uint8)
                cumulative_correction = np.bitwise_xor.reduce(
                    correction[:rounds * num_qubits].reshape(rounds, num_qubits),
                    axis=0,
                )
                residual = cumulative ^ cumulative_correction
                errors += int(np.any(np.asarray(logicals @ residual).ravel() % 2))

            trials += batch_size
            rate = errors / trials
            points[key] = {
                "sigma": sigma,
                "rate": rate,
                "error_bar": float(np.sqrt(rate * (1 - rate) / trials)),
                "trials": trials,
                "errors": errors,
                "rounds": rounds,
                "noise_model": "gkp_data_and_measurement_with_analog_weights",
                "uses_gkp_analog_information": True,
            }
            save_checkpoint(output, data)

        print(f"soft-GKP spacetime toric d={distance}: "
              f"{errors / trials:.6g} ({errors}/{trials})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=DISTANCES, required=True)
    parser.add_argument("--probability-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"), required=True)
    parser.add_argument("--num-points", type=int, default=21)
    parser.add_argument("--max-trials", type=int, default=2_000_000)
    parser.add_argument("--min-trials", type=int, default=50_000)
    parser.add_argument("--min-errors", type=int, default=500)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probabilities = np.linspace(*args.probability_range, args.num_points)
    sigmas = [sigma_for_physical_error_rate(p) for p in probabilities]
    simulate(args.distance, sigmas, args.max_trials, args.min_trials,
             args.min_errors, args.seed, args.output)


if __name__ == "__main__":
    main()

