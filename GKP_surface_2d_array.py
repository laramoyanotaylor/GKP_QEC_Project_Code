"""One-round (2D) soft-GKP planar surface-code simulation for a SLURM array."""

import argparse
from pathlib import Path

import numpy as np
import pymatching
from scipy.sparse import csc_matrix, eye, hstack, kron

from GKP_toric_2d_array import (
    DEFAULT_SIGMAS,
    DISTANCES,
    gkp_faults_and_weights,
    load_result,
    save_result,
    sigma_for_physical_error_rate,
)

SURFACE_DISTANCES = tuple(sorted((*DISTANCES, 4)))


def repetition_code(d):
    rows = np.repeat(np.arange(d - 1), 2)
    cols = np.column_stack((np.arange(d - 1), np.arange(1, d))).ravel()
    return csc_matrix((np.ones(2 * (d - 1), dtype=np.uint8), (rows, cols)), shape=(d - 1, d))


def surface_checks(d):
    """Hypergraph-product planar surface code used in the surface notebook."""
    h = repetition_code(d)
    i_n = eye(d, dtype=np.uint8, format="csc")
    i_r = eye(d - 1, dtype=np.uint8, format="csc")
    hx = hstack((kron(h, i_n), kron(i_r, h.T)), format="csc")
    hz = hstack((kron(i_n, h), kron(h.T, i_r)), format="csc")
    return hx, hz


def surface_logicals(d):
    count = d * d + (d - 1) ** 2
    logical_z = np.zeros(count, dtype=np.uint8)
    logical_x = np.zeros(count, dtype=np.uint8)
    logical_z[np.arange(d) * d] = 1
    logical_x[:d] = 1
    return logical_x, logical_z


def decode_quadrature(rng, sigma, checks, logical):
    fault, weights = gkp_faults_and_weights(rng, sigma, checks.shape[1])
    syndrome = np.asarray(checks @ fault).ravel() % 2  # perfect measurement
    decoder = pymatching.Matching.from_check_matrix(checks, weights=weights)
    correction = decoder.decode(syndrome).astype(np.uint8)
    return int(np.dot(fault ^ correction, logical) % 2)


def simulate(distance, sigmas, max_trials, min_trials, min_errors, seed, output):
    hx, hz = surface_checks(distance)
    logical_x, logical_z = surface_logicals(distance)
    rng = np.random.default_rng(seed)
    data = load_result(output, distance)
    distance_data = data[str(distance)]

    for sigma in sigmas:
        key = f"{sigma:.5f}"
        previous = distance_data.get(key, {})
        trials = int(previous.get("trials", 0))
        errors = int(previous.get("errors", 0))
        print(f"surface d={distance}, sigma={sigma:.5f}: starting at {trials} trials")

        # One shot tests both independent GKP quadratures. The reported rate is
        # therefore per decoded quadrature, matching the surface-code notebook.
        while trials < max_trials and (trials < min_trials or errors < min_errors):
            errors += decode_quadrature(rng, sigma, hz, logical_z)
            errors += decode_quadrature(rng, sigma, hx, logical_x)
            trials += 1

            if trials % 500 == 0:
                distance_data[key] = {"trials": trials, "errors": errors}
                save_result(output, data)

        samples = 2 * trials
        rate = errors / samples
        error_bar = np.sqrt(rate * (1 - rate) / samples)
        distance_data[key] = {
            "sigma": sigma,
            "rate": rate,
            "error_bar": error_bar,
            "trials": trials,
            "decoded_quadratures": samples,
            "errors": errors,
            "measurement_noise": False,
            "correction_rounds": 1,
        }
        save_result(output, data)
        print(f"surface d={distance}, sigma={sigma:.5f}: {rate:.6g} ({errors}/{samples})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=SURFACE_DISTANCES, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=None)
    parser.add_argument("--sigma-range", type=float, nargs=2, metavar=("MIN", "MAX"))
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
        sigmas = (np.linspace(*args.sigma_range, args.num_points)
                  if args.sigma_range else (args.sigmas or DEFAULT_SIGMAS))
    output = args.output or Path(f"gkp_surface_2d_d{args.distance}_results.json")
    simulate(args.distance, sigmas, args.max_trials, args.min_trials, args.min_errors, args.seed, output)


if __name__ == "__main__":
    main()
