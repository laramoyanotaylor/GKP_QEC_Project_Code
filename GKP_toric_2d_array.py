"""One-round (2D) soft-GKP toric-code simulation for a SLURM array."""

import argparse
import json
from pathlib import Path

import numpy as np
import pymatching
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.sparse import csc_matrix, eye, hstack, kron


SQRT_PI = np.sqrt(np.pi)
# Distances used by new simulations and comparison plots.  Older d=15 and
# d=20 result files remain valid, but are intentionally not rerun by default.
DISTANCES = (3, 5, 7, 9, 11)
DEFAULT_SIGMAS = (
    0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50,
    0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70,
)


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    """Marginal GKP hard-decision error probability for a shift width."""
    probability = 0.0
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * SQRT_PI / sigma
        upper = (2 * m + 1.5) * SQRT_PI / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability)


def sigma_for_physical_error_rate(probability):
    """Invert the GKP channel, allowing grids uniform in physical probability."""
    if not 0 < probability < 0.5:
        raise ValueError("physical error probabilities must lie between 0 and 0.5")
    return brentq(lambda sigma: gkp_physical_error_rate(sigma) - probability,
                  1e-3, 5.0)


def toric_repetition_code(d):
    rows = np.repeat(np.arange(d), 2)
    cols = np.column_stack((np.arange(d), (np.arange(d) + 1) % d)).ravel()
    return csc_matrix((np.ones(2 * d, dtype=np.uint8), (rows, cols)), shape=(d, d))


def toric_x_stabilisers(d):
    repetition = toric_repetition_code(d)
    identity = eye(d, dtype=np.uint8, format="csc")
    checks = hstack((kron(repetition, identity), kron(identity, repetition)), format="csc")
    checks.data %= 2
    checks.eliminate_zeros()
    return checks


def toric_x_logicals(d):
    logicals = np.zeros((2, 2 * d * d), dtype=np.uint8)
    logicals[0, :d] = 1
    logicals[1, d * d :: d] = 1
    return logicals


def gkp_faults_and_weights(rng, sigma, count):
    displacement = rng.normal(0.0, sigma, count)
    residual = (displacement + SQRT_PI / 2) % SQRT_PI - SQRT_PI / 2
    faults = np.rint((displacement - residual) / SQRT_PI).astype(np.uint8) % 2
    weights = ((SQRT_PI - np.abs(residual)) ** 2 - residual**2) / (2 * sigma**2)
    return faults, weights


def load_result(path, distance):
    if path.exists():
        with path.open() as file:
            data = json.load(file)
    else:
        data = {}
    data.setdefault(str(distance), {})
    return data


def save_result(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temporary.replace(path)


def simulate(distance, sigmas, max_trials, min_trials, min_errors, seed, output):
    checks = toric_x_stabilisers(distance)
    logicals = toric_x_logicals(distance)
    rng = np.random.default_rng(seed)
    data = load_result(output, distance)
    distance_data = data[str(distance)]

    for sigma in sigmas:
        key = f"{sigma:.5f}"
        previous = distance_data.get(key, {})
        trials = int(previous.get("trials", 0))
        errors = int(previous.get("errors", 0))
        print(f"toric d={distance}, sigma={sigma:.5f}: starting at {trials} trials")

        while trials < max_trials and (trials < min_trials or errors < min_errors):
            fault, weights = gkp_faults_and_weights(rng, sigma, checks.shape[1])
            syndrome = np.asarray(checks @ fault).ravel() % 2  # perfect measurement
            decoder = pymatching.Matching.from_check_matrix(checks, weights=weights)
            correction = decoder.decode(syndrome).astype(np.uint8)
            residual = fault ^ correction
            errors += int(np.any((logicals @ residual) % 2))
            trials += 1

            if trials % 500 == 0:
                distance_data[key] = {"trials": trials, "errors": errors}
                save_result(output, data)

        rate = errors / trials
        error_bar = np.sqrt(rate * (1 - rate) / trials)
        distance_data[key] = {
            "sigma": sigma,
            "rate": rate,
            "error_bar": error_bar,
            "trials": trials,
            "errors": errors,
            "measurement_noise": False,
            "correction_rounds": 1,
        }
        save_result(output, data)
        print(f"toric d={distance}, sigma={sigma:.5f}: {rate:.6g} ({errors}/{trials})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, choices=DISTANCES, required=True)
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
    output = args.output or Path(f"gkp_toric_2d_d{args.distance}_results.json")
    simulate(args.distance, sigmas, args.max_trials, args.min_trials, args.min_errors, args.seed, output)


if __name__ == "__main__":
    main()
