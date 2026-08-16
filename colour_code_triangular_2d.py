"""GKP simulations of triangular 2D hexagonal color codes."""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from ldpc import BpOsdDecoder
from ldpc.mod2 import rank
from scipy.optimize import brentq
from scipy.special import ndtr


SQRT_PI = np.sqrt(np.pi)
DEFAULT_SIGMAS = np.round(np.arange(0.20, 0.701, 0.01), 2)


def gkp_physical_error_rate(sigma, number_of_intervals=100):
    probability = 0.0
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * SQRT_PI / sigma
        upper = (2 * m + 1.5) * SQRT_PI / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability)


def sigma_for_physical_error_rate(probability):
    if not 0 < probability < 0.5:
        raise ValueError("physical error probabilities must lie between 0 and 0.5")
    return brentq(lambda sigma: gkp_physical_error_rate(sigma) - probability,
                  1e-3, 5.0)


def gkp_error_and_probability(sigma, n, cutoff=5):
    displacement = np.random.normal(0, sigma, n)
    residual = (displacement + SQRT_PI / 2) % SQRT_PI - SQRT_PI / 2
    error = np.rint((displacement - residual) / SQRT_PI).astype(np.uint8) % 2

    k = np.arange(-cutoff, cutoff + 1)
    likelihood = np.exp(
        -(residual[:, None] + k[None, :] * SQRT_PI) ** 2 / (2 * sigma**2)
    )
    p_error = likelihood[:, np.abs(k) % 2 == 1].sum(axis=1) / likelihood.sum(axis=1)
    return error, np.clip(p_error, 1e-12, 1 - 1e-12)


def is_logical_error(residual, stabilizers, stabilizer_rank):
    """A zero-syndrome residual is logical iff outside the stabilizer rowspace."""
    if not np.any(residual):
        return False
    extended = sp.vstack([stabilizers, sp.csr_matrix(residual.reshape(1, -1))])
    return rank(extended) > stabilizer_rank


def triangular_color_code(distance):
    """Construct the face-qubit incidence matrix of a 6.6.6 color code."""
    if distance not in (3, 5, 7):
        raise ValueError("distance must be 3, 5, or 7")

    data_qubits = set()
    face_qubits = set()
    colors = ("r", "b", "g")
    x_max = distance + distance // 2

    for y, row_length in enumerate(range(x_max, 0, -1)):
        color = colors[y % 3]
        for i in range(row_length):
            coordinate = (y + 2 * i, y)
            if ((color == "r" and i % 3 in (0, 2))
                    or (color == "b" and i % 3 in (0, 1))
                    or (color == "g" and i % 3 in (1, 2))):
                data_qubits.add(coordinate)
            else:
                face_qubits.add(coordinate)

    # Sorting makes the matrix and saved results reproducible across processes.
    data_qubits = sorted(data_qubits)
    face_qubits = sorted(face_qubits)
    data_index = {coordinate: index for index, coordinate in enumerate(data_qubits)}
    h = np.zeros((len(face_qubits), len(data_qubits)), dtype=np.uint8)
    neighbor_offsets = ((-1, 1), (-2, 0), (-1, -1),
                        (1, -1), (2, 0), (1, 1))
    for face_index, (x, y) in enumerate(face_qubits):
        for dx, dy in neighbor_offsets:
            if (x + dx, y + dy) in data_index:
                h[face_index, data_index[(x + dx, y + dy)]] = 1
    return sp.csr_matrix(h)


def simulate(hx, hz, sigma_vals, max_trials, min_errors, min_trials, filename,
             code_key):
    n = hx.shape[1]
    rank_hx, rank_hz = rank(hx), rank(hz)
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
        hz, error_rate=0.1, bp_method="product_sum", max_iter=n,
        schedule="serial", osd_method="osd_cs", osd_order=2,
    )

    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = json.load(file)
        print(f"Loaded checkpoint from {filename}")
    else:
        data = {}

    code_data = data.setdefault(code_key, {
        "sigma": [], "block_rate": [], "x_rate": [], "z_rate": [],
        "block_error_bar": [], "x_error_bar": [], "z_error_bar": [],
        "total_trials": [], "total_block_errors": [],
        "total_x_errors": [], "total_z_errors": [],
    })
    last_save = time.time()

    for sigma in sigma_vals:
        sigma = round(float(sigma), 5)
        if sigma in code_data["sigma"]:
            index = code_data["sigma"].index(sigma)
            block_errors = code_data["total_block_errors"][index]
            x_errors = code_data["total_x_errors"][index]
            z_errors = code_data["total_z_errors"][index]
            trials = code_data["total_trials"][index]
            if trials >= max_trials or (block_errors >= min_errors and trials >= min_trials):
                print(f"sigma={sigma} already finished: {trials} trials, "
                      f"{block_errors} block errors")
                continue
            print(f"Resuming sigma={sigma} from {trials} trials")
        else:
            index = None
            block_errors = x_errors = z_errors = trials = 0
            print(f"Starting sigma={sigma}")

        while trials < max_trials:
            batch_size = min(500, max_trials - trials)
            for _ in range(batch_size):
                x_error, p_x = gkp_error_and_probability(sigma, n)
                z_error, p_z = gkp_error_and_probability(sigma, n)

                z_syndrome = np.asarray(hx @ z_error).ravel() % 2
                z_decoder.update_channel_probs(p_z)
                z_correction = z_decoder.decode(z_syndrome.astype(np.uint8)).astype(np.uint8)
                assert np.array_equal(np.asarray(hx @ z_correction).ravel() % 2,
                                      z_syndrome)
                z_failed = is_logical_error((z_error + z_correction) % 2,
                                            hz, rank_hz)

                x_syndrome = np.asarray(hz @ x_error).ravel() % 2
                x_decoder.update_channel_probs(p_x)
                x_correction = x_decoder.decode(x_syndrome.astype(np.uint8)).astype(np.uint8)
                assert np.array_equal(np.asarray(hz @ x_correction).ravel() % 2,
                                      x_syndrome)
                x_failed = is_logical_error((x_error + x_correction) % 2,
                                            hx, rank_hx)

                x_errors += int(x_failed)
                z_errors += int(z_failed)
                block_errors += int(x_failed or z_failed)
            trials += batch_size
            if block_errors >= min_errors and trials >= min_trials:
                break

        block_rate, x_rate, z_rate = (block_errors / trials, x_errors / trials,
                                      z_errors / trials)
        err = lambda rate: np.sqrt(rate * (1 - rate) / trials)
        values = (sigma, block_rate, x_rate, z_rate, err(block_rate),
                  err(x_rate), err(z_rate), trials, block_errors, x_errors, z_errors)
        fields = ("sigma", "block_rate", "x_rate", "z_rate", "block_error_bar",
                  "x_error_bar", "z_error_bar", "total_trials",
                  "total_block_errors", "total_x_errors", "total_z_errors")
        if index is None:
            for field, value in zip(fields, values):
                code_data[field].append(value)
        else:
            for field, value in zip(fields[1:], values[1:]):
                code_data[field][index] = value

        if time.time() - last_save > 300 or sigma == round(float(sigma_vals[-1]), 5):
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)
            last_save = time.time()

    order = np.argsort(code_data["sigma"])
    return tuple(np.asarray([code_data[field][i] for i in order])
                 for field in ("sigma", "block_rate", "x_rate", "z_rate",
                               "block_error_bar", "x_error_bar", "z_error_bar"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one triangular 2D color code")
    parser.add_argument("--distance", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--sigma-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--probability-range", type=float, nargs=2,
                        metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=25)
    parser.add_argument("--max-trials", type=int, default=20_000_000)
    parser.add_argument("--min-errors", type=int, default=500)
    parser.add_argument("--min-trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    distance = args.distance
    hx = triangular_color_code(distance)
    hz = hx.copy()  # each face supports both an X and a Z stabilizer
    expected_n = (3 * distance**2 + 1) // 4
    assert hx.shape == ((expected_n - 1) // 2, expected_n)

    commutator = hx @ hz.T
    commutator.data %= 2
    commutator.eliminate_zeros()
    assert commutator.nnz == 0
    quantum_k = hx.shape[1] - rank(hx) - rank(hz)
    assert quantum_k == 1
    print(f"Using triangular 2D color code [[{expected_n},1,{distance}]]: "
          f"Hx={hx.shape}, Hz={hz.shape}")

    if args.seed is not None:
        np.random.seed(args.seed)
    if args.probability_range:
        probabilities = np.linspace(*args.probability_range, args.num_points)
        sigma_values = [sigma_for_physical_error_rate(p) for p in probabilities]
    else:
        sigma_values = (np.linspace(*args.sigma_range, args.num_points)
                        if args.sigma_range else np.asarray(args.sigmas, dtype=float))
    checkpoint = args.output or Path(
        f"gkp_colour_code_triangular_xz_d{distance}_results.json"
    )
    plot = (checkpoint.with_suffix(".png") if args.output else Path(
        f"GKP_colour_code_triangular_xz_d{distance}_plot.png"
    ))
    (sigmas, block_rates, x_rates, z_rates,
     block_bars, x_bars, z_bars) = simulate(
        hx, hz, sigma_values,
        max_trials=args.max_trials,
        min_errors=args.min_errors,
        min_trials=args.min_trials,
        filename=str(checkpoint),
        code_key=f"triangular_color_xz_n{expected_n}_d{distance}",
    )

    plt.figure(figsize=(8, 6))
    plt.errorbar(sigmas, block_rates, yerr=block_bars, fmt="o-", capsize=3,
                 label="X or Z block failure")
    plt.errorbar(sigmas, x_rates, yerr=x_bars, fmt="s--", capsize=3,
                 label="Logical X failure")
    plt.errorbar(sigmas, z_rates, yerr=z_bars, fmt="^--", capsize=3,
                 label="Logical Z failure")
    plt.yscale("log")
    plt.xlabel(r"Sigma ($\sigma$)")
    plt.ylabel("Logical error rate")
    plt.title(f"GKP [[{expected_n},1,{distance}]] Triangular 2D Color Code")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Results saved to {checkpoint}")
    print(f"Plot saved to {plot}")
