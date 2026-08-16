"""Squeezed-GKP simulation of asymmetric repetition-product HGP codes."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from ldpc import BpOsdDecoder
from ldpc.mod2 import rank


SQRT_PI = math.sqrt(math.pi)
CONVENTION_TAG = "qX_pZ_v2"
FAMILY_BY_CODE = {
    **{code: "HGP of unequal repetition codes (dZ:dX=2:1)"
       for code in ("6x3", "12x6", "18x9", "24x12")},
    **{code: "HGP of unequal repetition codes (dZ:dX=5:2)"
       for code in ("5x2", "10x4", "15x6", "20x8", "25x10")},
    **{code: "HGP of unequal repetition codes (dZ:dX=3:1)"
       for code in ("6x2", "12x4", "18x6", "24x8", "30x10")},
}
# HGP(rep(d_z), rep(d_x)) has parameters [[n, 1, d_x, d_z]].
CODE_SPECS = {
    "5x3": (5, 3),
    "10x6": (10, 6),
    "15x9": (15, 9),
    "20x12": (20, 12),
    # Stronger 2:1 dZ:dX family for testing larger positive q-squeezing bias.
    "6x3": (6, 3),
    "12x6": (12, 6),
    "18x9": (18, 9),
    "24x12": (24, 12),
    # Stronger families used to test whether greater code asymmetry can make
    # a mild GKP noise bias beneficial at the scalable threshold.
    "5x2": (5, 2),
    "10x4": (10, 4),
    "15x6": (15, 6),
    "20x8": (20, 8),
    "25x10": (25, 10),
    "6x2": (6, 2),
    "12x4": (12, 4),
    "18x6": (18, 6),
    "24x8": (24, 8),
    "30x10": (30, 10),
    "rep5xhamming7": (5, 3),
}

HAMMING_7_4_3 = sp.csr_matrix([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)


def repetition_check(length):
    rows = np.repeat(np.arange(length - 1), 2)
    cols = np.column_stack((np.arange(length - 1), np.arange(1, length))).ravel()
    return sp.csr_matrix(
        (np.ones(2 * (length - 1), dtype=np.uint8), (rows, cols)),
        shape=(length - 1, length),
    )


def construct_hgp_stabilizers(h1, h2):
    h1, h2 = sp.csc_matrix(h1), sp.csc_matrix(h2)
    r1, n1 = h1.shape
    r2, n2 = h2.shape
    hx = sp.hstack((
        sp.kron(h1, sp.eye(n2, dtype=np.uint8), format="csc"),
        sp.kron(sp.eye(r1, dtype=np.uint8), h2.T, format="csc"),
    ), format="csr")
    hz = sp.hstack((
        sp.kron(sp.eye(n1, dtype=np.uint8), h2, format="csc"),
        sp.kron(h1.T, sp.eye(r2, dtype=np.uint8), format="csc"),
    ), format="csr")
    return hx.astype(np.uint8), hz.astype(np.uint8)


def gkp_faults_and_probabilities(rng, sigma, count, cutoff=7):
    displacement = rng.normal(0.0, sigma, count)
    residual = (displacement + SQRT_PI / 2) % SQRT_PI - SQRT_PI / 2
    faults = np.rint((displacement - residual) / SQRT_PI).astype(np.uint8) % 2
    cells = np.arange(-cutoff, cutoff + 1)
    likelihood = np.exp(
        -(residual[:, None] + cells[None, :] * SQRT_PI) ** 2 / (2 * sigma**2)
    )
    odd = np.abs(cells) % 2 == 1
    probabilities = likelihood[:, odd].sum(axis=1) / likelihood.sum(axis=1)
    return faults, np.clip(probabilities, 1e-12, 1 - 1e-12)


def squeezed_sigmas(sigma, squeezing_db):
    amplitude = 10 ** (squeezing_db / 20)
    # q -> X and p -> Z: positive position squeezing makes Z faults more common.
    return sigma / amplitude, sigma * amplitude


def is_logical_error(residual, stabilizers, stabilizer_rank):
    if not np.any(residual):
        return False
    extended = sp.vstack((stabilizers, sp.csr_matrix(residual.reshape(1, -1))))
    return rank(extended) > stabilizer_rank


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temporary.replace(path)


def simulate(hx, hz, family, spec, sigmas, squeezing_db, limits, seed, output):
    max_trials, min_trials, min_errors = limits
    n = hx.shape[1]
    rank_hx, rank_hz = rank(hx), rank(hz)
    rng = np.random.default_rng(seed)
    z_decoder = BpOsdDecoder(
        hx, error_rate=0.1, bp_method="product_sum", max_iter=n,
        schedule="serial", osd_method="osd_cs", osd_order=2,
    )
    x_decoder = BpOsdDecoder(
        hz, error_rate=0.1, bp_method="product_sum", max_iter=n,
        schedule="serial", osd_method="osd_cs", osd_order=2,
    )
    if output.exists():
        with output.open() as file:
            data = json.load(file)
    else:
        data = {
            "family": family,
            "code": spec,
            "squeezing_db": squeezing_db,
            "noise": "independent Gaussian GKP q/p shifts with analogue BP priors",
            "measurement_noise": False,
            "quadrature_pauli_convention": "q displacement -> X; p displacement -> Z",
            "convention_tag": CONVENTION_TAG,
            "points": {},
        }
    points = data["points"]
    last_save = time.time()

    for sigma_value in sigmas:
        sigma = float(sigma_value)
        sigma_q, sigma_p = squeezed_sigmas(sigma, squeezing_db)
        key = f"{sigma:.8g}"
        previous = points.get(key, {})
        trials = int(previous.get("trials", 0))
        block_errors = int(previous.get("block_errors", 0))
        x_errors = int(previous.get("x_errors", 0))
        z_errors = int(previous.get("z_errors", 0))
        if trials >= max_trials or (trials >= min_trials and block_errors >= min_errors):
            print(f"Skipping completed sigma={sigma:g}")
            continue

        while trials < max_trials and (trials < min_trials or block_errors < min_errors):
            x_fault, p_x = gkp_faults_and_probabilities(rng, sigma_q, n)
            z_fault, p_z = gkp_faults_and_probabilities(rng, sigma_p, n)

            z_syndrome = np.asarray(hx @ z_fault).ravel() % 2
            z_decoder.update_channel_probs(p_z)
            z_correction = z_decoder.decode(z_syndrome.astype(np.uint8)).astype(np.uint8)
            z_failed = is_logical_error(z_fault ^ z_correction, hz, rank_hz)

            x_syndrome = np.asarray(hz @ x_fault).ravel() % 2
            x_decoder.update_channel_probs(p_x)
            x_correction = x_decoder.decode(x_syndrome.astype(np.uint8)).astype(np.uint8)
            x_failed = is_logical_error(x_fault ^ x_correction, hx, rank_hx)

            x_errors += int(x_failed)
            z_errors += int(z_failed)
            block_errors += int(x_failed or z_failed)
            trials += 1
            if trials % 500 == 0 and time.time() - last_save > 60:
                points[key] = {
                    "sigma": sigma, "sigma_q": sigma_q, "sigma_p": sigma_p,
                    "trials": trials, "block_errors": block_errors,
                    "x_errors": x_errors, "z_errors": z_errors,
                }
                save(output, data)
                last_save = time.time()

        def rate_and_bar(errors):
            rate = errors / trials
            return rate, math.sqrt(rate * (1 - rate) / trials)

        block_rate, block_bar = rate_and_bar(block_errors)
        x_rate, x_bar = rate_and_bar(x_errors)
        z_rate, z_bar = rate_and_bar(z_errors)
        points[key] = {
            "sigma": sigma, "sigma_q": sigma_q, "sigma_p": sigma_p,
            "trials": trials, "block_errors": block_errors,
            "x_errors": x_errors, "z_errors": z_errors,
            "block_rate": block_rate, "x_rate": x_rate, "z_rate": z_rate,
            "block_error_bar": block_bar, "x_error_bar": x_bar,
            "z_error_bar": z_bar,
        }
        save(output, data)
        print(
            f"{spec}, squeeze={squeezing_db:g} dB, sigma={sigma:g}: "
            f"block={block_rate:.5g}, X={x_rate:.5g}, Z={z_rate:.5g} "
            f"({trials} trials)", flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=CODE_SPECS, required=True)
    parser.add_argument("--squeezing-db", type=float, default=0.0)
    parser.add_argument("--sigmas", type=float, nargs="+")
    parser.add_argument("--sigma-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument("--max-trials", type=int, default=1_000_000)
    parser.add_argument("--min-trials", type=int, default=10_000)
    parser.add_argument("--min-errors", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sigmas and args.sigma_range:
        parser.error("use --sigmas or --sigma-range, not both")

    d_z, d_x = CODE_SPECS[args.code]
    if args.code == "rep5xhamming7":
        h1, h2 = repetition_check(5), HAMMING_7_4_3
        family = "HGP(repetition [5,1,5], Hamming [7,4,3])"
    else:
        h1, h2 = repetition_check(d_z), repetition_check(d_x)
        family = FAMILY_BY_CODE.get(args.code, "HGP of unequal repetition codes")
    hx, hz = construct_hgp_stabilizers(h1, h2)
    commutator = hx @ hz.T
    commutator.data %= 2
    commutator.eliminate_zeros()
    expected_n = h1.shape[1] * h2.shape[1] + h1.shape[0] * h2.shape[0]
    expected_k = ((h1.shape[1] - rank(h1)) * (h2.shape[1] - rank(h2))
                  + (h1.shape[0] - rank(h1.T)) * (h2.shape[0] - rank(h2.T)))
    quantum_k = expected_n - rank(hx) - rank(hz)
    assert (commutator.nnz == 0 and hx.shape[1] == expected_n
            and quantum_k == expected_k)
    spec = f"[[{expected_n},{quantum_k},dX={d_x},dZ={d_z}]]"
    print(f"Using {family} {spec}: Hx={hx.shape}, Hz={hz.shape}")

    sigmas = (
        np.linspace(*args.sigma_range, args.num_points)
        if args.sigma_range else np.asarray(args.sigmas or np.linspace(0.25, 0.55, args.num_points))
    )
    squeeze_tag = f"{args.squeezing_db:g}".replace("-", "m").replace(".", "p")
    output = args.output or Path(
        f"asymmetric_hgp_{CONVENTION_TAG}_{args.code}_"
        f"sq{squeeze_tag}dB_results.json"
    )
    simulate(
        hx, hz, family, spec, sigmas, args.squeezing_db,
        (args.max_trials, args.min_trials, args.min_errors), args.seed, output,
    )


if __name__ == "__main__":
    main()
