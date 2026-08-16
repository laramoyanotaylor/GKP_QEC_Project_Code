"""Code-capacity GKP simulation of the [[72,12,6]] bivariate bicycle code."""

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


def cyclic_shift(length, power):
    """Binary matrix for multiplication by z**power modulo z**length-1."""
    columns = np.arange(length)
    rows = (columns + power) % length
    return sp.csr_matrix(
        (np.ones(length, dtype=np.uint8), (rows, columns)),
        shape=(length, length),
    )


def bb72_checks():
    """Published (l,m)=(6,6), A=x^3+y+y^2, B=y^3+x+x^2 code."""
    ell = m = 6
    x = sp.kron(cyclic_shift(ell, 1), sp.eye(m, dtype=np.uint8), format="csr")
    y = sp.kron(sp.eye(ell, dtype=np.uint8), cyclic_shift(m, 1), format="csr")
    a = (x @ x @ x + y + y @ y).astype(np.uint8)
    b = (y @ y @ y + x + x @ x).astype(np.uint8)
    a.data %= 2
    b.data %= 2
    a.eliminate_zeros()
    b.eliminate_zeros()
    hx = sp.hstack((a, b), format="csr", dtype=np.uint8)
    hz = sp.hstack((b.T, a.T), format="csr", dtype=np.uint8)
    return hx, hz


def gkp_error_probability(sigma, cutoff=12):
    """Hard-decision logical Pauli probability after ideal square-GKP correction."""
    total = 0.0
    scale = math.sqrt(2.0) * sigma
    for cell in range(-cutoff, cutoff + 1):
        if abs(cell) % 2:
            lower = (cell * SQRT_PI - SQRT_PI / 2) / scale
            upper = (cell * SQRT_PI + SQRT_PI / 2) / scale
            total += 0.5 * (math.erf(upper) - math.erf(lower))
    return total


def sample_gkp(rng, sigma, count, analog, cutoff=7):
    displacement = rng.normal(0.0, sigma, count)
    residual = (displacement + SQRT_PI / 2) % SQRT_PI - SQRT_PI / 2
    faults = np.rint((displacement - residual) / SQRT_PI).astype(np.uint8) % 2
    if not analog:
        return faults, np.full(count, gkp_error_probability(sigma))
    cells = np.arange(-cutoff, cutoff + 1)
    likelihood = np.exp(
        -(residual[:, None] + cells[None, :] * SQRT_PI) ** 2 / (2 * sigma**2)
    )
    odd = np.abs(cells) % 2 == 1
    probabilities = likelihood[:, odd].sum(axis=1) / likelihood.sum(axis=1)
    return faults, np.clip(probabilities, 1e-12, 1 - 1e-12)


def is_logical_failure(residual, stabilizers, stabilizer_rank):
    if not np.any(residual):
        return False
    extended = sp.vstack((stabilizers, sp.csr_matrix(residual.reshape(1, -1))))
    return rank(extended) > stabilizer_rank


def atomic_save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temporary.replace(path)


def simulate(args):
    hx, hz = bb72_checks()
    commutator = hx @ hz.T
    commutator.data %= 2
    commutator.eliminate_zeros()
    rank_x, rank_z = rank(hx), rank(hz)
    n = hx.shape[1]
    k = n - rank_x - rank_z
    assert commutator.nnz == 0 and (n, k) == (72, 12)

    analog = args.decoder == "analog"
    amplitude = 10 ** (args.squeezing_db / 20)
    sigmas = np.asarray(
        args.sigmas if args.sigmas else np.linspace(*args.sigma_range, args.num_points)
    )
    squeeze_tag = f"{args.squeezing_db:g}".replace("-", "m").replace(".", "p")
    output = args.output or Path(
        f"bb72_gkp_{args.decoder}_sq{squeeze_tag}dB_results.json"
    )
    rng = np.random.default_rng(args.seed)
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
            "code": "[[72,12,6]] BB",
            "polynomials": "A=x^3+y+y^2; B=y^3+x+x^2; l=m=6",
            "decoder": args.decoder,
            "squeezing_db": args.squeezing_db,
            "noise": "independent Gaussian q/p shifts; ideal square-GKP correction",
            "measurement_noise": False,
            "quadrature_pauli_convention": "q displacement -> X; p displacement -> Z",
            "points": {},
        }

    print(f"Using [[{n},{k},6]] BB: Hx={hx.shape}, Hz={hz.shape}, output={output}")
    last_save = time.time()
    for base_sigma in sigmas:
        sigma = float(base_sigma)
        sigma_q, sigma_p = sigma / amplitude, sigma * amplitude
        key = f"{sigma:.8g}"
        previous = data["points"].get(key, {})
        trials = int(previous.get("trials", 0))
        block_errors = int(previous.get("block_errors", 0))
        x_errors = int(previous.get("x_errors", 0))
        z_errors = int(previous.get("z_errors", 0))
        complete = trials >= args.max_trials or (
            trials >= args.min_trials and block_errors >= args.min_errors
        )
        if complete:
            print(f"Skipping completed sigma={sigma:g}")
            continue

        while trials < args.max_trials and (
            trials < args.min_trials or block_errors < args.min_errors
        ):
            x_fault, p_x = sample_gkp(rng, sigma_q, n, analog)
            z_fault, p_z = sample_gkp(rng, sigma_p, n, analog)

            z_decoder.update_channel_probs(p_z)
            z_syndrome = np.asarray(hx @ z_fault).ravel().astype(np.uint8) % 2
            z_correction = z_decoder.decode(z_syndrome).astype(np.uint8)
            z_failed = is_logical_failure(z_fault ^ z_correction, hz, rank_z)

            x_decoder.update_channel_probs(p_x)
            x_syndrome = np.asarray(hz @ x_fault).ravel().astype(np.uint8) % 2
            x_correction = x_decoder.decode(x_syndrome).astype(np.uint8)
            x_failed = is_logical_failure(x_fault ^ x_correction, hx, rank_x)

            x_errors += int(x_failed)
            z_errors += int(z_failed)
            block_errors += int(x_failed or z_failed)
            trials += 1
            if trials % 500 == 0 and time.time() - last_save > 60:
                data["points"][key] = {
                    "sigma": sigma, "sigma_q": sigma_q, "sigma_p": sigma_p,
                    "physical_p_x": gkp_error_probability(sigma_q),
                    "physical_p_z": gkp_error_probability(sigma_p),
                    "trials": trials, "block_errors": block_errors,
                    "x_errors": x_errors, "z_errors": z_errors,
                }
                atomic_save(output, data)
                last_save = time.time()

        def estimate(errors):
            rate = errors / trials
            return rate, math.sqrt(rate * (1 - rate) / trials)

        block_rate, block_se = estimate(block_errors)
        x_rate, x_se = estimate(x_errors)
        z_rate, z_se = estimate(z_errors)
        data["points"][key] = {
            "sigma": sigma, "sigma_q": sigma_q, "sigma_p": sigma_p,
            "physical_p_x": gkp_error_probability(sigma_q),
            "physical_p_z": gkp_error_probability(sigma_p),
            "trials": trials, "block_errors": block_errors,
            "x_errors": x_errors, "z_errors": z_errors,
            "block_rate": block_rate, "x_rate": x_rate, "z_rate": z_rate,
            "block_standard_error": block_se,
            "x_standard_error": x_se, "z_standard_error": z_se,
        }
        atomic_save(output, data)
        print(
            f"{args.decoder}, squeeze={args.squeezing_db:g} dB, sigma={sigma:g}: "
            f"pX={data['points'][key]['physical_p_x']:.4g}, "
            f"pZ={data['points'][key]['physical_p_z']:.4g}, "
            f"block={block_rate:.5g}, X={x_rate:.5g}, Z={z_rate:.5g} "
            f"({trials} trials)", flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=("hard", "analog"), required=True)
    parser.add_argument("--squeezing-db", type=float, default=0.0)
    parser.add_argument("--sigmas", type=float, nargs="+")
    parser.add_argument("--sigma-range", type=float, nargs=2, default=(0.35, 0.65))
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument("--max-trials", type=int, default=1_000_000)
    parser.add_argument("--min-trials", type=int, default=10_000)
    parser.add_argument("--min-errors", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
