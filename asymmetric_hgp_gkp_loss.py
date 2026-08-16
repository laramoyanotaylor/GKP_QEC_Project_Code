"""Loss-aware squeezed-GKP simulation for asymmetric repetition-product HGP codes.

The pure-loss channel is placed after squeezing.  Its deterministic attenuation
is assumed to be compensated before GKP correction, so a channel with
transmission eta adds quadrature variance (1-eta)/(2*eta), in units where the
vacuum variance is 1/2.  Setting --transmission 1 exactly recovers the Gaussian
noise model in asymmetric_hgp_gkp.py.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from ldpc import BpOsdDecoder
from ldpc.mod2 import rank

from asymmetric_hgp_gkp import (
    CODE_SPECS,
    FAMILY_BY_CODE,
    HAMMING_7_4_3,
    CONVENTION_TAG,
    construct_hgp_stabilizers,
    gkp_faults_and_probabilities,
    is_logical_error,
    repetition_check,
    save,
)


LOSS_CONVENTION_TAG = f"{CONVENTION_TAG}_pure_loss_v1"


def loss_aware_sigmas(sigma, squeezing_db, transmission, vacuum_variance=0.5):
    """Return q/p widths after squeezing, compensated pure loss, and vacuum noise."""
    amplitude = 10 ** (squeezing_db / 20)
    loss_variance = vacuum_variance * (1 - transmission) / transmission
    sigma_q = math.sqrt((sigma / amplitude) ** 2 + loss_variance)
    sigma_p = math.sqrt((sigma * amplitude) ** 2 + loss_variance)
    return sigma_q, sigma_p, loss_variance


def number_tag(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def simulate(hx, hz, family, spec, sigmas, squeezing_db, transmission,
             vacuum_variance, limits, seed, output):
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
    expected_metadata = {
        "family": family,
        "code": spec,
        "squeezing_db": squeezing_db,
        "transmission": transmission,
        "photon_loss_probability": 1 - transmission,
        "vacuum_variance": vacuum_variance,
        "loss_model": "compensated bosonic pure loss after squeezing",
        "noise": "squeezed Gaussian GKP shifts plus pure-loss vacuum noise",
        "measurement_noise": False,
        "quadrature_pauli_convention": "q displacement -> X; p displacement -> Z",
        "convention_tag": LOSS_CONVENTION_TAG,
    }
    if output.exists():
        with output.open() as file:
            data = json.load(file)
        for key in ("code", "squeezing_db", "transmission", "vacuum_variance",
                    "convention_tag"):
            if data.get(key) != expected_metadata[key]:
                raise ValueError(f"Refusing to resume {output}: {key} does not match")
    else:
        data = {**expected_metadata, "points": {}}
    points = data["points"]
    last_save = time.time()

    for sigma_value in sigmas:
        sigma = float(sigma_value)
        sigma_q, sigma_p, loss_variance = loss_aware_sigmas(
            sigma, squeezing_db, transmission, vacuum_variance
        )
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
                    "loss_variance": loss_variance, "trials": trials,
                    "block_errors": block_errors, "x_errors": x_errors,
                    "z_errors": z_errors,
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
            "loss_variance": loss_variance, "trials": trials,
            "block_errors": block_errors, "x_errors": x_errors,
            "z_errors": z_errors, "block_rate": block_rate,
            "x_rate": x_rate, "z_rate": z_rate,
            "block_error_bar": block_bar, "x_error_bar": x_bar,
            "z_error_bar": z_bar,
        }
        save(output, data)
        print(
            f"{spec}, squeeze={squeezing_db:g} dB, eta={transmission:g}, "
            f"sigma={sigma:g}: block={block_rate:.5g}, X={x_rate:.5g}, "
            f"Z={z_rate:.5g} ({trials} trials)", flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=CODE_SPECS, required=True)
    parser.add_argument("--squeezing-db", type=float, default=0.0)
    parser.add_argument(
        "--transmission", type=float, required=True,
        help="Pure-loss channel transmission eta, with 0 < eta <= 1",
    )
    parser.add_argument(
        "--vacuum-variance", type=float, default=0.5,
        help="Vacuum quadrature variance (default: 0.5 for [q,p]=i)",
    )
    parser.add_argument("--sigmas", type=float, nargs="+")
    parser.add_argument("--sigma-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument("--max-trials", type=int, default=1_000_000)
    parser.add_argument("--min-trials", type=int, default=10_000)
    parser.add_argument("--min-errors", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.transmission <= 1:
        parser.error("--transmission must satisfy 0 < eta <= 1")
    if args.vacuum_variance < 0:
        parser.error("--vacuum-variance must be non-negative")
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
    assert commutator.nnz == 0 and quantum_k == expected_k
    spec = f"[[{expected_n},{quantum_k},dX={d_x},dZ={d_z}]]"
    print(f"Using {family} {spec}: Hx={hx.shape}, Hz={hz.shape}")

    sigmas = (np.linspace(*args.sigma_range, args.num_points) if args.sigma_range
              else np.asarray(args.sigmas or np.linspace(0.25, 0.55, args.num_points)))
    output = args.output or Path(
        f"asymmetric_hgp_{LOSS_CONVENTION_TAG}_{args.code}_"
        f"sq{number_tag(args.squeezing_db)}dB_eta{number_tag(args.transmission)}_results.json"
    )
    simulate(
        hx, hz, family, spec, sigmas, args.squeezing_db, args.transmission,
        args.vacuum_variance, (args.max_trials, args.min_trials, args.min_errors),
        args.seed, output,
    )


if __name__ == "__main__":
    main()
