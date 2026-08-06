import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ldpc import BpOsdDecoder
from ldpc.mod2 import rank
from scipy.special import ndtr

sqrt_pi = np.sqrt(np.pi)

def gkp_physical_error_rate(sigma, number_of_intervals=100):
    """Marginal inner-GKP logical error = outer-code physical error rate."""
    probability = 0.0
    for m in range(number_of_intervals):
        lower = (2 * m + 0.5) * sqrt_pi / sigma
        upper = (2 * m + 1.5) * sqrt_pi / sigma
        probability += 2 * (ndtr(-lower) - ndtr(-upper))
    return float(probability)

def construct_hgp_stabilizers(H1, H2):
    """
    Build the CSS X- and Z-check matrices of a hypergraph-product code.
    """
    H1 = sp.csc_matrix(H1)
    H2 = sp.csc_matrix(H2)
    
    r1, n1 = H1.shape
    r2, n2 = H2.shape
    
    I_n1 = sp.eye(n1, format='csc')
    I_n2 = sp.eye(n2, format='csc')
    I_r1 = sp.eye(r1, format='csc')
    I_r2 = sp.eye(r2, format='csc')
    
    # HX = [ H1 tensor  I_n2 ,  I_r1 tensor H2.T ]
    HX_left = sp.kron(H1, I_n2, format='csc')
    HX_right = sp.kron(I_r1, H2.T, format='csc')
    Hx = sp.hstack([HX_left, HX_right], format='csc')
    
    # HZ = [ I_n1 tensor H2 ,  H1.T tensor I_r2 ]
    HZ_left = sp.kron(I_n1, H2, format='csc')
    HZ_right = sp.kron(H1.T, I_r2, format='csc')
    Hz = sp.hstack([HZ_left, HZ_right], format='csc')
    
    return Hx, Hz

def gkp_error_and_probability(sigma, n, cutoff=5):
    """error and probability returned"""
    displacement = np.random.normal(0, sigma, n)

    residual = (
        (displacement + sqrt_pi / 2) % sqrt_pi
    ) - sqrt_pi / 2

    error = (
        np.rint((displacement - residual) / sqrt_pi).astype(np.uint8) % 2
    )
    #gaussian prob
    k = np.arange(-cutoff, cutoff + 1)
    likelihood = np.exp(
        -(residual[:, None] + k[None, :] * sqrt_pi) ** 2
        / (2 * sigma**2)
    )

    odd = (np.abs(k) % 2) == 1
    p_error = likelihood[:, odd].sum(axis=1) / likelihood.sum(axis=1)
    #for bp
    p_error = np.clip(p_error, 1e-12, 1 - 1e-12)

    return error, p_error

def is_logical_error(residual, stabilizers, stabilizer_rank):
    if not np.any(residual):
        return False

    extended = sp.vstack([
        stabilizers,
        sp.csr_matrix(residual.reshape(1, -1))
    ])

    return rank(extended) > stabilizer_rank

import os
import json
import time

def gkp_qldpc_xz(
    Hx,
    Hz,
    sigma_vals,
    max_trials=50_000,
    min_errors=100,
    min_trials=1_000,
    save_filename="gkp_qldpc_checkpoint.json",
    standard_bernoulli=False,
):
    Hx = sp.csr_matrix(Hx, dtype=np.uint8)
    Hz = sp.csr_matrix(Hz, dtype=np.uint8)

    # Check that the CSS stabilizers commute.
    commutator = Hx @ Hz.T
    commutator.data %= 2
    commutator.eliminate_zeros()

    assert commutator.nnz == 0, "Hx and Hz do not commute"

    n = Hx.shape[1]
    rank_Hx = rank(Hx)
    rank_Hz = rank(Hz)

    z_decoder = BpOsdDecoder(
        Hx,
        error_rate=0.1,
        bp_method="product_sum",
        max_iter=n,
        schedule="serial",
        osd_method="osd_cs",
        osd_order=2,
    )
    x_decoder = BpOsdDecoder(
        Hz,
        error_rate=0.1,
        bp_method="product_sum",
        max_iter=n,
        schedule="serial",
        osd_method="osd_cs",
        osd_order=2,
    )

    code_key = f"hgp_xz_n{n}"

    # Load an existing checkpoint if one exists.
    if os.path.exists(save_filename):
        with open(save_filename, "r") as file:
            saved_data = json.load(file)

        print(f"Loaded checkpoint from {save_filename}")
    else:
        saved_data = {}

    if code_key not in saved_data:
        saved_data[code_key] = {
            "sigma": [],
            "block_rate": [], "x_rate": [], "z_rate": [],
            "block_error_bar": [], "x_error_bar": [], "z_error_bar": [],
            "total_trials": [],
            "total_block_errors": [], "total_x_errors": [], "total_z_errors": [],
        }

    code_data = saved_data[code_key]
    last_save_time = time.time()

    for sigma in sigma_vals:
        sigma_rounded = round(float(sigma), 5)
        if standard_bernoulli:
            standard_probability = gkp_physical_error_rate(sigma_rounded)
            standard_channel = np.full(n, standard_probability)

        # Resume previously saved simulations.
        if sigma_rounded in code_data["sigma"]:
            index = code_data["sigma"].index(sigma_rounded)

            block_errors = code_data["total_block_errors"][index]
            x_errors = code_data["total_x_errors"][index]
            z_errors = code_data["total_z_errors"][index]
            trials_run = code_data["total_trials"][index]

            finished = (
                trials_run >= max_trials
                or (
                    block_errors >= min_errors
                    and trials_run >= min_trials
                )
            )

            if finished:
                print(
                    f"sigma={sigma_rounded} already finished: "
                    f"{trials_run} trials, "
                    f"{block_errors} block errors"
                )
                continue
            else:
                print(
                    f"Resuming sigma={sigma_rounded} from "
                    f"{trials_run} trials..."
                )

        # Start a new simulation.
        else:
            block_errors = x_errors = z_errors = 0
            trials_run = 0

            print(f"Starting sigma={sigma_rounded}...")

        while trials_run < max_trials:
            batch_size = min(500, max_trials - trials_run)

            for _ in range(batch_size):
                if standard_bernoulli:
                    x_error = (
                        np.random.random(n) < standard_probability
                    ).astype(np.uint8)
                    z_error = (
                        np.random.random(n) < standard_probability
                    ).astype(np.uint8)
                    p_x = p_z = standard_channel
                else:
                    x_error, p_x = gkp_error_and_probability(sigma, n)
                    z_error, p_z = gkp_error_and_probability(sigma, n)

                z_syndrome = np.asarray(
                    Hx @ z_error
                ).ravel() % 2
                z_decoder.update_channel_probs(p_z)
                z_correction = z_decoder.decode(
                    z_syndrome.astype(np.uint8)
                ).astype(np.uint8)
                assert np.array_equal(
                    np.asarray(Hx @ z_correction).ravel() % 2,
                    z_syndrome,
                )
                z_failed = is_logical_error(
                    (z_error + z_correction) % 2, Hz, rank_Hz
                )

                x_syndrome = np.asarray(Hz @ x_error).ravel() % 2
                x_decoder.update_channel_probs(p_x)
                x_correction = x_decoder.decode(
                    x_syndrome.astype(np.uint8)
                ).astype(np.uint8)
                assert np.array_equal(
                    np.asarray(Hz @ x_correction).ravel() % 2,
                    x_syndrome,
                )
                x_failed = is_logical_error(
                    (x_error + x_correction) % 2, Hx, rank_Hx
                )

                x_errors += int(x_failed)
                z_errors += int(z_failed)
                block_errors += int(x_failed or z_failed)

            trials_run += batch_size

            # Stop after reaching both minimum conditions.
            if (
                block_errors >= min_errors
                and trials_run >= min_trials
            ):
                break

        block_rate = block_errors / trials_run
        x_rate = x_errors / trials_run
        z_rate = z_errors / trials_run
        def binomial_error(rate):
            return np.sqrt(rate * (1.0 - rate) / trials_run)

        if sigma_rounded in code_data["sigma"]:
            index = code_data["sigma"].index(sigma_rounded)
            code_data["block_rate"][index] = float(block_rate)
            code_data["x_rate"][index] = float(x_rate)
            code_data["z_rate"][index] = float(z_rate)
            code_data["block_error_bar"][index] = float(binomial_error(block_rate))
            code_data["x_error_bar"][index] = float(binomial_error(x_rate))
            code_data["z_error_bar"][index] = float(binomial_error(z_rate))
            code_data["total_trials"][index] = int(trials_run)
            code_data["total_block_errors"][index] = int(block_errors)
            code_data["total_x_errors"][index] = int(x_errors)
            code_data["total_z_errors"][index] = int(z_errors)
        else:
            code_data["sigma"].append(sigma_rounded)
            code_data["block_rate"].append(float(block_rate))
            code_data["x_rate"].append(float(x_rate))
            code_data["z_rate"].append(float(z_rate))
            code_data["block_error_bar"].append(float(binomial_error(block_rate)))
            code_data["x_error_bar"].append(float(binomial_error(x_rate)))
            code_data["z_error_bar"].append(float(binomial_error(z_rate)))
            code_data["total_trials"].append(int(trials_run))
            code_data["total_block_errors"].append(int(block_errors))
            code_data["total_x_errors"].append(int(x_errors))
            code_data["total_z_errors"].append(int(z_errors))

        current_time = time.time()
        if current_time - last_save_time > 300 or sigma == sigma_vals[-1]:
            with open(save_filename, "w") as file:
                json.dump(saved_data, file, indent=4)
            print(
                f"--> Saved progress checkpoint to {save_filename} "
                f"(Trials: {trials_run}, Block errors: {block_errors})"
            )
            last_save_time = current_time

    sorted_indices = np.argsort(code_data["sigma"])

    rates_sigma = np.asarray([
        code_data["sigma"][i]
        for i in sorted_indices
    ])

    fields = ("block_rate", "x_rate", "z_rate", "block_error_bar",
              "x_error_bar", "z_error_bar")
    arrays = [np.asarray([code_data[field][i] for i in sorted_indices])
              for field in fields]
    return (rates_sigma, *arrays)

# Parity-check matrix of the classical MKMN [16, 4, 6] seed code.
# Source: quantumgizmos/bp_osd examples/codes/classical_seed_codes.
H_mkmn_16_4_6 = sp.csr_matrix([
[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
[1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
], dtype=np.uint8)

def seed_matrix(text):
    """Convert an embedded whitespace-separated parity-check matrix to CSR."""
    return sp.csr_matrix([
        [int(value) for value in row.split()]
        for row in text.strip().splitlines()
    ], dtype=np.uint8)


H_mkmn_20_5_8 = seed_matrix("""
0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1
0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 1 0 0 0
0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0
0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 1 0
0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0
0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 1 0 0
0 1 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 1 0 0 0 1 0 1 0 0 0 0 0
0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 1 0 0 0
0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0
1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1
1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0
1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0
0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 1 0 0
""")

H_mkmn_24_6_10 = seed_matrix("""
0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0
0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0
0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 1 0 0 0 0
0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0
0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 1 0 0 0 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0
0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0
1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0
0 0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 1 0 0
0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0
""")

SEEDS = {
    "16_4_6": (H_mkmn_16_4_6, 16, 4, 6),
    "20_5_8": (H_mkmn_20_5_8, 20, 5, 8),
    "24_6_10": (H_mkmn_24_6_10, 24, 6, 10),
}

parser = argparse.ArgumentParser(description="Run one MKMN self-HGP simulation")
parser.add_argument("--seed", choices=SEEDS, default="16_4_6")
parser.add_argument(
    "--quick",
    action="store_true",
    help="Use a nine-point exploratory sigma grid and at most 100,000 trials",
)
parser.add_argument(
    "--standard",
    action="store_true",
    help=(
        "Use independent Bernoulli Pauli errors without GKP analogue "
        "information; combine with --quick for the matching nine-point run"
    ),
)
parser.add_argument(
    "--focused",
    action="store_true",
    help=(
        "Use a dense finite-size-crossing grid and save to separate focused "
        "result files"
    ),
)
args = parser.parse_args()

if args.quick and args.focused:
    parser.error("--quick and --focused select different grids; use only one")

H_seed, classical_n, classical_k, classical_d = SEEDS[args.seed]
H1 = H_seed
H2 = H_seed

Hx, Hz = construct_hgp_stabilizers(H1, H2)

# Guard against accidentally running a different seed or malformed HGP code.
seed_rank = rank(H_seed)
classical_r = classical_n - classical_k
assert H_seed.shape == (classical_r, classical_n)
assert seed_rank == classical_r
expected_quantum_n = classical_n**2 + classical_r**2
expected_quantum_k = classical_k**2
assert Hx.shape == (classical_r * classical_n, expected_quantum_n)
assert Hz.shape == (classical_r * classical_n, expected_quantum_n)

commutator = Hx @ Hz.T
commutator.data %= 2
commutator.eliminate_zeros()
assert commutator.nnz == 0

quantum_k = Hx.shape[1] - rank(Hx) - rank(Hz)
assert quantum_k == expected_quantum_k
print(
    f"Using MKMN [{classical_n},{classical_k},{classical_d}] self-HGP code: "
    f"n={Hx.shape[1]}, k={quantum_k}, "
    f"Hx={Hx.shape}, Hz={Hz.shape}"
)

if args.focused:
    if args.standard:
        # Equivalent p_GKP = 0.0392 ... 0.0883, surrounding the observed
        # standard finite-size crossing near p = 0.065--0.070.
        sigma_vals = np.round(np.arange(0.43, 0.521, 0.01), 2)
        result_tag = "standard_focused_"
    else:
        # p_GKP = 0.0489 ... 0.1200, surrounding the observed soft-GKP
        # finite-size crossing near sigma = 0.54--0.55.
        sigma_vals = np.round(np.arange(0.45, 0.571, 0.01), 2)
        result_tag = "focused_"
    max_trials = 1_000_000
    min_errors = 500
    min_trials = 20_000
elif args.quick:
    # Rounded construction includes both endpoints without floating-point drift.
    sigma_vals = np.round(np.arange(0.30, 0.701, 0.05), 2)
    max_trials = 100_000
    min_errors = 200
    min_trials = 10_000
    result_tag = "standard_quick_" if args.standard else "quick_"
elif args.standard:
    sigma_vals = np.round(np.arange(0.20, 0.50, 0.01), 2)
    max_trials = 1_000_000
    min_errors = 200
    min_trials = 10_000
    result_tag = "standard_"
else:
    sigma_vals = np.arange(0.2, 0.5, 0.01)
    max_trials = 20_000_000
    min_errors = 500
    min_trials = 20_000
    result_tag = ""

# These names are deliberately distinct from the Hamming-code results.
checkpoint_file = f"gkp_qldpc_mkmn_xz_{result_tag}{args.seed}_results.json"
plot_file = f"GKP_qldpc_mkmn_xz_{result_tag}{args.seed}_plot.png"

(rates_sigma, block_rates, x_rates, z_rates,
 block_errs, x_errs, z_errs) = gkp_qldpc_xz(
    Hx=Hx,
    Hz=Hz,
    sigma_vals=sigma_vals,
    max_trials=max_trials,
    min_errors=min_errors,
    min_trials=min_trials,
    save_filename=checkpoint_file,
    standard_bernoulli=args.standard,
)

plt.figure(figsize=(8, 6))

plt.errorbar(rates_sigma, block_rates, yerr=block_errs, fmt="o-", capsize=3,
             label="X or Z block failure")
plt.errorbar(rates_sigma, x_rates, yerr=x_errs, fmt="s--", capsize=3,
             label="Logical X failure")
plt.errorbar(rates_sigma, z_rates, yerr=z_errs, fmt="^--", capsize=3,
             label="Logical Z failure")

plt.yscale("log")
plt.xlabel(r"Sigma ($\sigma$)")
plt.ylabel("Logical error rate")
plt.title(
    f"{'Standard Bernoulli' if args.standard else 'GKP'} "
    f"MKMN [{classical_n},{classical_k},{classical_d}] "
    "Hypergraph-Product Code"
)
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.tight_layout()

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Results saved to {checkpoint_file}")
print(f"Plot saved to {plot_file}")
