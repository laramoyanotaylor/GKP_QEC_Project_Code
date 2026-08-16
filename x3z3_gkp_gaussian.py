"""Simulate the X3Z3 Floquet memory under squeezed Gaussian GKP noise.

The detector circuits are the CC-BY-4.0 reference circuits accompanying
Setiawan and McLauchlan, doi:10.5281/zenodo.14258878. At every marked
data-noise location this program explicitly samples independent Gaussian q
and p displacements, performs ideal square-GKP correction, and injects only
the resulting GKP logical fault into an otherwise noiseless circuit.

Convention: q shifts cause X faults and p shifts cause Z faults. Positive
``squeezing_db`` reduces sigma_q and increases sigma_p while keeping their
geometric mean fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.special import ndtr


SQRT_PI = math.sqrt(math.pi)
REFERENCE_ARCHIVE = Path("Stim_circuit_and_data.zip")
REFERENCE_DOI = "10.5281/zenodo.14258878"
CHANNEL_PATTERN = re.compile(r"PAULI_CHANNEL_1\([^)]*\)")
CONVENTION_TAG = "qX_pZ_v2"


def gkp_odd_probability(sigma: float, tail_tolerance: float = 1e-15) -> float:
    """Probability that ideal square-GKP correction produces a Pauli fault."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    probability = 0.0
    interval = 0
    while True:
        lower = (2 * interval + 0.5) * SQRT_PI / sigma
        upper = (2 * interval + 1.5) * SQRT_PI / sigma
        contribution = 2.0 * (ndtr(-lower) - ndtr(-upper))
        probability += contribution
        if contribution < tail_tolerance:
            break
        interval += 1
        if interval == 100_000:
            raise RuntimeError("GKP Gaussian-tail sum did not converge")
    return float(np.clip(probability, 0.0, 0.5))


def squeezed_sigmas(sigma: float, squeezing_db: float) -> tuple[float, float]:
    """Return (sigma_q, sigma_p) for fixed geometric-mean noise sigma."""
    amplitude = 10.0 ** (squeezing_db / 20.0)
    return sigma / amplitude, sigma * amplitude


def pauli_probabilities(sigma: float, squeezing_db: float) -> dict[str, float]:
    sigma_q, sigma_p = squeezed_sigmas(sigma, squeezing_db)
    p_x_marginal = gkp_odd_probability(sigma_q)
    p_z_marginal = gkp_odd_probability(sigma_p)
    # Independent q and p shifts can jointly produce a Y fault.
    p_x = p_x_marginal * (1.0 - p_z_marginal)
    p_y = p_x_marginal * p_z_marginal
    p_z = (1.0 - p_x_marginal) * p_z_marginal
    return {
        "sigma_q": sigma_q,
        "sigma_p": sigma_p,
        "p_x_marginal": p_x_marginal,
        "p_z_marginal": p_z_marginal,
        "p_x": p_x,
        "p_y": p_y,
        "p_z": p_z,
    }


def reference_member(logical: str) -> str:
    suffix = (
        f"X3Z3_L_12_num_QEC_rounds_18_logical_op_{logical}_p_0.055_"
        "meas_flip_0.0_bias_1.0_code_capacity.stim"
    )
    return f"Stim_circuit_and_data/{suffix}"


def load_reference(archive: Path, logical: str) -> str:
    if not archive.exists():
        raise FileNotFoundError(
            f"Missing {archive}. Download the reference archive from "
            f"https://doi.org/{REFERENCE_DOI}."
        )
    with zipfile.ZipFile(archive) as bundle:
        return bundle.read(reference_member(logical)).decode("utf-8")


def make_circuit(template: str, probabilities: dict[str, float]) -> stim.Circuit:
    replacement = "PAULI_CHANNEL_1({:.17g}, {:.17g}, {:.17g})".format(
        probabilities["p_x"], probabilities["p_y"], probabilities["p_z"]
    )
    circuit_text, substitutions = CHANNEL_PATTERN.subn(replacement, template)
    if substitutions == 0:
        raise ValueError("Reference circuit contains no PAULI_CHANNEL_1 locations")
    return stim.Circuit(circuit_text)


def make_gaussian_input_circuit(
    template_circuit: stim.Circuit,
) -> tuple[stim.Circuit, np.ndarray, np.ndarray]:
    """Replace noise markers by sweep-controlled X/Z fault inputs."""
    circuit = stim.Circuit()
    x_sweep_indices = []
    z_sweep_indices = []
    next_sweep = 0
    # Flatten first so noise markers inside the published REPEAT block are
    # replaced as well; otherwise the reference Pauli noise would remain.
    for instruction in template_circuit.flattened():
        if instruction.name != "PAULI_CHANNEL_1":
            circuit.append(instruction)
            continue
        qubits = [target.value for target in instruction.targets_copy()]
        for qubit in qubits:
            circuit.append("CX", [stim.target_sweep_bit(next_sweep), qubit])
            x_sweep_indices.append(next_sweep)
            next_sweep += 1
        for qubit in qubits:
            circuit.append("CZ", [stim.target_sweep_bit(next_sweep), qubit])
            z_sweep_indices.append(next_sweep)
            next_sweep += 1
    return (
        circuit,
        np.asarray(x_sweep_indices, dtype=int),
        np.asarray(z_sweep_indices, dtype=int),
    )


def sample_failures(
    template_circuit: stim.Circuit,
    decoder_circuit: stim.Circuit,
    sigma_q: float,
    sigma_p: float,
    max_shots: int,
    min_shots: int,
    max_errors: int,
    batch_size: int,
    seed: int | None,
) -> tuple[int, int]:
    """Sample Gaussian shifts explicitly; Stim only propagates their flips."""
    dem = decoder_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    matching = pymatching.Matching.from_detector_error_model(dem)
    gaussian_circuit, x_indices, z_indices = make_gaussian_input_circuit(
        template_circuit
    )
    measurement_sampler = gaussian_circuit.compile_sampler(seed=seed)
    measurement_converter = gaussian_circuit.compile_m2d_converter()
    rng = np.random.default_rng(seed)
    shots = errors = 0
    while shots < max_shots and (shots < min_shots or errors < max_errors):
        batch = min(batch_size, max_shots - shots)
        sweep_bits = np.zeros(
            (batch, gaussian_circuit.num_sweep_bits), dtype=np.bool_
        )
        # Chunking avoids allocating multi-gigabyte float arrays at the
        # cluster default batch size. The retained sweep array is boolean.
        for indices, shift_sigma in ((x_indices, sigma_q), (z_indices, sigma_p)):
            for start in range(0, len(indices), 512):
                selected = indices[start:start + 512]
                shifts = rng.normal(0.0, shift_sigma, size=(batch, len(selected)))
                sweep_bits[:, selected] = (
                    np.rint(shifts / SQRT_PI).astype(np.int64) & 1
                ).astype(bool)
        measurements = measurement_sampler.sample(shots=batch)
        detection_events, observables = measurement_converter.convert(
            measurements=measurements,
            sweep_bits=sweep_bits,
            separate_observables=True,
        )
        predictions = matching.decode_batch(detection_events)
        errors += int(np.count_nonzero(np.any(predictions != observables, axis=1)))
        shots += batch
    return shots, errors


def load_results(path: Path) -> dict:
    if path.exists():
        with path.open() as file:
            return json.load(file)
    return {
        "model": "explicit_gaussian_shifts_with_ideal_square_gkp_correction",
        "code": "X3Z3 Floquet",
        "reference_doi": REFERENCE_DOI,
        "lattice_L": 12,
        "qec_rounds": 18,
        "measurement_noise": False,
        "fault_sampling": "external Gaussian q/p shifts",
        "stim_noise_in_sampled_circuit": False,
        "decoder_priors": "analytic probabilities from the same GKP Gaussian model",
        "quadrature_pauli_convention": "q displacement -> X; p displacement -> Z",
        "convention_tag": CONVENTION_TAG,
        "points": {},
    }


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical", choices=("X", "Z"), required=True)
    parser.add_argument("--squeezing-db", type=float, required=True)
    parser.add_argument("--sigmas", type=float, nargs="+")
    parser.add_argument("--sigma-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--num-points", type=int, default=13)
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--min-shots", type=int, default=10_000)
    parser.add_argument("--max-errors", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--reference-archive", type=Path, default=REFERENCE_ARCHIVE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.sigmas and args.sigma_range:
        parser.error("use either --sigmas or --sigma-range, not both")
    sigmas = (
        np.linspace(*args.sigma_range, args.num_points)
        if args.sigma_range
        else np.asarray(args.sigmas or np.linspace(0.25, 0.55, args.num_points))
    )
    output = args.output or Path(
        f"x3z3_gkp_{CONVENTION_TAG}_L12_{args.logical}_"
        f"sq{args.squeezing_db:+g}dB_results.json"
    )
    template = load_reference(args.reference_archive, args.logical)
    results = load_results(output)
    points = results["points"]

    for point_index, sigma_value in enumerate(sigmas):
        sigma = float(sigma_value)
        key = f"sigma={sigma:.8g},squeezing_db={args.squeezing_db:.8g}"
        if key in points:
            print(f"Skipping completed {key}", flush=True)
            continue
        probabilities = pauli_probabilities(sigma, args.squeezing_db)
        template_circuit = stim.Circuit(template)
        # This circuit is used only to assign decoder priors. Actual faults are
        # sampled as Gaussian shifts above, not by Stim noise instructions.
        decoder_circuit = make_circuit(template, probabilities)
        point_seed = None if args.seed is None else args.seed + point_index
        shots, errors = sample_failures(
            template_circuit, decoder_circuit,
            probabilities["sigma_q"], probabilities["sigma_p"],
            args.max_shots, args.min_shots, args.max_errors,
            args.batch_size, point_seed,
        )
        rate = errors / shots
        points[key] = {
            "logical": args.logical,
            "sigma": sigma,
            "squeezing_db": args.squeezing_db,
            **probabilities,
            "shots": shots,
            "errors": errors,
            "logical_error_rate": rate,
            "error_bar": math.sqrt(rate * (1.0 - rate) / shots),
        }
        save_results(output, results)
        print(
            f"{key}: pL({args.logical})={rate:.6g} ({errors}/{shots}), "
            f"pX={probabilities['p_x_marginal']:.4g}, "
            f"pZ={probabilities['p_z_marginal']:.4g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
