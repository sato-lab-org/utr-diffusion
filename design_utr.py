"""Design 50-nt UTR/CDS junctions with the released MCML model.

MRL, MFE, and CAI are exposed as design targets.  Exact nucleotide, sparse
amino-acid, and suffix-filling CDS amino-acid constraints are independent
sequence-constraint layers.  The CAI input controls codon relative
adaptiveness; achieved sequence CAI is measured and reported separately.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import subprocess
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.metrics.cai import calculate_cai, translate_codons
from src.models.mcml_config import MCML_CHECKPOINT_PATH, build_mcml_diffusion
from src.models.repaint.repaint_amino_cml import (
    RePaint_Amino_Continuous_Multi_Labels as Repaint_Amino_MCML,
)
from src.models.repaint.repaint_codon_cml import (
    RePaint_Codon_Continuous_Multi_Labels as Repaint_Codon_MCML,
)
from src.models.repaint.utils import (
    build_target_adaptiveness_probability_table,
    bulid_gt_and_mask_from_codons,
)


SEQUENCE_LENGTH = 50
DNA_BASES = frozenset("ACGT")
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class CdsAminoConstraint:
    """A complete amino-acid suffix occupying the end of the 50-nt sequence."""

    peptide: str
    start: int

    @property
    def codon_positions(self) -> tuple[int, ...]:
        return tuple(range(self.start, SEQUENCE_LENGTH, 3))

    @property
    def cai_positions(self) -> tuple[int, ...]:
        # The fixed initiating AUG is excluded, matching the manuscript.
        return self.codon_positions[1:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Design 50-nt sequences with optional MRL, MFE, and CAI targets, "
            "plus nucleotide, sparse amino-acid, or CDS amino-acid constraints."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=MCML_CHECKPOINT_PATH,
        help="Final train_mcml.py checkpoint (.pt).",
    )
    parser.add_argument(
        "--checkpoint-weights",
        choices=("auto", "ema", "model"),
        default="model",
        help=(
            "Checkpoint state to load. The default 'model' reproduces the manuscript "
            "Evaluation 3 / Benchmark 2 scripts; 'auto' prefers EMA when present."
        ),
    )
    parser.add_argument("--mrl", "--MRL", dest="mrl", type=float, help="Target MRL.")
    parser.add_argument("--mfe", "--MFE", dest="mfe", type=float, help="Target MFE.")
    parser.add_argument(
        "--cai",
        "--target-adaptiveness",
        "--target-cai",
        dest="cai",
        type=float,
        help=(
            "Target codon relative-adaptiveness alpha in (0,1]. Requires "
            "--cds-amino; achieved sequence CAI is reported after generation."
        ),
    )
    constraint_group = parser.add_mutually_exclusive_group()
    constraint_group.add_argument(
        "--nucleotide",
        nargs="+",
        metavar="POS:SEQ",
        help=(
            "Exact nucleotide subsequences at 0-based starts; sequences may have "
            "arbitrary length, for example: --nucleotide 2:A 8:AGC 20:GGACU."
        ),
    )
    constraint_group.add_argument(
        "--amino",
        nargs="+",
        metavar="POS:AA",
        help="Sparse amino-acid constraints, for example: --amino 26:M 31:D 37:L.",
    )
    constraint_group.add_argument(
        "--cds-amino",
        metavar="PEPTIDE",
        help=(
            "Complete suffix-filling CDS peptide including the initial M. Its "
            "0-based start is inferred as 50 - 3 * peptide length."
        ),
    )

    # Advanced batch interface retained because scalar --mrl/--mfe flags
    # intentionally describe one target condition per invocation.
    parser.add_argument(
        "--targets",
        nargs="+",
        metavar="MRL,MFE",
        help=(
            "Advanced batch interface for one or more joint MRL,MFE pairs. "
            "Cannot be combined with --mrl or --mfe."
        ),
    )

    # One-release compatibility with the original public constraint CLI.
    # These options are absent from --help and normalized to the new interface.
    parser.add_argument("--mode", choices=("codon", "amino"), help=argparse.SUPPRESS)
    constraint_group.add_argument(
        "--codon", nargs="+", metavar="POS:CODON", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.04,
        help="Strength of the codon-usage distance weighting (default: 0.04).",
    )
    parser.add_argument(
        "--out",
        default="outputs/design_utr.fasta",
        help="Output FASTA path; supported suffixes are .fasta, .fa, and .fna.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cond-weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda:0", help="cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacement of existing FASTA, CSV, and plot outputs.",
    )
    parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Run the bundled evaluator and create MRL/MFE and constraint plots.",
    )
    parser.add_argument(
        "--eval-dir",
        "--eval-repo",
        dest="eval_dir",
        default="evaluation",
        help="Directory containing evaluate.py (default: evaluation).",
    )
    parser.add_argument("--eval-model", default="Model/model.pt")
    return parser


def parse_targets(tokens: list[str]) -> list[list[float]]:
    """Parse advanced batch MRL,MFE target pairs."""

    targets: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for token in tokens:
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"invalid --targets item {token!r}; use MRL,MFE, for example 4,-20"
            )
        pair = [float(parts[0]), float(parts[1])]
        if not all(math.isfinite(value) for value in pair):
            raise ValueError(f"target values must be finite: {token!r}")
        key = (pair[0], pair[1])
        if key in seen:
            raise ValueError(
                f"duplicate --targets pair {token!r}; each MRL,MFE pair must be unique"
            )
        seen.add(key)
        targets.append(pair)
    return targets


def resolve_targets(args: argparse.Namespace) -> list[list[float]]:
    """Return fixed-width ``[MRL, MFE]`` rows for the two-label MCML model.

    A label omitted by the user is encoded as NaN.  The MCML diffusion derives
    a separate presence mask before replacing NaN values with zeros, so neither
    a one-column tensor nor a literal zero is a valid representation of a
    missing label.
    """

    batch_targets = getattr(args, "targets", None)
    mrl = getattr(args, "mrl", None)
    mfe = getattr(args, "mfe", None)
    cai = getattr(args, "cai", None)

    if batch_targets is not None:
        if mrl is not None or mfe is not None:
            raise ValueError("--targets cannot be combined with --mrl or --mfe")
        return parse_targets(batch_targets)

    for option, value in (("--mrl", mrl), ("--mfe", mfe)):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{option} must be finite")

    if mrl is None and mfe is None and cai is None:
        raise ValueError("specify at least one design target: --mrl, --mfe, or --cai")
    return [[
        float(mrl) if mrl is not None else float("nan"),
        float(mfe) if mfe is not None else float("nan"),
    ]]


def parse_index_value_pairs(
    items: list[str], value_name: str
) -> tuple[list[int], list[str]]:
    positions: list[int] = []
    values: list[str] = []
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"invalid --{value_name} item {item!r}; use pos:value, for example 2:AGC"
            )
        raw_position, raw_value = item.split(":", 1)
        position = int(raw_position)
        value = raw_value.strip().upper()
        if value_name != "amino":
            raise ValueError(f"unsupported constraint type: {value_name!r}")
        if len(value) != 1 or value not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"invalid amino acid {raw_value!r}; use a standard one-letter code")
        if position < 0 or position + 3 > SEQUENCE_LENGTH:
            raise ValueError(f"codon start {position} must be between 0 and 47")
        positions.append(position)
        values.append(value)

    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError(f"--{value_name} positions must be unique and sorted")
    if any(current - previous < 3 for previous, current in zip(positions, positions[1:])):
        raise ValueError(f"--{value_name} constraints must not overlap")
    return positions, values


def parse_nucleotide_constraints(
    items: list[str], *, require_codon: bool = False
) -> tuple[list[int], list[str]]:
    """Parse non-overlapping ``POS:SEQ`` exact-nucleotide constraints."""

    positions: list[int] = []
    sequences: list[str] = []
    for item in items:
        if ":" not in item:
            raise ValueError(
                f"invalid nucleotide constraint {item!r}; use POS:SEQ, for example 8:AGC"
            )
        raw_position, raw_sequence = item.split(":", 1)
        try:
            position = int(raw_position)
        except ValueError as exc:
            raise ValueError(f"invalid nucleotide start {raw_position!r}") from exc
        sequence = raw_sequence.strip().upper().replace("U", "T")
        if not sequence or not set(sequence).issubset(DNA_BASES):
            raise ValueError(
                f"invalid nucleotide sequence {raw_sequence!r}; use only A/C/G/T/U"
            )
        if require_codon and len(sequence) != 3:
            raise ValueError(
                f"legacy --codon value {raw_sequence!r} must contain exactly three bases"
            )
        if position < 0 or position + len(sequence) > SEQUENCE_LENGTH:
            raise ValueError(
                f"nucleotide constraint {position}:{sequence} must lie within positions 0-49"
            )
        positions.append(position)
        sequences.append(sequence)

    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("--nucleotide positions must be unique and sorted")
    for previous_position, previous_sequence, current_position in zip(
        positions, sequences, positions[1:]
    ):
        if previous_position + len(previous_sequence) > current_position:
            raise ValueError("--nucleotide constraints must not overlap")
    return positions, sequences


def cds_amino_constraints(
    peptide: str, *, require_cai: bool = False
) -> CdsAminoConstraint:
    """Validate and position a complete CDS peptide at the sequence suffix."""

    normalized = peptide.strip().upper()
    if not normalized:
        raise ValueError("--cds-amino cannot be empty")
    invalid = sorted(set(normalized).difference(STANDARD_AMINO_ACIDS))
    if invalid:
        raise ValueError(f"--cds-amino contains unsupported amino acids: {', '.join(invalid)}")
    if normalized[0] != "M":
        raise ValueError("--cds-amino must begin with M so its first codon is AUG")
    if require_cai and len(normalized) < 2:
        raise ValueError("--cai requires at least one downstream amino acid after the initial M")
    if 3 * len(normalized) > SEQUENCE_LENGTH:
        raise ValueError(
            f"--cds-amino contains {len(normalized)} amino acids but at most "
            f"{SEQUENCE_LENGTH // 3} fit in a {SEQUENCE_LENGTH}-nt sequence"
        )
    start = SEQUENCE_LENGTH - 3 * len(normalized)
    constraint = CdsAminoConstraint(peptide=normalized, start=start)
    if len(constraint.codon_positions) != len(normalized):
        raise RuntimeError("internal CDS amino-acid positions do not fill the sequence suffix")
    return constraint


def _constraint_kind(args: argparse.Namespace) -> str | None:
    if getattr(args, "nucleotide", None) is not None or getattr(args, "codon", None) is not None:
        return "nucleotide"
    if getattr(args, "amino", None) is not None:
        return "amino"
    if getattr(args, "cds_amino", None) is not None:
        return "cds_amino"
    return None


def validate_arguments(
    args: argparse.Namespace, targets: list[list[float]] | None = None
) -> None:
    output_suffix = Path(args.out).suffix.lower()
    if output_suffix not in {".fasta", ".fa", ".fna"}:
        raise ValueError("--out must end in .fasta, .fa, or .fna")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not math.isfinite(args.cond_weight):
        raise ValueError("--cond-weight must be finite")
    if not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("--gamma must be finite and positive")

    if targets is None:
        targets = resolve_targets(args)
    if any(len(target) != 2 for target in targets):
        raise ValueError("MCML target rows must contain exactly [MRL, MFE]")

    cai = getattr(args, "cai", None)
    if cai is not None and (not math.isfinite(cai) or not 0 < cai <= 1):
        raise ValueError("--cai must be finite and in (0, 1]")
    if cai is not None and getattr(args, "cds_amino", None) is None:
        raise ValueError("--cai requires --cds-amino")

    if getattr(args, "nucleotide", None) is not None:
        parse_nucleotide_constraints(args.nucleotide)
    if getattr(args, "codon", None) is not None:
        warnings.warn(
            "--codon is deprecated; use --nucleotide POS:SEQ",
            FutureWarning,
            stacklevel=2,
        )
        parse_nucleotide_constraints(args.codon, require_codon=True)
    if getattr(args, "amino", None) is not None:
        parse_index_value_pairs(args.amino, "amino")
    if getattr(args, "cds_amino", None) is not None:
        cds_amino_constraints(args.cds_amino, require_cai=cai is not None)

    legacy_mode = getattr(args, "mode", None)
    if legacy_mode is not None:
        warnings.warn(
            "--mode is deprecated; constraint flags now select the sampling strategy",
            FutureWarning,
            stacklevel=2,
        )
        expected_kind = "nucleotide" if legacy_mode == "codon" else "amino"
        actual_kind = _constraint_kind(args)
        if actual_kind != expected_kind:
            expected_option = "--codon/--nucleotide" if legacy_mode == "codon" else "--amino"
            raise ValueError(f"legacy --mode {legacy_mode} requires {expected_option}")


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but CUDA is unavailable; use --device cpu"
        )
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    return checkpoint


def build_diffusion(args: argparse.Namespace, device: torch.device):
    diffusion = build_mcml_diffusion(condition_weight=args.cond_weight)
    checkpoint = _load_checkpoint(Path(args.checkpoint))

    if args.checkpoint_weights == "auto":
        ema_state = checkpoint.get("ema_model")
        state_key = (
            "ema_model"
            if isinstance(ema_state, Mapping) and len(ema_state) > 0
            else "model"
        )
    else:
        state_key = "ema_model" if args.checkpoint_weights == "ema" else "model"
    if state_key not in checkpoint:
        raise KeyError(f"checkpoint does not contain requested state {state_key!r}")
    state = checkpoint[state_key]
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint state {state_key!r} is empty or is not a state dictionary")

    diffusion.load_state_dict(state, strict=True)
    diffusion = diffusion.to(device)
    diffusion.eval()
    epoch = checkpoint.get("epoch", "unknown")
    print(f"[model] loaded {state_key} weights from epoch {epoch}: {args.checkpoint}")
    return diffusion


def decode_samples(
    samples: Any,
    targets: list[list[float]],
    batch_size: int,
    cai: float | None = None,
) -> list[dict[str, Any]]:
    if isinstance(samples, dict):
        samples = samples.get("samples")
        if isinstance(samples, list):
            samples = samples[-1]
    if isinstance(samples, list):
        samples = samples[-1]
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if samples is None or getattr(samples, "ndim", None) != 4:
        raise ValueError("sampler result must have shape [batch, 1, 4, 50]")
    expected_count = len(targets) * batch_size
    if tuple(samples.shape[1:]) != (1, 4, SEQUENCE_LENGTH) or samples.shape[0] != expected_count:
        raise ValueError(
            f"unexpected sampler shape {tuple(samples.shape)}; expected "
            f"({expected_count}, 1, 4, {SEQUENCE_LENGTH})"
        )

    bases = "ACGT"
    records: list[dict[str, Any]] = []
    for target_index, (target_mrl, target_mfe) in enumerate(targets):
        for sample_index in range(batch_size):
            matrix = samples[target_index * batch_size + sample_index, 0]
            sequence = "".join(bases[index] for index in matrix.argmax(axis=0))
            target_parts: list[str] = []
            if math.isfinite(target_mrl):
                target_parts.append(f"mrl_{target_mrl}")
            if math.isfinite(target_mfe):
                target_parts.append(f"mfe_{target_mfe}")
            if cai is not None:
                target_parts.append(f"cai_{cai}")
            target_name = "_".join(target_parts) if target_parts else "no_mrl_mfe"
            record_id = f"_{target_name}_idx_{sample_index}"
            records.append(
                {
                    "ID": record_id,
                    "Sequence": sequence,
                    "target_MRL": target_mrl,
                    "target_MFE": target_mfe,
                    "sample_index": sample_index,
                }
            )
    return records


def write_fasta(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        lines.extend((f">{record['ID']}", record["Sequence"]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {output_path}")


def generate_samples(args: argparse.Namespace, diffusion, targets: list[list[float]]):
    constraint_kind = _constraint_kind(args)
    if constraint_kind is None:
        labels = torch.tensor(
            targets,
            dtype=torch.float32,
            device=diffusion.device,
        ).repeat_interleave(args.batch_size, dim=0)
        return diffusion.sample(
            classes=labels,
            shape=(len(labels), 1, 4, SEQUENCE_LENGTH),
            cond_weight=args.cond_weight,
            output_all_steps=False,
        )

    if constraint_kind == "nucleotide":
        legacy_codons = getattr(args, "codon", None)
        items = args.nucleotide if getattr(args, "nucleotide", None) is not None else legacy_codons
        positions, sequences = parse_nucleotide_constraints(
            items,
            require_codon=legacy_codons is not None,
        )
        repaint = Repaint_Codon_MCML(
            diffusion=diffusion,
            sample_bs=args.batch_size,
            seq_len=SEQUENCE_LENGTH,
            cond_weight=args.cond_weight,
            tgt_labels=targets,
            return_all=False,
        )
        ground_truth, mask = bulid_gt_and_mask_from_codons(
            codon_list=sequences,
            pos_list=positions,
            total_length=SEQUENCE_LENGTH,
        )
        return repaint.p_resample(ground_truth.to(diffusion.device), mask.to(diffusion.device))

    if constraint_kind == "amino":
        positions, amino_acids = parse_index_value_pairs(args.amino, "amino")
        repaint = Repaint_Amino_MCML(
            diffusion=diffusion,
            sample_bs=args.batch_size,
            seq_len=SEQUENCE_LENGTH,
            cond_weight=args.cond_weight,
            tgt_labels=targets,
            return_all=False,
        )
        repaint.setup(amino_list=amino_acids, pos_list=positions)
        return repaint.p_resample()

    constraint = cds_amino_constraints(
        args.cds_amino,
        require_cai=getattr(args, "cai", None) is not None,
    )
    repaint_kwargs = {
        "diffusion": diffusion,
        "sample_bs": args.batch_size,
        "seq_len": SEQUENCE_LENGTH,
        "cond_weight": args.cond_weight,
        "tgt_labels": targets,
        "return_all": False,
    }
    if args.cai is not None:
        repaint_kwargs.update(
            strategy="specific_adaptiveness",
            gamma_for_usage_frequency=args.gamma,
        )
    repaint = Repaint_Amino_MCML(**repaint_kwargs)
    repaint.setup(
        amino_list=list(constraint.peptide),
        pos_list=list(constraint.codon_positions),
        adaptiveness=args.cai,
    )
    return repaint.p_resample()


def _derived_output_path(fasta_path: Path, suffix: str) -> Path:
    return fasta_path.with_name(f"{fasta_path.stem}{suffix}")


def planned_output_paths(args: argparse.Namespace) -> list[Path]:
    """Return every file the requested run may replace."""

    fasta_path = Path(args.out)
    paths = [fasta_path]
    if getattr(args, "cai", None) is not None:
        paths.extend(
            _derived_output_path(fasta_path, suffix)
            for suffix in ("_cai.csv", "_cai_summary.csv", "_cai.jpg")
        )
    if getattr(args, "do_eval", False):
        paths.extend(
            (fasta_path.with_suffix(".csv"), _derived_output_path(fasta_path, "_dist.jpg"))
        )
        if _constraint_kind(args) is not None:
            paths.append(_derived_output_path(fasta_path, "_constraint.jpg"))
    return list(dict.fromkeys(paths))


def ensure_outputs_available(args: argparse.Namespace) -> None:
    existing = [path for path in planned_output_paths(args) if path.exists()]
    if existing and not getattr(args, "force", False):
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing output(s): {formatted}; use --force to replace them"
        )


def verify_generated_constraints(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    """Fail closed if decoded sequences do not preserve requested constraints."""

    constraint_kind = _constraint_kind(args)
    if constraint_kind is None:
        return

    if constraint_kind == "nucleotide":
        legacy_codons = getattr(args, "codon", None)
        items = args.nucleotide if getattr(args, "nucleotide", None) is not None else legacy_codons
        positions, expected_values = parse_nucleotide_constraints(
            items,
            require_codon=legacy_codons is not None,
        )

        def is_valid(sequence: str) -> bool:
            normalized = sequence.upper().replace("U", "T")
            return all(
                normalized[position : position + len(expected)] == expected
                for position, expected in zip(positions, expected_values)
            )

    else:
        if constraint_kind == "amino":
            positions, expected_values = parse_index_value_pairs(args.amino, "amino")
        else:
            constraint = cds_amino_constraints(
                args.cds_amino,
                require_cai=getattr(args, "cai", None) is not None,
            )
            positions = list(constraint.codon_positions)
            expected_values = list(constraint.peptide)

        def is_valid(sequence: str) -> bool:
            try:
                return translate_codons(sequence, positions) == expected_values
            except ValueError:
                return False

    invalid_ids = [record["ID"] for record in records if not is_valid(record["Sequence"])]
    if invalid_ids:
        preview = ", ".join(invalid_ids[:5])
        suffix = "" if len(invalid_ids) <= 5 else ", ..."
        raise RuntimeError(
            f"sequence constraints were not preserved for {len(invalid_ids)}/{len(records)} "
            f"generated sequences: {preview}{suffix}"
        )


def add_cai_measurements(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], CdsAminoConstraint]:
    constraint = cds_amino_constraints(args.cds_amino, require_cai=True)
    positions = list(constraint.codon_positions)
    cai_positions = list(constraint.cai_positions)
    expected_amino_acids = list(constraint.peptide)
    expected_downstream = expected_amino_acids[1:]
    _, effective_by_amino = build_target_adaptiveness_probability_table(
        args.cai,
        return_effective_targets=True,
    )
    effective_by_position = [
        effective_by_amino[amino] for amino in expected_downstream
    ]
    effective_description = ";".join(
        f"{position}:{amino}={effective:.6f}"
        for position, amino, effective in zip(
            cai_positions,
            expected_downstream,
            effective_by_position,
        )
    )
    clipped = [
        (position, amino, effective)
        for position, amino, effective in zip(
            cai_positions,
            expected_downstream,
            effective_by_position,
        )
        if not math.isclose(effective, args.cai, abs_tol=1e-9)
    ]
    effective_geomean = math.exp(
        statistics.fmean(math.log(value) for value in effective_by_position)
    )
    for record in records:
        observed = translate_codons(record["Sequence"], positions)
        peptide_valid = observed == expected_amino_acids
        error = ""
        try:
            cai = calculate_cai(record["Sequence"], cai_positions)
        except ValueError as exc:
            cai = float("nan")
            error = str(exc)
        record.update(
            {
                "cds_amino_start_0based": constraint.start,
                "cds_amino_length": len(constraint.peptide),
                "requested_adaptiveness": args.cai,
                "effective_adaptiveness_by_codon": effective_description,
                "effective_adaptiveness_geomean_reference": effective_geomean,
                "n_effective_adaptiveness_clipped": len(clipped),
                "CAI": cai,
                "peptide_expected": "".join(expected_amino_acids),
                "peptide_observed": "".join(observed),
                "peptide_valid": peptide_valid,
                "CAI_codon_positions_0based": ",".join(map(str, cai_positions)),
                "CAI_expected_amino_acids": "".join(expected_downstream),
                "CAI_error": error,
            }
        )
    if clipped:
        details = ", ".join(
            f"position {position} ({amino}) -> {effective:.6f}"
            for position, amino, effective in clipped
        )
        print(
            f"[CAI warning] requested alpha={args.cai:.6f} is outside "
            f"the feasible synonymous-codon range at: {details}"
        )
    return records, constraint


def _summary_row(
    records: list[dict[str, Any]], scope: str, target_mrl: float | str, target_mfe: float | str
) -> dict[str, Any]:
    observed_values = [
        float(record["CAI"])
        for record in records
        if math.isfinite(float(record["CAI"]))
    ]
    values = [
        float(record["CAI"])
        for record in records
        if record["peptide_valid"] and math.isfinite(float(record["CAI"]))
    ]
    valid_count = sum(bool(record["peptide_valid"]) for record in records)
    return {
        "scope": scope,
        "target_MRL": target_mrl,
        "target_MFE": target_mfe,
        "requested_adaptiveness": records[0]["requested_adaptiveness"],
        "effective_adaptiveness_by_codon": records[0]["effective_adaptiveness_by_codon"],
        "effective_adaptiveness_geomean_reference": records[0][
            "effective_adaptiveness_geomean_reference"
        ],
        "n_effective_adaptiveness_clipped": records[0][
            "n_effective_adaptiveness_clipped"
        ],
        "n_sequences": len(records),
        "n_with_observed_CAI": len(observed_values),
        "n_with_CAI": len(values),
        "CAI_mean": statistics.fmean(values) if values else float("nan"),
        "CAI_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "CAI_min": min(values) if values else float("nan"),
        "CAI_max": max(values) if values else float("nan"),
        "peptide_valid_count": valid_count,
        "peptide_preservation_rate": valid_count / len(records),
    }


def _same_target_value(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


def write_cai_outputs(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    fasta_path: Path,
    targets: list[list[float]] | None = None,
) -> tuple[Path, Path, Path]:
    records, constraint = add_cai_measurements(records, args)
    detail_path = _derived_output_path(fasta_path, "_cai.csv")
    summary_path = _derived_output_path(fasta_path, "_cai_summary.csv")
    plot_path = _derived_output_path(fasta_path, "_cai.jpg")

    detail_fields = list(records[0])
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(records)

    summary_rows = [_summary_row(records, "overall", "all", "all")]
    if targets is None:
        targets = resolve_targets(args)
    for target_mrl, target_mfe in targets:
        selected = [
            record
            for record in records
            if _same_target_value(float(record["target_MRL"]), target_mrl)
            and _same_target_value(float(record["target_MFE"]), target_mfe)
        ]
        summary_rows.append(
            _summary_row(selected, "target_condition", target_mrl, target_mfe)
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    from src.plot.visualization import plot_cai_response

    overall = summary_rows[0]
    plot_cai_response(
        records,
        args.cai,
        str(plot_path),
        effective_reference=overall["effective_adaptiveness_geomean_reference"],
    )
    print(
        f"[CAI] target={args.cai:.4f}; "
        f"peptide-valid achieved={overall['CAI_mean']:.4f} +/- {overall['CAI_std']:.4f}; "
        f"range=[{overall['CAI_min']:.4f}, {overall['CAI_max']:.4f}]"
    )
    print(
        f"[constraint] CDS amino suffix: start={constraint.start}, "
        f"peptide={constraint.peptide}, codons={list(constraint.codon_positions)}, "
        f"preserved={overall['peptide_valid_count']}/{overall['n_sequences']} "
        f"({overall['peptide_preservation_rate']:.1%})"
    )
    print(f"[saved] {detail_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {plot_path}")
    return detail_path, summary_path, plot_path


def run_evaluator(
    fasta_path: str | Path,
    eval_dir: str | Path = "evaluation",
    eval_script: str = "evaluate.py",
    device: str = "cpu",
    model_path: str = "Model/model.pt",
    batch_toks: int = 4096 * 8,
    seed: int = 1337,
    mfe_batch: int = 100,
) -> str:
    fasta_path = Path(fasta_path).resolve()
    eval_dir = Path(eval_dir).resolve()
    eval_path = eval_dir / eval_script
    if not eval_path.is_file():
        raise FileNotFoundError(f"evaluator not found: {eval_path}")
    command = [
        sys.executable,
        eval_script,
        "--fasta",
        str(fasta_path),
        "--model",
        model_path,
        "--device",
        device,
        "--batch-toks",
        str(batch_toks),
        "--seed",
        str(seed),
        "--mfe-batch",
        str(mfe_batch),
    ]
    print("[eval] running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=eval_dir, check=True)
    output_csv = fasta_path.with_suffix(".csv")
    print(f"[eval] saved {output_csv}")
    return str(output_csv)


def design_utr(args: argparse.Namespace) -> list[dict[str, Any]]:
    targets = resolve_targets(args)
    validate_arguments(args, targets)
    ensure_outputs_available(args)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    diffusion = build_diffusion(args, device)
    samples = generate_samples(args, diffusion, targets)
    records = decode_samples(samples, targets, args.batch_size, cai=args.cai)
    verify_generated_constraints(records, args)

    output_path = Path(args.out)
    write_fasta(records, output_path)
    if args.cai is not None:
        write_cai_outputs(records, args, output_path, targets)

    if args.do_eval:
        output_csv = run_evaluator(
            fasta_path=output_path,
            eval_dir=Path(args.eval_dir),
            device=args.device,
            model_path=args.eval_model,
            seed=args.seed,
        )
        from src.plot.visualization import read_csv_and_plot

        read_csv_and_plot(str(output_csv), args)
    return records


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        design_utr(args)
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
