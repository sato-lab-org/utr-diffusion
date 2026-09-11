#!/usr/bin/env python3
"""Aggregate and evaluate all Plan B + Plan C experiment-2 methods.

Plan B methods provide an already assembled 50-nt RNA:

    20-nt 5' UTR + AUG + 27-nt CDS

Plan C methods provide the two modules independently.  A UTR candidate and a
CDS candidate are paired *only* by ``(peptide_id, pairing_index)`` and are
then concatenated as ``utr20 + cds30``.  The CDS inputs already contain AUG;
the script must not add another start codon.

The pipeline validates every source before evaluation, calculates a common
human codon-usage CAI on the nine downstream codons (CDS30 excluding AUG),
predicts MRL with UTRLM, calculates ViennaRNA MFE, and writes per-method files
plus a combined table, summary, manifest, checksums, and SUCCESS marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.repaint.amino_codon_table import AA_TO_CODON_USAGE_HUMAN_RNA


EXPECTED_PEPTIDES = tuple(f"pep_{index:02d}" for index in range(1, 31))
EXPECTED_ROWS_PER_PEPTIDE = 100
EXPECTED_ROWS = 3_000
RNA_PATTERN = re.compile(r"^[ACGU]+$")
UTR_DIFFUSION_FASTA_PATTERN = re.compile(r"^(pep_\d{2})_UTR_20nt$")
UTR_DIFFUSION_RECORD_PATTERN = re.compile(
    r"^_?mrl_(?P<target_mrl>-?\d+(?:\.\d+)?)_"
    r"mfe_(?P<target_mfe>-?\d+(?:\.\d+)?)_idx_(?P<index>\d+)$"
)

MRNADESIGNER_PATH = Path(
    "/gs/bs/tga-satolab-gtex/dai/myscript/mRNAdesigner/experiments/"
    "plan_bc_utr20_cds30/web/formal_v5_100perpeptide/deliverables/"
    "mRNAdesigner_PlanBC_RNA50_3000_v5.tsv"
)
BASELINE_PANEL_ROOT = Path(
    "/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/"
    "plan_bc_common_panel_n100_seed20260804_v1"
)
COMMON_PANEL_PATH = BASELINE_PANEL_ROOT / "common_pairing_panel_n100.tsv"
GEMORNA_PATH = BASELINE_PANEL_ROOT / "gemorna_rna50_candidates_n100.tsv"
UTRGAN_PATH = BASELINE_PANEL_ROOT / "utrgan_utr20_candidates_n100.tsv"
OPTIMUS_PATH = BASELINE_PANEL_ROOT / "optimus_utr20_candidates_n100.tsv"
LINEARDESIGN_PATH = BASELINE_PANEL_ROOT / "lineardesign_cds30_candidates_n100.tsv"
LINEARCDSFOLD_PATH = BASELINE_PANEL_ROOT / "linearcdsfold_cds30_candidates_n100.tsv"
DERNA_PATH = BASELINE_PANEL_ROOT / "derna_cds30_candidates_n100.tsv"
PEPTIDE_CSV_PATH = (
    PROJECT_ROOT
    / "src/experiment/exp2_30_peptide_candidates/exp2_30_human_peptide_candidates.csv"
)
UTR_DIFFUSION_INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MRL_MFE_CAI_optimization/mrl_9_mfe_-20_cond_4_gamma_0.04"
)
DEFAULT_OUTPUT_DIR = UTR_DIFFUSION_INPUT_ROOT / "benchmark_evaluation"
DEFAULT_RNAFOLD_PATH = Path(
    "/gs/bs/tga-satolab-gtex/dai/miniconda3/envs/utr-diffusion-eval/bin/RNAfold"
)
DEFAULT_UTRLM_MODEL_PATH = Path(__file__).resolve().parent / "Model/model.pt"


CODON_TO_AA: dict[str, str] = {}
CODON_TO_RELATIVE_ADAPTIVENESS: dict[str, float] = {}
for amino_acid, usage_by_codon in AA_TO_CODON_USAGE_HUMAN_RNA.items():
    maximum = max(float(value) for value in usage_by_codon.values())
    if maximum <= 0:
        raise RuntimeError(f"invalid codon-usage maximum for {amino_acid}")
    for codon, usage in usage_by_codon.items():
        normalized_codon = codon.upper().replace("T", "U")
        CODON_TO_AA[normalized_codon] = amino_acid
        CODON_TO_RELATIVE_ADAPTIVENESS[normalized_codon] = float(usage) / maximum


def normalize_rna(value: object, label: str = "sequence") -> str:
    if pd.isna(value):
        raise ValueError(f"{label}: missing sequence")
    sequence = str(value).strip().upper().replace("T", "U")
    if not sequence or not RNA_PATTERN.fullmatch(sequence):
        raise ValueError(f"{label}: expected only A/C/G/U, got {value!r}")
    return sequence


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def translate_rna(cds: str) -> str:
    if len(cds) % 3:
        raise ValueError(f"CDS length is not divisible by three: {len(cds)}")
    amino_acids = []
    for offset in range(0, len(cds), 3):
        codon = cds[offset : offset + 3]
        if codon not in CODON_TO_AA:
            raise ValueError(f"codon is absent from the human usage table: {codon}")
        amino_acids.append(CODON_TO_AA[codon])
    return "".join(amino_acids)


def calculate_cai(cds30: str) -> float:
    """Human CAI over CDS27 only; exclude the fixed initiating AUG."""
    if len(cds30) != 30 or not cds30.startswith("AUG"):
        raise ValueError("CAI input must be CDS30 beginning with AUG")
    cds27 = cds30[3:]
    weights = []
    for offset in range(0, len(cds27), 3):
        codon = cds27[offset : offset + 3]
        if codon not in CODON_TO_RELATIVE_ADAPTIVENESS:
            raise ValueError(f"codon is absent from the human usage table: {codon}")
        weight = CODON_TO_RELATIVE_ADAPTIVENESS[codon]
        if weight <= 0:
            raise ValueError(f"codon has non-positive relative adaptiveness: {codon}")
        weights.append(weight)
    return math.exp(sum(math.log(weight) for weight in weights) / len(weights))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_keys(frame: pd.DataFrame) -> frozenset[tuple[str, int]]:
    return frozenset(
        (str(peptide_id), int(pairing_index))
        for peptide_id, pairing_index in frame[["peptide_id", "pairing_index"]].itertuples(
            index=False, name=None
        )
    )


def validate_balanced_panel(
    frame: pd.DataFrame,
    label: str,
    expected_keys: frozenset[tuple[str, int]] | None = None,
) -> None:
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"{label}: expected {EXPECTED_ROWS} rows, got {len(frame)}")
    if frame.duplicated(["peptide_id", "pairing_index"]).any():
        raise ValueError(f"{label}: duplicate (peptide_id, pairing_index) keys")
    if set(frame["peptide_id"]) != set(EXPECTED_PEPTIDES):
        raise ValueError(f"{label}: peptide IDs are not exactly pep_01..pep_30")
    for peptide_id, group in frame.groupby("peptide_id", sort=False):
        if len(group) != EXPECTED_ROWS_PER_PEPTIDE:
            raise ValueError(
                f"{label}: expected {EXPECTED_ROWS_PER_PEPTIDE} rows for "
                f"{peptide_id}, got {len(group)}"
            )
    if expected_keys is not None and frame_keys(frame) != expected_keys:
        missing = sorted(expected_keys - frame_keys(frame))[:5]
        extra = sorted(frame_keys(frame) - expected_keys)[:5]
        raise ValueError(
            f"{label}: keys differ from common Plan C panel; "
            f"missing_sample={missing}, extra_sample={extra}"
        )


def load_common_panel_keys(path: Path) -> frozenset[tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"common Plan C panel: {path}")
    panel = pd.read_csv(path, sep="\t")
    require_columns(panel, ["peptide_id", "pairing_index"], "common Plan C panel")
    panel = panel[["peptide_id", "pairing_index"]].copy()
    panel["peptide_id"] = panel["peptide_id"].astype(str)
    panel["pairing_index"] = pd.to_numeric(
        panel["pairing_index"], errors="raise"
    ).astype(int)
    validate_balanced_panel(panel, "common Plan C panel")
    pairing_sets = [
        frozenset(group["pairing_index"])
        for _, group in panel.groupby("peptide_id", sort=True)
    ]
    if len(pairing_sets) != 30 or any(values != pairing_sets[0] for values in pairing_sets[1:]):
        raise ValueError("common Plan C panel: pairing-index set is not shared by all peptides")
    print(
        f"[INPUT] common Plan C panel: rows={len(panel)}, "
        f"shared_pairing_indices={len(pairing_sets[0])}, path={path}",
        flush=True,
    )
    return frame_keys(panel)


def validate_rna_layout(frame: pd.DataFrame, label: str) -> None:
    bad_utr = frame["utr20"].str.len().ne(20) | ~frame["utr20"].str.fullmatch(
        r"[ACGU]{20}"
    )
    bad_cds = frame["cds30"].str.len().ne(30) | ~frame["cds30"].str.fullmatch(
        r"AUG[ACGU]{27}"
    )
    bad_full = frame["Sequence"].str.len().ne(50) | ~frame["Sequence"].str.fullmatch(
        r"[ACGU]{50}"
    )
    bad_join = frame["Sequence"].ne(frame["utr20"] + frame["cds30"])
    if bad_utr.any() or bad_cds.any() or bad_full.any() or bad_join.any():
        raise ValueError(
            f"{label}: invalid rows utr={int(bad_utr.sum())}, "
            f"cds={int(bad_cds.sum())}, full={int(bad_full.sum())}, "
            f"join={int(bad_join.sum())}"
        )


def validate_translation(frame: pd.DataFrame, label: str) -> None:
    if "expected_peptide_10aa" not in frame.columns:
        return
    translated = frame["cds30"].map(translate_rna)
    expected = frame["expected_peptide_10aa"].astype(str).str.upper()
    mismatch = translated.ne(expected)
    if mismatch.any():
        sample = frame.loc[mismatch, ["peptide_id", "pairing_index", "cds30"]].head()
        raise ValueError(f"{label}: {int(mismatch.sum())} translation mismatches\n{sample}")


def classify_completeness(
    frame: pd.DataFrame,
    expected_keys: frozenset[tuple[str, int]] | None = None,
) -> str:
    if len(frame) != EXPECTED_ROWS:
        return f"partial_{len(frame)}_of_{EXPECTED_ROWS}"
    try:
        validate_balanced_panel(frame, "completeness check", expected_keys=expected_keys)
    except ValueError:
        return f"noncanonical_{len(frame)}_rows"
    return "complete_30x100"


def load_direct_plan_b(
    name: str,
    input_path: Path,
    strict_complete: bool,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"{name}: {input_path}")
    source = pd.read_csv(input_path, sep="\t")
    if name == "mRNADesigner":
        pairing_column = "peptide_candidate_index"
        sequence_column = "rna50"
        peptide_column = "expected_peptide_10aa"
        require_columns(
            source,
            ["candidate_id", "peptide_id", "peptide_candidate_index", "rna50", "utr20", "cds30", "expected_peptide_10aa"],
            name,
        )
    elif name == "GEMORNA":
        pairing_column = "pairing_index" if "pairing_index" in source else "candidate_index"
        sequence_column = "raw_sequence"
        peptide_column = "optimizer_input_10aa"
        require_columns(
            source,
            ["candidate_id", "peptide_id", "candidate_index", "raw_sequence", "utr20", "cds30", "optimizer_input_10aa"],
            name,
        )
    else:
        raise ValueError(f"unsupported direct Plan B method: {name}")

    result = pd.DataFrame(
        {
            "peptide_id": source["peptide_id"],
            "pairing_index": source[pairing_column],
            "candidate_id": source["candidate_id"],
            "Sequence": source[sequence_column],
            "utr20": source["utr20"],
            "cds30": source["cds30"],
            "expected_peptide_10aa": source[peptide_column],
        }
    )
    result["pairing_index"] = result["pairing_index"].astype(int)
    for column in ["Sequence", "utr20", "cds30"]:
        result[column] = result[column].map(lambda value, c=column: normalize_rna(value, f"{name}.{c}"))
    if result.duplicated(["peptide_id", "pairing_index"]).any():
        raise ValueError(f"{name}: duplicate (peptide_id, pairing_index) keys")

    result.insert(0, "model", name)
    result.insert(1, "setting", "direct_plan_b")
    result["component"] = "Plan B direct 50-nt design"
    result["utr_model"] = name
    result["cds_model"] = name
    result["source_candidate_id"] = result.pop("candidate_id")
    result["candidate_id"] = result["source_candidate_id"]
    result["source_path"] = str(input_path.resolve())
    result["input_status"] = classify_completeness(result)
    result["CAI"] = result["cds30"].map(calculate_cai)
    validate_rna_layout(result, name)
    validate_translation(result, name)
    if strict_complete:
        validate_balanced_panel(result, name)
    print(
        f"[INPUT] {name}: rows={len(result)}, status={result['input_status'].iloc[0]}, "
        f"path={input_path}",
        flush=True,
    )
    return result


def load_utr_candidates(
    name: str,
    input_path: Path,
    expected_keys: frozenset[tuple[str, int]],
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"{name}: {input_path}")
    source = pd.read_csv(input_path, sep="\t")
    sequence_column = "terminal_20nt_utr" if name == "UTRGAN" else "terminal20_utr"
    require_columns(source, ["group_index", "pairing_index", sequence_column], name)
    result = pd.DataFrame(
        {
            "peptide_id": source["group_index"].map(lambda value: f"pep_{int(value):02d}"),
            "pairing_index": source["pairing_index"].astype(int),
            "utr20": source[sequence_column].map(
                lambda value: normalize_rna(value, f"{name}.{sequence_column}")
            ),
        }
    )
    result["utr_source_candidate_id"] = result.apply(
        lambda row: f"{name.lower()}__{row.peptide_id}__{int(row.pairing_index):04d}",
        axis=1,
    )
    if name == "UTRGAN" and "terminal_20nt_mrl" in source:
        result["utr_source_score"] = pd.to_numeric(source["terminal_20nt_mrl"])
        result["utr_source_score_type"] = "terminal20_mrl"
    elif name == "Optimus" and "optimized_predicted_mrl" in source:
        result["utr_source_score"] = pd.to_numeric(source["optimized_predicted_mrl"])
        result["utr_source_score_type"] = "optimized_full50_mrl"
    else:
        result["utr_source_score"] = float("nan")
        result["utr_source_score_type"] = "unavailable"
    result["utr_source_path"] = str(input_path.resolve())
    validate_balanced_panel(result, name, expected_keys=expected_keys)
    if not result["utr20"].str.fullmatch(r"[ACGU]{20}").all():
        raise ValueError(f"{name}: invalid 20-nt UTR")
    print(f"[INPUT] {name}: rows={len(result)}, complete 30x100 common panel", flush=True)
    return result


def load_cds_candidates(
    name: str,
    input_path: Path,
    expected_keys: frozenset[tuple[str, int]],
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(f"{name}: {input_path}")
    source = pd.read_csv(input_path, sep="\t")
    require_columns(
        source,
        ["reference_id", "pairing_index", "candidate_id", "raw_sequence", "optimizer_input_10aa"],
        name,
    )
    result = pd.DataFrame(
        {
            "peptide_id": source["reference_id"].astype(str),
            "pairing_index": source["pairing_index"].astype(int),
            "cds30": source["raw_sequence"].map(
                lambda value: normalize_rna(value, f"{name}.raw_sequence")
            ),
            "expected_peptide_10aa": source["optimizer_input_10aa"].astype(str),
            "cds_source_candidate_id": source["candidate_id"].astype(str),
        }
    )
    result["cds_source_path"] = str(input_path.resolve())
    result["tool_reported_cai"] = pd.to_numeric(
        source.get("tool_reported_cai", pd.Series(float("nan"), index=source.index)),
        errors="coerce",
    )
    result["tool_reported_mfe30"] = pd.to_numeric(
        source.get("tool_reported_mfe30", pd.Series(float("nan"), index=source.index)),
        errors="coerce",
    )
    validate_balanced_panel(result, name, expected_keys=expected_keys)
    if not result["cds30"].str.fullmatch(r"AUG[ACGU]{27}").all():
        raise ValueError(f"{name}: CDS rows are not AUG + 27 nt")
    validate_translation(result, name)
    print(f"[INPUT] {name}: rows={len(result)}, complete 30x100 common panel", flush=True)
    return result


def concatenate_utr_cds(
    utr_name: str,
    utr_frame: pd.DataFrame,
    cds_name: str,
    cds_frame: pd.DataFrame,
    expected_keys: frozenset[tuple[str, int]],
) -> pd.DataFrame:
    """Pair Plan C modules one-to-one and assemble exactly 50 nt."""
    merged = utr_frame.merge(
        cds_frame,
        on=["peptide_id", "pairing_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_ROWS:
        raise ValueError(
            f"{utr_name}+{cds_name}: merge produced {len(merged)} rows, expected {EXPECTED_ROWS}"
        )
    merged.insert(0, "model", f"{utr_name}+{cds_name}")
    merged.insert(1, "setting", "modular_plan_c")
    merged["component"] = "Plan C paired UTR20 + CDS30"
    merged["utr_model"] = utr_name
    merged["cds_model"] = cds_name
    merged["Sequence"] = merged["utr20"] + merged["cds30"]
    merged["candidate_id"] = merged.apply(
        lambda row: (
            f"{utr_name.lower()}__{cds_name.lower()}__{row.peptide_id}__"
            f"{int(row.pairing_index):04d}"
        ),
        axis=1,
    )
    merged["source_path"] = merged["utr_source_path"] + ";" + merged["cds_source_path"]
    merged["input_status"] = "complete_30x100_common_panel"
    merged["CAI"] = merged["cds30"].map(calculate_cai)
    validate_balanced_panel(
        merged, f"{utr_name}+{cds_name}", expected_keys=expected_keys
    )
    validate_rna_layout(merged, f"{utr_name}+{cds_name}")
    validate_translation(merged, f"{utr_name}+{cds_name}")
    print(f"[ASSEMBLED] {utr_name}+{cds_name}: {len(merged)} rows", flush=True)
    return merged


def load_expected_peptide_10aa(peptide_path: Path) -> dict[str, str]:
    source = pd.read_csv(peptide_path)
    require_columns(source, ["candidate_id", "downstream_peptide_16aa"], "peptide panel")
    mapping = {
        str(row.candidate_id): "M" + str(row.downstream_peptide_16aa)[:9].upper()
        for row in source.itertuples(index=False)
    }
    if set(mapping) != set(EXPECTED_PEPTIDES):
        raise ValueError("peptide panel IDs are not exactly pep_01..pep_30")
    return mapping


def evaluate_utr_diffusion(
    input_root: Path,
    peptide_path: Path = PEPTIDE_CSV_PATH,
) -> list[pd.DataFrame]:
    """Load and validate completed final UTR-Diffusion FASTAs.

    Each ``pep_XX_UTR_20nt.fasta`` record is already a complete 50-nt
    ``UTR20 + CDS30`` sequence.  No legacy leading-base correction is applied.
    MRL and MFE are evaluated later by the shared metric pipeline; CAI is
    calculated here from the encoded CDS30.
    """
    from Bio import SeqIO

    frames: list[pd.DataFrame] = []
    if not input_root.is_dir():
        print(
            f"[PENDING] UTR-Diffusion result root is not ready; skipped: {input_root}",
            flush=True,
        )
        return frames
    expected_peptides = load_expected_peptide_10aa(peptide_path)
    setting_folders = sorted(
        folder
        for folder in input_root.glob("*adaptiveness_*")
        if folder.is_dir()
    )
    for folder in setting_folders:
        fasta_paths = sorted(folder.glob("pep_*_UTR_20nt.fasta"))
        if not fasta_paths:
            print(
                f"[PENDING] UTR-Diffusion/{folder.name}: no FASTA files; skipped",
                flush=True,
            )
            continue
        observed_peptides = {}
        for fasta_path in fasta_paths:
            match = UTR_DIFFUSION_FASTA_PATTERN.fullmatch(fasta_path.stem)
            if match is None:
                raise ValueError(f"UTR-Diffusion: unexpected FASTA name: {fasta_path}")
            peptide_id = match.group(1)
            if peptide_id in observed_peptides:
                raise ValueError(f"UTR-Diffusion/{folder.name}: duplicate FASTA for {peptide_id}")
            observed_peptides[peptide_id] = fasta_path
        if set(observed_peptides) != set(EXPECTED_PEPTIDES):
            missing = sorted(set(EXPECTED_PEPTIDES) - set(observed_peptides))
            extra = sorted(set(observed_peptides) - set(EXPECTED_PEPTIDES))
            raise ValueError(
                f"UTR-Diffusion/{folder.name}: expected one FASTA for every peptide; "
                f"missing={missing}, extra={extra}"
            )

        rows = []
        for peptide_id in EXPECTED_PEPTIDES:
            fasta_path = observed_peptides[peptide_id]
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                sequence = normalize_rna(str(record.seq), str(fasta_path))
                record_match = UTR_DIFFUSION_RECORD_PATTERN.fullmatch(record.id)
                if record_match is None:
                    raise ValueError(
                        f"UTR-Diffusion/{folder.name}/{peptide_id}: "
                        f"unexpected FASTA record ID {record.id!r}"
                    )
                zero_based_index = int(record_match.group("index"))
                pairing_index = zero_based_index + 1
                rows.append(
                    {
                        "model": "UTR-Diffusion",
                        "setting": folder.name,
                        "component": "UTR-Diffusion main method",
                        "peptide_id": peptide_id,
                        "pairing_index": pairing_index,
                        "candidate_id": (
                            f"utr_diffusion__{folder.name}__{peptide_id}__"
                            f"{pairing_index:04d}"
                        ),
                        "source_record_id": record.id,
                        "Sequence": sequence,
                        "utr20": sequence[:20],
                        "cds30": sequence[20:],
                        "expected_peptide_10aa": expected_peptides[peptide_id],
                        "target_MRL": float(record_match.group("target_mrl")),
                        "target_MFE": float(record_match.group("target_mfe")),
                        "CAI": calculate_cai(sequence[20:]),
                        "utr_model": "UTR-Diffusion",
                        "cds_model": "UTR-Diffusion",
                        "source_path": str(fasta_path.resolve()),
                        "input_status": "complete_30x100",
                    }
                )
        frame = pd.DataFrame(rows)
        validate_balanced_panel(frame, f"UTR-Diffusion/{folder.name}")
        expected_pairings = set(range(1, EXPECTED_ROWS_PER_PEPTIDE + 1))
        for peptide_id, group in frame.groupby("peptide_id", sort=False):
            if set(group["pairing_index"]) != expected_pairings:
                raise ValueError(
                    f"UTR-Diffusion/{folder.name}/{peptide_id}: "
                    "record indices are not exactly 0..99"
                )
        validate_rna_layout(frame, f"UTR-Diffusion/{folder.name}")
        validate_translation(frame, f"UTR-Diffusion/{folder.name}")
        frame.sort_values(["peptide_id", "pairing_index"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        print(
            f"[INPUT] UTR-Diffusion/{folder.name}: rows={len(frame)}, "
            "complete 30x100, sequence_length=50",
            flush=True,
        )
        frames.append(frame)
    if not frames:
        print(
            f"[PENDING] no completed UTR-Diffusion FASTA collection found under {input_root}",
            flush=True,
        )
    return frames


def build_max_cai_reference(
    peptide_path: Path,
    random_reference: pd.DataFrame,
) -> pd.DataFrame:
    source = pd.read_csv(peptide_path)
    require_columns(source, ["candidate_id", "downstream_peptide_16aa"], "peptide panel")
    validate_balanced_panel(random_reference, "Random reference for Max-CAI")
    max_cai_by_peptide = {}
    for row in source.itertuples(index=False):
        downstream_9aa = row.downstream_peptide_16aa[:9]
        peptide_10aa = "M" + downstream_9aa
        cds27 = "".join(
            max(
                AA_TO_CODON_USAGE_HUMAN_RNA[amino_acid],
                key=AA_TO_CODON_USAGE_HUMAN_RNA[amino_acid].get,
            )
            for amino_acid in downstream_9aa
        )
        max_cai_by_peptide[row.candidate_id] = (peptide_10aa, "AUG" + cds27)

    rows = []
    for random_row in random_reference.itertuples(index=False):
        peptide_10aa, cds30 = max_cai_by_peptide[random_row.peptide_id]
        rows.append(
            {
                "model": "Max-CAI",
                "setting": "random_utr20_aug_max_cai_cds27",
                "component": "reference baseline",
                "peptide_id": random_row.peptide_id,
                "pairing_index": random_row.pairing_index,
                "candidate_id": (
                    f"max_cai__{random_row.peptide_id}__"
                    f"{random_row.pairing_index:04d}"
                ),
                "Sequence": random_row.utr20 + cds30,
                "utr20": random_row.utr20,
                "cds30": cds30,
                "expected_peptide_10aa": peptide_10aa,
                "CAI": calculate_cai(cds30),
                "source_path": str(peptide_path.resolve()),
                "input_status": "complete_30x100_shared_random_utr20",
            }
        )
    result = pd.DataFrame(rows)
    validate_balanced_panel(result, "Max-CAI")
    validate_rna_layout(result, "Max-CAI")
    validate_translation(result, "Max-CAI")
    random_utr20 = random_reference.set_index(["peptide_id", "pairing_index"])["utr20"]
    max_cai_utr20 = result.set_index(["peptide_id", "pairing_index"])["utr20"]
    if not max_cai_utr20.equals(random_utr20):
        raise RuntimeError("Max-CAI and Random do not share the same UTR20 panel")
    return result


def build_random_reference(count: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        peptide_index = (
            index // EXPECTED_ROWS_PER_PEPTIDE + 1 if count == EXPECTED_ROWS else 0
        )
        pairing_index = index % EXPECTED_ROWS_PER_PEPTIDE + 1
        utr20 = "".join(rng.choices("ACGU", k=20))
        cds30 = "AUG" + "".join(rng.choices("ACGU", k=27))
        rows.append(
            {
                "model": "Random",
                "setting": "random_utr20_aug_random27",
                "component": "reference baseline",
                "peptide_id": f"pep_{peptide_index:02d}" if peptide_index else "unassigned",
                "pairing_index": pairing_index,
                "candidate_id": f"random_{index + 1:05d}",
                "Sequence": utr20 + cds30,
                "utr20": utr20,
                "cds30": cds30,
                "CAI": float("nan"),
                "source_path": "generated",
                "input_status": f"generated_n{count}_seed{seed}",
            }
        )
    result = pd.DataFrame(rows)
    validate_rna_layout(result, "Random")
    return result


def evaluate_max_CAI_sequences(
    input_path: Path = PEPTIDE_CSV_PATH,
    random_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the 30x100 Plan B+C Max-CAI reference panel.

    Max-CAI and Random share UTR20 for every (peptide_id, pairing_index).
    """
    if random_reference is None:
        raise ValueError("random_reference is required for the shared UTR20 panel")
    result = build_max_cai_reference(input_path, random_reference)
    print(
        f"[REFERENCE] Max-CAI: rows={len(result)}, sequence_length=50, "
        "shared_utr20_with=Random",
        flush=True,
    )
    return result


def evaluate_random(
    count: int = EXPECTED_ROWS,
    seed: int = 1337,
) -> pd.DataFrame:
    """Create the reproducible final Plan B+C random 50-nt reference panel."""
    result = build_random_reference(count=count, seed=seed)
    if count == EXPECTED_ROWS:
        validate_balanced_panel(result, "Random")
    print(
        f"[REFERENCE] Random: rows={len(result)}, sequence_length=50, seed={seed}",
        flush=True,
    )
    return result


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def initialize_utrlm(
    model_path: Path,
    device_requested: str,
    inp_len: int,
    batch_toks: int,
    seed: int,
) -> dict[str, object]:
    """Load the same UTRLM architecture/checkpoint used by evaluation_pipeline.py."""
    import torch

    from evaluation.evaluation_pipeline import (
        Config,
        build_alphabet,
        load_model,
        set_seed,
    )

    if not model_path.is_file():
        raise FileNotFoundError(f"UTRLM checkpoint not found: {model_path}")
    cfg = Config(
        seed=seed,
        inp_len=inp_len,
        batch_toks=batch_toks,
        device=device_requested,
    )
    set_seed(cfg.seed)
    if device_requested.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_requested)
    else:
        device = torch.device("cpu")
        if device_requested.startswith("cuda"):
            print("[UTRLM] CUDA unavailable; falling back to CPU", flush=True)
    alphabet = build_alphabet()
    model = load_model(str(model_path), cfg, alphabet, device=device)
    print(
        f"[UTRLM] loaded checkpoint={model_path.resolve()}, device={device}, "
        f"inp_len={cfg.inp_len}, batch_toks={cfg.batch_toks}",
        flush=True,
    )
    return {
        "cfg": cfg,
        "alphabet": alphabet,
        "model": model,
        "device": device,
        "model_path": model_path.resolve(),
    }


def predict_frame_mrl(
    frame: pd.DataFrame,
    runtime: dict[str, object],
) -> pd.DataFrame:
    """Predict MRL in memory while preserving the exact input-row order."""
    from evaluation.evaluation_pipeline import build_dataloader, predict_mrl

    if "Sequence" not in frame.columns:
        raise KeyError("frame must contain 'Sequence' column to predict MRL")
    cfg = runtime["cfg"]
    ids = [f"row_{index:09d}" for index in range(len(frame))]
    sequences = [
        str(sequence).strip().upper().replace("U", "T")[-cfg.inp_len :]
        for sequence in frame["Sequence"]
    ]
    invalid = [
        index
        for index, sequence in enumerate(sequences)
        if not sequence or not set(sequence).issubset(set("AGCT"))
    ]
    if invalid:
        raise ValueError(f"UTRLM received invalid sequences at rows {invalid[:5]}")

    dataloader = build_dataloader(
        ids,
        sequences,
        runtime["alphabet"],
        batch_toks=cfg.batch_toks,
    )
    predictions = predict_mrl(
        dataloader,
        runtime["model"],
        device=runtime["device"],
    )
    if predictions["ID"].duplicated().any() or set(predictions["ID"]) != set(ids):
        raise RuntimeError("UTRLM predictions do not map one-to-one to input rows")
    prediction_by_id = predictions.set_index("ID")["MRL"]
    values = pd.to_numeric(pd.Series(ids).map(prediction_by_id), errors="raise")
    if values.isna().any() or not values.map(math.isfinite).all():
        raise RuntimeError("UTRLM produced missing or non-finite MRL predictions")

    result = frame.copy()
    result["MRL"] = values.to_numpy()
    print(
        f"[MRL] model={result['model'].iloc[0]}, rows={len(result)}, "
        f"mean={result['MRL'].mean():.6f}",
        flush=True,
    )
    return result


def evaluate_sequences(
    frame: pd.DataFrame,
    output_dir: Path,
    rnafold_path: str,
    batch_size: int,
    utrlm_runtime: dict[str, object] | None,
    skip_mrl: bool,
    skip_mfe: bool,
) -> tuple[pd.DataFrame, Path]:
    result = frame.copy()
    if not skip_mrl:
        if utrlm_runtime is None:
            raise RuntimeError("UTRLM runtime is required unless --skip-mrl is used")
        result = predict_frame_mrl(result, utrlm_runtime)
    if not skip_mfe:
        from evaluation.evaluation_pipeline import predict_mfe

        result = predict_mfe(result, rnafold_path=rnafold_path, batch_size=batch_size)
    result["sequence_length"] = result["Sequence"].str.len()
    model = str(result["model"].iloc[0])
    setting = str(result["setting"].iloc[0])
    suffix_parts = []
    if not skip_mrl:
        suffix_parts.append("mrl")
    if not skip_mfe:
        suffix_parts.append("mfe")
    suffix_parts.append("cai")
    suffix = "_".join(suffix_parts)
    if model == "UTR-Diffusion":
        output_path = output_dir / f"utr_diffusion_{slugify(setting)}_{suffix}.csv"
    else:
        output_path = output_dir / f"{slugify(model)}__{slugify(setting)}_{suffix}.csv"
    result.to_csv(output_path, index=False)
    print(f"[SAVED] {len(result)} sequences -> {output_path}", flush=True)
    return result, output_path


def build_summary(result: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, object]] = {
        "sequence_count": ("Sequence", "size"),
        "unique_sequence_count": ("Sequence", "nunique"),
        "sequence_length_min": ("sequence_length", "min"),
        "sequence_length_max": ("sequence_length", "max"),
        "sequence_length_mean": ("sequence_length", "mean"),
        "cai_count": ("CAI", "count"),
        "cai_mean": ("CAI", "mean"),
        "cai_std": ("CAI", "std"),
        "cai_min": ("CAI", "min"),
        "cai_max": ("CAI", "max"),
    }
    if "MRL" in result.columns:
        aggregations.update(
            {
                "mrl_count": ("MRL", "count"),
                "mrl_mean": ("MRL", "mean"),
                "mrl_std": ("MRL", "std"),
                "mrl_min": ("MRL", "min"),
                "mrl_max": ("MRL", "max"),
            }
        )
    if "MFE" in result.columns:
        aggregations.update(
            {
                "mfe_count": ("MFE", "count"),
                "mfe_mean": ("MFE", "mean"),
                "mfe_std": ("MFE", "std"),
                "mfe_min": ("MFE", "min"),
                "mfe_max": ("MFE", "max"),
            }
        )
    return (
        result.groupby(["model", "setting", "component", "input_status"], dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["model", "setting"])
        .reset_index(drop=True)
    )


def input_record(name: str, path: Path, row_count: int, status: str) -> dict[str, object]:
    return {
        "name": name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": row_count,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and evaluate Plan B + Plan C experiment-2 sequences."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rnafold-path", default=str(DEFAULT_RNAFOLD_PATH))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--random-count", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--mrnadesigner-path", type=Path, default=MRNADESIGNER_PATH)
    parser.add_argument("--common-panel-path", type=Path, default=COMMON_PANEL_PATH)
    parser.add_argument(
        "--utr-diffusion-input-root", type=Path, default=UTR_DIFFUSION_INPUT_ROOT
    )
    parser.add_argument(
        "--utrlm-model-path", type=Path, default=DEFAULT_UTRLM_MODEL_PATH
    )
    parser.add_argument("--utrlm-device", default="cuda")
    parser.add_argument("--utrlm-inp-len", type=int, default=50)
    parser.add_argument("--utrlm-batch-toks", type=int, default=4096 * 8)
    parser.add_argument("--utrlm-seed", type=int, default=1337)
    parser.add_argument("--strict-complete", action="store_true")
    parser.add_argument("--exclude-incomplete-mrnadesigner", action="store_true")
    parser.add_argument("--exclude-utr-diffusion", action="store_true")
    parser.add_argument("--exclude-reference-baselines", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-mrl", action="store_true")
    parser.add_argument("--skip-mfe", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.random_count <= 0
        or args.utrlm_inp_len <= 0
        or args.utrlm_batch_toks <= 0
    ):
        parser.error(
            "--batch-size, --random-count, --utrlm-inp-len, and "
            "--utrlm-batch-toks must be positive"
        )
    return args


def main() -> int:
    args = parse_args()

    common_panel_keys = load_common_panel_keys(args.common_panel_path)
    gemorna_frame = load_direct_plan_b("GEMORNA", GEMORNA_PATH, strict_complete=True)
    validate_balanced_panel(
        gemorna_frame, "GEMORNA common panel", expected_keys=common_panel_keys
    )
    mrnadesigner_frame = load_direct_plan_b(
        "mRNADesigner", args.mrnadesigner_path, strict_complete=args.strict_complete
    )
    direct_frames = [gemorna_frame, mrnadesigner_frame]
    if (
        args.exclude_incomplete_mrnadesigner
        and mrnadesigner_frame["input_status"].iloc[0] != "complete_30x100"
    ):
        print("[SKIP] incomplete mRNADesigner input", flush=True)
        direct_frames = direct_frames[:1]

    utr_frames = {
        "UTRGAN": load_utr_candidates("UTRGAN", UTRGAN_PATH, common_panel_keys),
        "Optimus": load_utr_candidates("Optimus", OPTIMUS_PATH, common_panel_keys),
    }
    cds_frames = {
        "LinearDesign": load_cds_candidates(
            "LinearDesign", LINEARDESIGN_PATH, common_panel_keys
        ),
        "LinearCDSFold": load_cds_candidates(
            "LinearCDSFold", LINEARCDSFOLD_PATH, common_panel_keys
        ),
        "DERNA": load_cds_candidates("DERNA", DERNA_PATH, common_panel_keys),
    }

    frames: list[pd.DataFrame] = list(direct_frames)
    for utr_name, utr_frame in utr_frames.items():
        for cds_name, cds_frame in cds_frames.items():
            frames.append(
                concatenate_utr_cds(
                    utr_name,
                    utr_frame,
                    cds_name,
                    cds_frame,
                    common_panel_keys,
                )
            )

    utr_diffusion_frames: list[pd.DataFrame] = []
    if not args.exclude_utr_diffusion:
        utr_diffusion_frames = evaluate_utr_diffusion(
            args.utr_diffusion_input_root,
            peptide_path=PEPTIDE_CSV_PATH,
        )
        frames.extend(utr_diffusion_frames)
    if not args.exclude_reference_baselines:
        random_reference = evaluate_random(args.random_count, args.random_seed)
        frames.append(
            evaluate_max_CAI_sequences(
                PEPTIDE_CSV_PATH,
                random_reference=random_reference,
            )
        )
        frames.append(random_reference)

    preflight_rows = []
    for frame in frames:
        preflight_rows.append(
            {
                "model": frame["model"].iloc[0],
                "setting": frame["setting"].iloc[0],
                "rows": len(frame),
                "unique_sequences": int(frame["Sequence"].nunique()),
                "length_min": int(frame["Sequence"].str.len().min()),
                "length_max": int(frame["Sequence"].str.len().max()),
                "input_status": frame["input_status"].iloc[0],
            }
        )
    print(json.dumps({"status": "PREFLIGHT_PASS", "methods": preflight_rows}, indent=2))
    if args.preflight_only:
        return 0

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; use --overwrite to replace named files"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    utrlm_runtime = None
    if not args.skip_mrl:
        utrlm_runtime = initialize_utrlm(
            model_path=args.utrlm_model_path,
            device_requested=args.utrlm_device,
            inp_len=args.utrlm_inp_len,
            batch_toks=args.utrlm_batch_toks,
            seed=args.utrlm_seed,
        )

    evaluated_frames: list[pd.DataFrame] = []
    written_paths: list[Path] = []
    for frame in frames:
        evaluated, output_path = evaluate_sequences(
            frame,
            output_dir=output_dir,
            rnafold_path=args.rnafold_path,
            batch_size=args.batch_size,
            utrlm_runtime=utrlm_runtime,
            skip_mrl=args.skip_mrl,
            skip_mfe=args.skip_mfe,
        )
        evaluated_frames.append(evaluated)
        written_paths.append(output_path)

    combined = pd.concat(evaluated_frames, ignore_index=True, sort=False)
    combined_metric_parts = []
    if not args.skip_mrl:
        combined_metric_parts.append("mrl")
    if not args.skip_mfe:
        combined_metric_parts.append("mfe")
    combined_metric_parts.append("cai")
    combined_path = output_dir / (
        f"all_plan_bc_methods_{'_'.join(combined_metric_parts)}.csv"
    )
    combined.to_csv(combined_path, index=False)
    written_paths.append(combined_path)

    summary = build_summary(combined)
    summary_path = output_dir / "plan_bc_method_summary.csv"
    summary.to_csv(summary_path, index=False)
    written_paths.append(summary_path)

    source_frames = {
        "GEMORNA": gemorna_frame,
        "mRNADesigner": mrnadesigner_frame,
        "UTRGAN": utr_frames["UTRGAN"],
        "Optimus": utr_frames["Optimus"],
        "LinearDesign": cds_frames["LinearDesign"],
        "LinearCDSFold": cds_frames["LinearCDSFold"],
        "DERNA": cds_frames["DERNA"],
    }
    source_paths = {
        "GEMORNA": GEMORNA_PATH,
        "mRNADesigner": args.mrnadesigner_path,
        "UTRGAN": UTRGAN_PATH,
        "Optimus": OPTIMUS_PATH,
        "LinearDesign": LINEARDESIGN_PATH,
        "LinearCDSFold": LINEARCDSFOLD_PATH,
        "DERNA": DERNA_PATH,
    }
    common_panel_source_names = {
        "GEMORNA",
        "UTRGAN",
        "Optimus",
        "LinearDesign",
        "LinearCDSFold",
        "DERNA",
    }
    input_records = []
    for name, path in source_paths.items():
        expected_keys = common_panel_keys if name in common_panel_source_names else None
        input_records.append(
            input_record(
                name,
                path,
                len(source_frames[name]),
                classify_completeness(source_frames[name], expected_keys=expected_keys),
            )
        )
    manifest = {
        "status": "PASS",
        "experiment": "EXP2 Plan B + Plan C post-processing",
        "pairing_rule": "one-to-one on peptide_id and pairing_index",
        "assembly_rule": "utr20 + cds30; cds30 already contains AUG",
        "baseline_expected_shape": "30 peptides x 100 candidates",
        "reference_baselines": {
            "random_seed": args.random_seed,
            "shared_utr20_by_peptide_id_and_pairing_index": True,
            "max_cai_construction": "random_utr20 + AUG + peptide_specific_max_cai_cds27",
            "random_construction": "random_utr20 + AUG + random_cds27",
        },
        "cai_definition": (
            "geometric mean of human relative synonymous-codon weights over "
            "CDS30[3:] (nine codons); initiating AUG excluded"
        ),
        "common_pairing_panel": {
            "path": str(args.common_panel_path.resolve()),
            "sha256": sha256_file(args.common_panel_path),
            "row_count": len(common_panel_keys),
        },
        "mrl_calculated": not args.skip_mrl,
        "utrlm": (
            {
                "model_path": str(args.utrlm_model_path.resolve()),
                "model_sha256": sha256_file(args.utrlm_model_path),
                "device_requested": args.utrlm_device,
                "device_used": str(utrlm_runtime["device"]),
                "input_length": args.utrlm_inp_len,
                "batch_toks": args.utrlm_batch_toks,
                "seed": args.utrlm_seed,
            }
            if utrlm_runtime is not None
            else None
        ),
        "mfe_calculated": not args.skip_mfe,
        "rnafold_path": args.rnafold_path,
        "cai_calculated": True,
        "utr_diffusion_input_root": str(args.utr_diffusion_input_root.resolve()),
        "utr_diffusion_settings": [
            {
                "setting": str(frame["setting"].iloc[0]),
                "row_count": len(frame),
                "peptide_count": int(frame["peptide_id"].nunique()),
                "rows_per_peptide": EXPECTED_ROWS_PER_PEPTIDE,
                "fasta_file_count": int(frame["source_path"].nunique()),
                "input_status": str(frame["input_status"].iloc[0]),
            }
            for frame in utr_diffusion_frames
        ],
        "combined_row_count": len(combined),
        "method_count": len(frames),
        "methods": preflight_rows,
        "inputs": input_records,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written_paths.append(manifest_path)

    checksums_path = output_dir / "checksums.sha256"
    with checksums_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(written_paths):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    (output_dir / "SUCCESS").write_text("status=SUCCESS\n", encoding="utf-8")
    print(f"[DONE] rows={len(combined)}, methods={len(frames)}, output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
