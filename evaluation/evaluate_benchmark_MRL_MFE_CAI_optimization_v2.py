#!/usr/bin/env python3
"""Evaluate full-length EXP2 sequence-generation methods.

LinearDesign and DnaChisel each provide complete 50-nt RNA sequences with the
layout ``UTR20 + AUG + CDS27``.  UTR-Diffusion FASTA records use the same
layout.  No UTR/CDS outputs from separate models are paired or concatenated.
The reproducible Random and Max-CAI reference panels are generated in memory;
each panel independently samples a random UTR20 for every sequence.

The pipeline validates every source before evaluation, calculates human
codon-usage CAI on the nine downstream codons (CDS30 excluding AUG), predicts
MRL with UTRLM, calculates ViennaRNA MFE, and writes per-method files plus a
combined table, summary, manifest, checksums, and SUCCESS marker.
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

LINEARDESIGN_PATH = Path(
    "/gs/bs/tga-satolab-gtex/dai/myscript/LinearDesign/"
    "exp2_final_v2_lineardesign_leaderopen_exports/"
    "LinearDesign_EXP2_final_v2_leaderopen_n3000_20260808/"
    "LinearDesign_EXP2_final_v2_leaderopen_candidates_n3000.tsv"
)
DNACHISEL_PATH = Path(
    "/gs/bs/tga-satolab-gtex/dai/myscript/DnaChisel/"
    "exp2_final_v2_exports/0.1-draft/"
    "dnachisel_final_v2_v1_20260816T010406Z/candidates.tsv"
)

PEPTIDE_CSV_PATH = (
    PROJECT_ROOT
    / "src/experiment/exp2_30_peptide_candidates/exp2_30_human_peptide_candidates.csv"
)
UTR_DIFFUSION_INPUT_ROOT = (
    PROJECT_ROOT
    / "outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MRL_MFE_CAI_optimization_v2/mrl_9_mfe_0_cond_4_gamma_0.04"
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


def validate_balanced_panel(frame: pd.DataFrame, label: str) -> None:
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


def load_full_length_baseline(name: str, input_path: Path) -> pd.DataFrame:
    """Load one baseline that directly generated complete 50-nt sequences."""
    if not input_path.is_file():
        raise FileNotFoundError(f"{name}: {input_path}")
    source = pd.read_csv(input_path, sep="\t")
    require_columns(
        source,
        [
            "peptide_id",
            "pairing_index",
            "candidate_id",
            "utr20",
            "cds30",
            "junction50",
            "optimizer_input_10aa",
            "generation_status",
            "parser_status",
        ],
        name,
    )

    unsuccessful = source["generation_status"].ne("success") | source[
        "parser_status"
    ].ne("success")
    if unsuccessful.any():
        raise ValueError(f"{name}: {int(unsuccessful.sum())} unsuccessful rows")

    result = pd.DataFrame(
        {
            "peptide_id": source["peptide_id"].astype(str),
            "pairing_index": pd.to_numeric(
                source["pairing_index"], errors="raise"
            ).astype(int),
            "source_candidate_id": source["candidate_id"].astype(str),
            "Sequence": source["junction50"],
            "utr20": source["utr20"],
            "cds30": source["cds30"],
            "expected_peptide_10aa": source["optimizer_input_10aa"].astype(str),
        }
    )
    for column in ["Sequence", "utr20", "cds30"]:
        result[column] = result[column].map(
            lambda value, c=column: normalize_rna(value, f"{name}.{c}")
        )

    result.insert(0, "model", name)
    result.insert(1, "setting", "full_length_baseline")
    result["component"] = "directly generated UTR20 + CDS30"
    result["utr_model"] = name
    result["cds_model"] = name
    result["candidate_id"] = result["source_candidate_id"]
    result["source_path"] = str(input_path.resolve())
    result["input_status"] = "complete_30x100"
    result["tool_reported_cai"] = pd.to_numeric(
        source.get("generation_cai9", pd.Series(float("nan"), index=source.index)),
        errors="coerce",
    )
    result["tool_reported_mfe50"] = pd.to_numeric(
        source.get("generation_mfe50", pd.Series(float("nan"), index=source.index)),
        errors="coerce",
    )
    result["CAI"] = result["cds30"].map(calculate_cai)
    validate_balanced_panel(result, name)
    validate_rna_layout(result, name)
    validate_translation(result, name)
    result.sort_values(["peptide_id", "pairing_index"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    print(
        f"[INPUT] {name}: rows={len(result)}, complete 30x100, "
        f"sequence_length=50, path={input_path}",
        flush=True,
    )
    return result


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
    utr_seed: int,
) -> pd.DataFrame:
    """Build random UTR20 + AUG + peptide-specific Max-CAI CDS27 sequences."""
    source = pd.read_csv(peptide_path)
    require_columns(source, ["candidate_id", "downstream_peptide_16aa"], "peptide panel")
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
    if set(max_cai_by_peptide) != set(EXPECTED_PEPTIDES):
        raise ValueError("peptide panel IDs are not exactly pep_01..pep_30")

    rng = random.Random(utr_seed)
    rows = []
    for peptide_id in EXPECTED_PEPTIDES:
        peptide_10aa, cds30 = max_cai_by_peptide[peptide_id]
        for pairing_index in range(1, EXPECTED_ROWS_PER_PEPTIDE + 1):
            utr20 = "".join(rng.choices("ACGU", k=20))
            rows.append(
                {
                    "model": "Max-CAI",
                    "setting": "random_utr20_aug_max_cai_cds27",
                    "component": "reference baseline",
                    "peptide_id": peptide_id,
                    "pairing_index": pairing_index,
                    "candidate_id": (
                        f"max_cai__{peptide_id}__{pairing_index:04d}"
                    ),
                    "Sequence": utr20 + cds30,
                    "utr20": utr20,
                    "cds30": cds30,
                    "expected_peptide_10aa": peptide_10aa,
                    "CAI": calculate_cai(cds30),
                    "source_path": str(peptide_path.resolve()),
                    "input_status": (
                        "complete_30x100_independent_random_utr20_"
                        f"seed{utr_seed}"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    validate_balanced_panel(result, "Max-CAI")
    validate_rna_layout(result, "Max-CAI")
    validate_translation(result, "Max-CAI")
    return result


def build_random_reference(count: int, seed: int) -> pd.DataFrame:
    """Build random UTR20 + AUG + random CDS27 reference sequences."""
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
                "peptide_id": (
                    f"pep_{peptide_index:02d}" if peptide_index else "unassigned"
                ),
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
    utr_seed: int = 1338,
) -> pd.DataFrame:
    """Create the 30x100 Max-CAI panel with independently random UTR20s."""
    result = build_max_cai_reference(input_path, utr_seed=utr_seed)
    print(
        f"[REFERENCE] Max-CAI: rows={len(result)}, sequence_length=50, "
        f"independent_utr20_seed={utr_seed}",
        flush=True,
    )
    return result


def evaluate_random(
    count: int = EXPECTED_ROWS,
    seed: int = 1337,
) -> pd.DataFrame:
    """Create the reproducible 30x100 random 50-nt reference panel."""
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
        description=(
            "Evaluate full-length EXP2 sequences from LinearDesign, DnaChisel, "
            "and UTR-Diffusion."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rnafold-path", default=str(DEFAULT_RNAFOLD_PATH))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--random-count", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--max-cai-utr-seed", type=int, default=1338)
    parser.add_argument("--lineardesign-path", type=Path, default=LINEARDESIGN_PATH)
    parser.add_argument("--dnachisel-path", type=Path, default=DNACHISEL_PATH)
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
    if (
        not args.exclude_reference_baselines
        and args.random_count != EXPECTED_ROWS
    ):
        parser.error(
            f"--random-count must be {EXPECTED_ROWS} when reference baselines "
            "are enabled"
        )
    if (
        not args.exclude_reference_baselines
        and args.max_cai_utr_seed == args.random_seed
    ):
        parser.error(
            "--max-cai-utr-seed and --random-seed must differ so the UTR20 "
            "panels use independent random streams"
        )
    return args


def main() -> int:
    args = parse_args()

    baseline_frames = [
        load_full_length_baseline("LinearDesign", args.lineardesign_path),
        load_full_length_baseline("DnaChisel", args.dnachisel_path),
    ]
    frames: list[pd.DataFrame] = list(baseline_frames)

    utr_diffusion_frames: list[pd.DataFrame] = []
    if not args.exclude_utr_diffusion:
        utr_diffusion_frames = evaluate_utr_diffusion(
            args.utr_diffusion_input_root,
            peptide_path=PEPTIDE_CSV_PATH,
        )
        frames.extend(utr_diffusion_frames)
    reference_baseline_frames: list[pd.DataFrame] = []
    if not args.exclude_reference_baselines:
        random_reference = evaluate_random(args.random_count, args.random_seed)
        reference_baseline_frames = [
            evaluate_max_CAI_sequences(
                PEPTIDE_CSV_PATH,
                utr_seed=args.max_cai_utr_seed,
            ),
            random_reference,
        ]
        frames.extend(reference_baseline_frames)

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
        f"all_full_length_methods_{'_'.join(combined_metric_parts)}.csv"
    )
    combined.to_csv(combined_path, index=False)
    written_paths.append(combined_path)

    summary = build_summary(combined)
    summary_path = output_dir / "full_length_method_summary.csv"
    summary.to_csv(summary_path, index=False)
    written_paths.append(summary_path)

    input_records = [
        input_record(
            str(frame["model"].iloc[0]),
            path,
            len(frame),
            str(frame["input_status"].iloc[0]),
        )
        for frame, path in zip(
            baseline_frames,
            [args.lineardesign_path, args.dnachisel_path],
        )
    ]
    manifest = {
        "status": "PASS",
        "experiment": "EXP2 full-length sequence benchmark",
        "baseline_models": ["LinearDesign", "DnaChisel"]
        + [str(frame["model"].iloc[0]) for frame in reference_baseline_frames],
        "baseline_generation_mode": {
            "LinearDesign": "direct full-length 50-nt generation",
            "DnaChisel": "direct full-length 50-nt generation",
            "Max-CAI": "random UTR20 + AUG + peptide-specific Max-CAI CDS27",
            "Random": "random UTR20 + AUG + random CDS27",
        },
        "assembly_rule": "use provided junction50; verify junction50 == utr20 + cds30",
        "baseline_expected_shape": "30 peptides x 100 candidates",
        "reference_baselines": {
            "enabled": not args.exclude_reference_baselines,
            "random_seed": args.random_seed,
            "max_cai_utr_seed": args.max_cai_utr_seed,
            "random_count": args.random_count,
            "shared_utr20_by_peptide_id_and_pairing_index": False,
            "independent_random_utr20_per_baseline": True,
        },
        "cai_definition": (
            "geometric mean of human relative synonymous-codon weights over "
            "CDS30[3:] (nine codons); initiating AUG excluded"
        ),
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
