"""Human codon-adaptation index helpers.

The public design workflow generates a 50-nt 5' UTR--CDS junction.  CAI is
therefore evaluated only at explicitly supplied CDS codon positions; it is
never inferred from the complete 50-nt sequence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from src.models.repaint.amino_codon_table import (
    AA_TO_CODON_USAGE_HUMAN_RNA,
    get_amino_for_codon,
)


def _normalize_rna(sequence: str) -> str:
    normalized = str(sequence).strip().upper().replace("T", "U")
    if not normalized:
        raise ValueError("sequence must not be empty")
    invalid = sorted(set(normalized).difference("ACGU"))
    if invalid:
        raise ValueError(
            "sequence must contain only A/C/G/T/U; "
            f"found {', '.join(invalid)}"
        )
    return normalized


def _validate_positions(sequence: str, codon_positions: Sequence[int]) -> tuple[int, ...]:
    positions = tuple(int(position) for position in codon_positions)
    if not positions:
        raise ValueError("at least one CAI codon position is required")
    if tuple(sorted(set(positions))) != positions:
        raise ValueError("codon positions must be unique and sorted")
    for previous, current in zip(positions, positions[1:]):
        if current - previous < 3:
            raise ValueError("codon positions must not overlap")
    for position in positions:
        if position < 0 or position + 3 > len(sequence):
            raise ValueError(
                f"codon start {position} is outside a sequence of length {len(sequence)}"
            )
    return positions


def translate_codons(sequence: str, codon_positions: Sequence[int]) -> list[str]:
    """Translate codons at zero-based start positions."""

    rna = _normalize_rna(sequence)
    positions = _validate_positions(rna, codon_positions)
    amino_acids: list[str] = []
    for position in positions:
        codon = rna[position : position + 3]
        try:
            amino_acids.append(get_amino_for_codon(codon))
        except KeyError as error:
            raise ValueError(f"codon is absent from the human codon table: {codon}") from error
    return amino_acids


def calculate_cai(
    sequence: str,
    codon_positions: Sequence[int],
    expected_amino_acids: Sequence[str] | None = None,
) -> float:
    """Calculate CAI over selected codons using human synonymous-codon weights.

    This matches the manuscript benchmark convention: the fixed initiating AUG
    is excluded by the caller, while every supplied downstream codon contributes
    one relative-adaptiveness weight to the geometric mean.
    """

    rna = _normalize_rna(sequence)
    positions = _validate_positions(rna, codon_positions)

    expected: tuple[str, ...] | None = None
    if expected_amino_acids is not None:
        expected = tuple(str(amino).strip().upper() for amino in expected_amino_acids)
        if len(expected) != len(positions):
            raise ValueError(
                "expected amino-acid count must equal the number of CAI codon positions"
            )

    log_weights: list[float] = []
    for index, position in enumerate(positions):
        codon = rna[position : position + 3]
        try:
            amino_acid = get_amino_for_codon(codon)
        except KeyError as error:
            raise ValueError(f"codon is absent from the human codon table: {codon}") from error

        if amino_acid == "*":
            raise ValueError(f"stop codon is not valid in the designed peptide: {codon}")
        if expected is not None and amino_acid != expected[index]:
            raise ValueError(
                f"codon {codon} at position {position} encodes {amino_acid}, "
                f"expected {expected[index]}"
            )

        usage = AA_TO_CODON_USAGE_HUMAN_RNA[amino_acid]
        weight = float(usage[codon]) / max(float(value) for value in usage.values())
        if weight <= 0:
            raise ValueError(f"codon has non-positive relative adaptiveness: {codon}")
        log_weights.append(math.log(weight))

    return math.exp(sum(log_weights) / len(log_weights))
