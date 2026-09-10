from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("logomaker")

import src.plot.visualization as visualization  # noqa: E402
from src.plot.visualization import (  # noqa: E402
    _amino_codon_counts_with_invalid,
    _describe_conditions,
    _derived_plot_path,
    _expanded_axis_limits,
    _finite_target_pairs,
    _normalise_conditions,
    _scatter_target_pairs,
)


def test_plot_paths_are_derived_for_any_fasta_extension():
    assert _derived_plot_path("demo.fa", "_dist.jpg") == "demo_dist.jpg"
    assert _derived_plot_path("demo.FASTA", "_constraint.jpg") == "demo_constraint.jpg"


def test_amino_codon_counts_report_invalid_sequences_without_pseudocounts():
    labels, counts, invalid = _amino_codon_counts_with_invalid(
        ["AUG", "AAA", "AUG"], "M", 0
    )

    assert labels == ["AUG", "Invalid"]
    assert counts == [2, 1]
    assert invalid == 1


def test_all_invalid_amino_codons_are_not_drawn_as_valid():
    labels, counts, invalid = _amino_codon_counts_with_invalid(["AAA", "CCC"], "M", 0)

    assert labels == ["AUG", "Invalid"]
    assert counts == [0, 2]
    assert invalid == 2


def test_scatter_limits_expand_to_include_out_of_range_targets():
    lower, upper = _expanded_axis_limits([1.0, 12.0], (2.0, 9.0))

    assert lower < 1.0
    assert upper > 12.0


@pytest.mark.parametrize(
    ("mrl", "mfe", "cai", "expected", "expected_description", "scatter_targets"),
    [
        (8.0, None, None, {"mrl": 8.0}, "MRL=8", []),
        (None, -20.0, None, {"mfe": -20.0}, "MFE=-20", []),
        (
            8.0,
            -2.0,
            None,
            {"mrl": 8.0, "mfe": -2.0},
            "MRL=8, MFE=-2",
            [[8.0, -2.0]],
        ),
        (None, None, None, {}, "unconditional generation", []),
        (None, None, 0.9, {"cai": 0.9}, "CAI-control alpha=0.9", []),
        (
            8.0,
            -2.0,
            0.9,
            {"mrl": 8.0, "mfe": -2.0, "cai": 0.9},
            "MRL=8, MFE=-2, CAI-control alpha=0.9",
            [[8.0, -2.0]],
        ),
    ],
)
def test_conditions_omit_missing_values_and_include_cai(
    mrl, mfe, cai, expected, expected_description, scatter_targets
):
    conditions = _normalise_conditions(SimpleNamespace(mrl=mrl, mfe=mfe, cai=cai))

    assert conditions == expected
    assert _describe_conditions(conditions) == expected_description
    assert _scatter_target_pairs(conditions) == scatter_targets
    assert "nan" not in expected_description.lower()


def test_only_complete_target_pairs_are_drawable():
    targets = [[8.0, np.nan], [np.nan, -20.0], [7.0, -2.0]]

    assert _finite_target_pairs(targets) == [[7.0, -2.0]]


def _write_plot_input(tmp_path):
    csv_path = tmp_path / "generated.csv"
    pd.DataFrame(
        {
            "Sequence": ["A" * 50, "C" * 50],
            "MRL": [7.5, 8.1],
            "MFE": [-3.0, -2.5],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def _plot_args(tmp_path, **overrides):
    values = {
        "mrl": 8.0,
        "mfe": -2.0,
        "nucleotide": None,
        "amino": None,
        "cds_amino": None,
        "cai": None,
        "out": str(tmp_path / "generated.fasta"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("mrl", "mfe", "cai", "expected_condition"),
    [
        (8.0, None, None, "MRL=8"),
        (None, -20.0, None, "MFE=-20"),
        (None, None, None, "unconditional generation"),
        (None, None, 0.9, "CAI-control alpha=0.9"),
    ],
)
def test_read_csv_and_plot_omits_missing_conditions_and_target_point(
    tmp_path, monkeypatch, mrl, mfe, cai, expected_condition
):
    calls = {}
    monkeypatch.setattr(
        visualization,
        "plot_MRL_MFE_scatter",
        lambda **kwargs: calls.update(kwargs),
    )

    visualization.read_csv_and_plot(
        _write_plot_input(tmp_path),
        _plot_args(tmp_path, mrl=mrl, mfe=mfe, cai=cai),
    )

    assert calls["title"] == (
        "MRL-MFE Distribution of Generated Sequences(n=2)\n"
        f"Condition: {expected_condition}"
    )
    assert calls["targets"] == []
    assert "nan" not in calls["title"].lower()


def test_cai_plot_condition_label_includes_alpha_for_cai_only(tmp_path, monkeypatch):
    seen_conditions = []
    describe_conditions = visualization._describe_conditions

    def capture_conditions(conditions):
        seen_conditions.append(dict(conditions))
        return describe_conditions(conditions)

    monkeypatch.setattr(visualization, "_describe_conditions", capture_conditions)
    visualization.plot_cai_response(
        records=[
            {
                "target_MRL": float("nan"),
                "target_MFE": float("nan"),
                "CAI": 0.85,
                "peptide_valid": True,
            }
        ],
        target_adaptiveness=0.9,
        savepath=tmp_path / "cai.jpg",
    )

    assert seen_conditions == [{"cai": 0.9}]


def test_read_csv_and_plot_passes_arbitrary_nucleotide_regions(
    tmp_path, monkeypatch
):
    calls = {}
    monkeypatch.setattr(visualization, "plot_MRL_MFE_scatter", lambda **_: None)
    monkeypatch.setattr(
        visualization,
        "plot_codon_constraint_duopanel",
        lambda **kwargs: calls.update(kwargs),
    )

    visualization.read_csv_and_plot(
        _write_plot_input(tmp_path),
        _plot_args(tmp_path, nucleotide=["8:AGCU", "20:A"]),
    )

    assert calls["nucleotide_regions"] == [(8, 4), (20, 1)]
    assert calls["title"] == (
        "Nucleotide Constraints: 8:AGCU 20:A\n"
        "Condition: MRL=8, MFE=-2"
    )


def test_read_csv_and_plot_preserves_sparse_amino_positions(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(visualization, "plot_MRL_MFE_scatter", lambda **_: None)
    monkeypatch.setattr(
        visualization,
        "plot_amino_constraint_tripanel",
        lambda **kwargs: calls.update(kwargs),
    )

    visualization.read_csv_and_plot(
        _write_plot_input(tmp_path),
        _plot_args(tmp_path, amino=["26:M", "31:D", "37:L"]),
    )

    assert calls["amino"] == ["M", "D", "L"]
    assert calls["amino_pos"] == [26, 31, 37]
    assert calls["title"] == (
        "Amino-acid Constraints: 26:M 31:D 37:L\n"
        "Condition: MRL=8, MFE=-2"
    )


def test_read_csv_and_plot_derives_cds_amino_tail_positions(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(visualization, "plot_MRL_MFE_scatter", lambda **_: None)
    monkeypatch.setattr(
        visualization,
        "plot_amino_constraint_tripanel",
        lambda **kwargs: calls.update(kwargs),
    )

    visualization.read_csv_and_plot(
        _write_plot_input(tmp_path),
        _plot_args(tmp_path, cds_amino="MGKVKVGV", cai=0.9),
    )

    assert calls["amino"] == list("MGKVKVGV")
    assert calls["amino_pos"] == [26, 29, 32, 35, 38, 41, 44, 47]
    assert calls["title"] == (
        "CDS Amino-acid Constraint: MGKVKVGV\n"
        "Condition: MRL=8, MFE=-2, CAI-control alpha=0.9"
    )
