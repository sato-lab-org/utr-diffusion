from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("logomaker")

import src.plot.visualization as visualization  # noqa: E402
from src.plot.visualization import (  # noqa: E402
    _amino_codon_counts_with_invalid,
    _describe_targets,
    _derived_plot_path,
    _expanded_axis_limits,
    _finite_target_pairs,
    _normalise_target_pairs,
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
    ("mrl", "mfe", "expected_pair", "expected_description"),
    [
        (8.0, None, [8.0, np.nan], "MRL-only target MRL=8"),
        (None, -20.0, [np.nan, -20.0], "MFE-only target MFE=-20"),
        (8.0, -2.0, [8.0, -2.0], "target MRL=8, MFE=-2"),
        (None, None, [np.nan, np.nan], "MRL/MFE-unconditioned generation"),
    ],
)
def test_explicit_targets_support_single_label_and_unconditioned_titles(
    mrl, mfe, expected_pair, expected_description
):
    pairs = _normalise_target_pairs(
        SimpleNamespace(mrl=mrl, mfe=mfe, targets=None)
    )

    np.testing.assert_equal(pairs[0], expected_pair)
    assert _describe_targets(pairs) == expected_description


def test_only_complete_target_pairs_are_drawable():
    targets = [[8.0, np.nan], [np.nan, -20.0], [7.0, -2.0]]

    assert _finite_target_pairs(targets) == [[7.0, -2.0]]


def test_advanced_batch_targets_remain_readable():
    args = SimpleNamespace(
        mrl=None,
        mfe=None,
        targets=["4,-20", "8,-2"],
    )

    assert _normalise_target_pairs(args) == [[4.0, -20.0], [8.0, -2.0]]


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
        "targets": None,
        "nucleotide": None,
        "amino": None,
        "cds_amino": None,
        "cai": None,
        "out": str(tmp_path / "generated.fasta"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_read_csv_and_plot_uses_single_label_title_without_nan_target_point(
    tmp_path, monkeypatch
):
    calls = {}
    monkeypatch.setattr(
        visualization,
        "plot_MRL_MFE_scatter",
        lambda **kwargs: calls.update(kwargs),
    )

    visualization.read_csv_and_plot(
        _write_plot_input(tmp_path),
        _plot_args(tmp_path, mrl=8.0, mfe=None),
    )

    assert "MRL-only target MRL=8" in calls["title"]
    assert np.isnan(calls["targets"][0][1])
    assert _finite_target_pairs(calls["targets"]) == []


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
    assert calls["title"] == "Nucleotide Constraints: 8:AGCU 20:A"


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
    assert "Requested codon relative adaptiveness alpha=0.9" in calls["title"]
