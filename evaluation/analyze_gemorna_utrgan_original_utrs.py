#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


GEMORNA_PATH = Path("/gs/bs/tga-satolab-gtex/dai/myscript/GEMORNA/commit-3bba88e4c0a3/output/full/gemorna_full_v1_20260802T091506Z/candidates.tsv")
UTRGAN_PATH = Path("/gs/bs/tga-satolab-gtex/dai/myscript/UTRGAN/src/mrl_te_optimization/outputs/exp2/utrgan_exp2_posthoc20_v1_20260803T1532/final/all_30000_candidates.tsv")
OUTPUT_DIR = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/original_utr_mrl_analysis/gemorna_utrgan_original_v1_20260806")
PANEL_ROOT = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/plan_bc_common_panel_n100_seed20260804_v1")
BENCHMARK_PATH = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MRL_MFE_CAI_optimization/mrl_9_mfe_-20_cond_4_gamma_0.04/benchmark_evaluation/all_plan_bc_methods_mrl_mfe_cai.csv")


def write_fasta(path, rows):
    with path.open("w") as handle:
        for candidate_id, sequence in rows:
            handle.write(f">{candidate_id}\n{sequence}\n")


def prepare():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gemorna = pd.read_csv(GEMORNA_PATH, sep="\t", usecols=["candidate_id", "utr_raw_sequence"])
    utrgan = pd.read_csv(UTRGAN_PATH, sep="\t", usecols=["group_index", "pairing_index", "optimized_full_utr"])
    gemorna["candidate_id"] = gemorna["candidate_id"].astype(str)
    gemorna["sequence"] = gemorna["utr_raw_sequence"].str.upper().str.replace("T", "U")
    utrgan["candidate_id"] = utrgan.apply(lambda row: f"utrgan__g{int(row.group_index):02d}__p{int(row.pairing_index):04d}", axis=1)
    utrgan["sequence"] = utrgan["optimized_full_utr"].str.upper().str.replace("T", "U")
    lengths = pd.concat([
        pd.DataFrame({"model": "GEMORNA", "candidate_id": gemorna["candidate_id"], "original_length": gemorna["sequence"].str.len()}),
        pd.DataFrame({"model": "UTRGAN", "candidate_id": utrgan["candidate_id"], "original_length": utrgan["sequence"].str.len()}),
    ], ignore_index=True)
    lengths.to_csv(OUTPUT_DIR / "original_utr_lengths.tsv", sep="\t", index=False)
    lengths.groupby(["model", "original_length"]).size().rename("count").reset_index().to_csv(OUTPUT_DIR / "original_utr_length_counts.tsv", sep="\t", index=False)
    summary = lengths.groupby("model")["original_length"].agg(["count", "min", "median", "max", "mean", "std"]).reset_index()
    summary["n_below_50"] = summary["model"].map(lengths[lengths.original_length < 50].groupby("model").size()).fillna(0).astype(int)
    summary["n_50_to_100"] = summary["model"].map(lengths[lengths.original_length.between(50, 100)].groupby("model").size()).fillna(0).astype(int)
    summary["n_above_100"] = summary["model"].map(lengths[lengths.original_length > 100].groupby("model").size()).fillna(0).astype(int)
    summary.to_csv(OUTPUT_DIR / "original_utr_length_summary.tsv", sep="\t", index=False)
    datasets = {
        "gemorna_terminal50_from_original_ge50": [(row.candidate_id, row.sequence[-50:]) for row in gemorna.itertuples() if len(row.sequence) >= 50],
        "gemorna_native50_100": [(row.candidate_id, row.sequence) for row in gemorna.itertuples() if 50 <= len(row.sequence) <= 100],
        "utrgan_terminal50_from_original_ge50": [(row.candidate_id, row.sequence[-50:]) for row in utrgan.itertuples() if len(row.sequence) >= 50],
        "utrgan_native50_100": [(row.candidate_id, row.sequence) for row in utrgan.itertuples() if 50 <= len(row.sequence) <= 100],
        "utrgan_terminal100_from_original_gt100": [(row.candidate_id, row.sequence[-100:]) for row in utrgan.itertuples() if len(row.sequence) > 100],
    }
    for name, rows in datasets.items():
        write_fasta(OUTPUT_DIR / f"{name}.fasta", rows)
        print(name, len(rows))
    print(summary.to_string(index=False))


def summarize():
    rows = []
    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        data = pd.read_csv(path)
        rows.append({"dataset": path.stem, "n": len(data), "length_min": data.Sequence.str.len().min(), "length_median": data.Sequence.str.len().median(), "length_max": data.Sequence.str.len().max(), "mrl_mean": data.MRL.mean(), "mrl_std": data.MRL.std(), "mrl_median": data.MRL.median(), "mrl_q05": data.MRL.quantile(0.05), "mrl_q95": data.MRL.quantile(0.95), "mrl_min": data.MRL.min(), "mrl_max": data.MRL.max()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "utrlm_mrl_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    comparisons = []
    for name, terminal_path, alternate_path in [
        ("GEMORNA terminal50 vs native50_100", "gemorna_terminal50_from_original_ge50.csv", "gemorna_native50_100.csv"),
        ("UTRGAN terminal50 vs native50_100", "utrgan_terminal50_from_original_ge50.csv", "utrgan_native50_100.csv"),
        ("UTRGAN terminal50 vs terminal100_gt100", "utrgan_terminal50_from_original_ge50.csv", "utrgan_terminal100_from_original_gt100.csv"),
    ]:
        terminal = pd.read_csv(OUTPUT_DIR / terminal_path).set_index("ID")
        alternate = pd.read_csv(OUTPUT_DIR / alternate_path).set_index("ID")
        terminal = terminal.loc[alternate.index]
        delta = terminal.MRL - alternate.MRL
        comparisons.append({"comparison": name, "n": len(delta), "terminal50_mrl_mean": terminal.MRL.mean(), "alternate_mrl_mean": alternate.MRL.mean(), "terminal50_minus_alternate_mean": delta.mean(), "terminal50_minus_alternate_median": delta.median(), "pearson_r": terminal.MRL.corr(alternate.MRL)})
    pd.DataFrame(comparisons).to_csv(OUTPUT_DIR / "utrlm_window_sensitivity.tsv", sep="\t", index=False)
    benchmark = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    gemorna_panel = pd.read_csv(PANEL_ROOT / "gemorna_rna50_candidates_n100.tsv", sep="\t")
    utrgan_panel = pd.read_csv(PANEL_ROOT / "utrgan_utr20_candidates_n100.tsv", sep="\t")
    gemorna_original = pd.read_csv(OUTPUT_DIR / "gemorna_terminal50_from_original_ge50.csv").set_index("ID")["MRL"]
    gemorna_panel = gemorna_panel[gemorna_panel.candidate_id.isin(gemorna_original.index)].copy()
    gemorna_panel["original_MRL"] = gemorna_panel.candidate_id.map(gemorna_original)
    utrgan_panel["ID"] = utrgan_panel.apply(lambda row: f"utrgan__g{int(row.group_index):02d}__p{int(row.pairing_index):04d}", axis=1)
    utrgan_original = pd.read_csv(OUTPUT_DIR / "utrgan_terminal50_from_original_ge50.csv").set_index("ID")["MRL"]
    utrgan_panel = utrgan_panel[utrgan_panel.ID.isin(utrgan_original.index)].copy()
    utrgan_panel["original_MRL"] = utrgan_panel.ID.map(utrgan_original)
    context = []
    for source, panel, models in [
        ("GEMORNA", gemorna_panel, ["GEMORNA"]),
        ("UTRGAN", utrgan_panel, ["UTRGAN+LinearDesign", "UTRGAN+LinearCDSFold", "UTRGAN+DERNA"]),
    ]:
        original = panel.set_index(["peptide_id", "pairing_index"])["original_MRL"]
        for model in models:
            assembled = benchmark[benchmark.model.eq(model)].set_index(["peptide_id", "pairing_index"])["MRL"].loc[original.index]
            delta = assembled - original
            context.append({"source": source, "assembled_model": model, "n": len(delta), "original_terminal50_mrl_mean": original.mean(), "assembled_rna50_mrl_mean": assembled.mean(), "assembled_minus_original_mean": delta.mean(), "assembled_minus_original_median": delta.median(), "pearson_r": original.corr(assembled)})
    pd.DataFrame(context).to_csv(OUTPUT_DIR / "exp2_common_panel_original_vs_assembled.tsv", sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "summarize"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else summarize()
