#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


OPTIMUS_PATH = Path("/gs/bs/tga-satolab-gtex/dai/myscript/human_5utr_modeling/outputs/exp2/optimus_exp2_posthoc20_target9_v1_20260803T1706JST/final/all_30000_candidates.tsv")
PANEL_PATH = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/plan_bc_common_panel_n100_seed20260804_v1/optimus_utr20_candidates_n100.tsv")
BENCHMARK_PATH = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MRL_MFE_CAI_optimization/mrl_9_mfe_-20_cond_4_gamma_0.04/benchmark_evaluation/all_plan_bc_methods_mrl_mfe_cai.csv")
OUTPUT_DIR = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/original_utr_mrl_analysis/optimus_original_v1_20260806")


def candidate_id(row):
    return f"optimus__g{int(row.group_index):02d}__p{int(row.pairing_index):04d}"


def prepare():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(OPTIMUS_PATH, sep="\t")
    data["candidate_id"] = data.apply(candidate_id, axis=1)
    data["sequence"] = data.optimized_full50_utr.str.upper().str.replace("T", "U")
    lengths = data[["candidate_id"]].copy()
    lengths["original_length"] = data.sequence.str.len()
    lengths.to_csv(OUTPUT_DIR / "optimus_original_lengths.tsv", sep="\t", index=False)
    lengths.groupby("original_length").size().rename("count").reset_index().to_csv(OUTPUT_DIR / "optimus_original_length_counts.tsv", sep="\t", index=False)
    lengths.original_length.agg(["count", "min", "median", "max", "mean", "std"]).to_frame().T.to_csv(OUTPUT_DIR / "optimus_original_length_summary.tsv", sep="\t", index=False)
    with (OUTPUT_DIR / "optimus_optimized_full50.fasta").open("w") as handle:
        for row in data.itertuples():
            handle.write(f">{row.candidate_id}\n{row.sequence}\n")
    print(lengths.original_length.value_counts().sort_index().to_string())


def summarize():
    source = pd.read_csv(OPTIMUS_PATH, sep="\t")
    source["ID"] = source.apply(candidate_id, axis=1)
    predicted = pd.read_csv(OUTPUT_DIR / "optimus_optimized_full50.csv")
    merged = source.merge(predicted[["ID", "MRL", "MFE"]], on="ID", validate="one_to_one")
    summary = pd.DataFrame([{"n": len(merged), "length_min": merged.optimized_full50_utr.str.len().min(), "length_median": merged.optimized_full50_utr.str.len().median(), "length_max": merged.optimized_full50_utr.str.len().max(), "utrlm_mrl_mean": merged.MRL.mean(), "utrlm_mrl_std": merged.MRL.std(), "utrlm_mrl_median": merged.MRL.median(), "utrlm_mrl_q05": merged.MRL.quantile(.05), "utrlm_mrl_q95": merged.MRL.quantile(.95), "utrlm_mrl_min": merged.MRL.min(), "utrlm_mrl_max": merged.MRL.max(), "optimus_mrl_mean": merged.optimized_predicted_mrl.mean(), "optimus_mrl_std": merged.optimized_predicted_mrl.std(), "optimus_vs_utrlm_pearson_r": merged.optimized_predicted_mrl.corr(merged.MRL)}])
    summary.to_csv(OUTPUT_DIR / "optimus_utrlm_mrl_summary.tsv", sep="\t", index=False)
    panel = pd.read_csv(PANEL_PATH, sep="\t")
    panel["ID"] = panel.apply(candidate_id, axis=1)
    original = panel.merge(predicted[["ID", "MRL"]], on="ID", validate="one_to_one").set_index(["peptide_id", "pairing_index"])["MRL"]
    benchmark = pd.read_csv(BENCHMARK_PATH, low_memory=False)
    rows = []
    for model in ["Optimus+LinearDesign", "Optimus+LinearCDSFold", "Optimus+DERNA"]:
        assembled = benchmark[benchmark.model.eq(model)].set_index(["peptide_id", "pairing_index"])["MRL"].loc[original.index]
        delta = assembled - original
        rows.append({"assembled_model": model, "n": len(delta), "all_original_full50_mrl_mean": merged.MRL.mean(), "selected_original_full50_mrl_mean": original.mean(), "assembled_rna50_mrl_mean": assembled.mean(), "assembled_minus_original_mean": delta.mean(), "assembled_minus_original_median": delta.median(), "pearson_r": original.corr(assembled)})
    context = pd.DataFrame(rows)
    context.to_csv(OUTPUT_DIR / "optimus_common_panel_original_vs_assembled.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(context.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "summarize"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else summarize()
