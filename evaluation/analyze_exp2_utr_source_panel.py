#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion")
PANEL_ROOT = ROOT / "outputs/plan_bc_common_panel_n100_seed20260804_v1"
UTRDIFFUSION_PATH = ROOT / "outputs/real_MRL_pred_MFE_260k_mcml/samples/MRL_9.0_3k_samples.csv"
OUTPUT_DIR = ROOT / "outputs/exp2_utr_source_panel_analysis/mrl9_nan_v1_20260806"
GROUP_ORDER = ["UTRGAN terminal-50", "UTRGAN native 50-100", "Optimus original 50", "UTR-Diffusion MRL=9 original 50", "UTR-Diffusion+LinearDesign", "UTR-Diffusion+LinearCDSFold", "UTR-Diffusion+DERNA"]
COLORS = ["#4C8DA5", "#7AAFC1", "#B477A8", "#D58A5D", "#708BC1", "#5678B8", "#3E64A7"]


def normalize(sequence):
    return str(sequence).strip().upper().replace("T", "U")


def write_fasta(path, rows):
    with path.open("w") as handle:
        for record_id, sequence in rows:
            handle.write(f">{record_id}\n{normalize(sequence)}\n")


def prepare():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    utrgan = pd.read_csv(PANEL_ROOT / "utrgan_utr20_candidates_n100.tsv", sep="\t")
    optimus = pd.read_csv(PANEL_ROOT / "optimus_utr20_candidates_n100.tsv", sep="\t")
    common = pd.read_csv(PANEL_ROOT / "common_pairing_panel_n100.tsv", sep="\t").sort_values(["peptide_id", "pairing_index"]).reset_index(drop=True)
    keys = set(map(tuple, common[["peptide_id", "pairing_index"]].to_numpy()))
    utrgan["ID"] = utrgan.apply(lambda row: f"utrgan__g{int(row.group_index):02d}__p{int(row.pairing_index):04d}", axis=1)
    utrgan["sequence"] = utrgan.optimized_full_utr.map(normalize)
    utrgan["original_length"] = utrgan.sequence.str.len()
    utrgan[["peptide_id", "pairing_index", "ID", "original_length"]].to_csv(OUTPUT_DIR / "utrgan_selected_original_lengths.tsv", sep="\t", index=False)
    counts = utrgan.original_length.value_counts().sort_index().rename_axis("original_length").rename("count").reset_index()
    counts.to_csv(OUTPUT_DIR / "utrgan_selected_original_length_counts.tsv", sep="\t", index=False)
    length_summary = pd.DataFrame([{"n": len(utrgan), "min": utrgan.original_length.min(), "max": utrgan.original_length.max(), "mean": utrgan.original_length.mean(), "median": utrgan.original_length.median(), "n_below_50": utrgan.original_length.lt(50).sum(), "n_50_to_100": utrgan.original_length.between(50, 100).sum(), "n_above_100": utrgan.original_length.gt(100).sum()}])
    length_summary.to_csv(OUTPUT_DIR / "utrgan_selected_original_length_summary.tsv", sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(counts.original_length, counts["count"], width=.85, color="#4C8DA5")
    ax.set_xlabel("Original optimized UTR length (nt)")
    ax.set_ylabel("Sequence count")
    ax.set_title("UTRGAN original length distribution: Exp2 selected 3,000 sequences")
    ax.set_xticks(range(int(counts.original_length.min()), int(counts.original_length.max()) + 1, 3))
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "utrgan_selected_original_length_bar.png", dpi=300)
    plt.close(fig)
    write_fasta(OUTPUT_DIR / "utrgan_terminal50_actual3000.fasta", [(row.ID, row.sequence[-50:]) for row in utrgan.itertuples()])
    write_fasta(OUTPUT_DIR / "utrgan_native50_100.fasta", [(row.ID, row.sequence) for row in utrgan.itertuples() if 50 <= row.original_length <= 100])
    optimus["ID"] = optimus.apply(lambda row: f"optimus__g{int(row.group_index):02d}__p{int(row.pairing_index):04d}", axis=1)
    optimus["sequence"] = optimus.optimized_full50_utr.map(normalize)
    write_fasta(OUTPUT_DIR / "optimus_original50.fasta", optimus[["ID", "sequence"]].itertuples(index=False, name=None))
    diffusion = pd.read_csv(UTRDIFFUSION_PATH)
    diffusion["sample_index"] = diffusion.ID.str.extract(r"_seq_(\d+)$")[0].astype(int)
    diffusion.sort_values("sample_index", inplace=True)
    diffusion.reset_index(drop=True, inplace=True)
    if len(diffusion) != len(common) or diffusion.sample_index.tolist() != list(range(len(diffusion))):
        raise ValueError("UTR-Diffusion sample IDs are not exactly seq_0..seq_2999")
    diffusion[["peptide_id", "pairing_index"]] = common[["peptide_id", "pairing_index"]]
    diffusion.to_csv(OUTPUT_DIR / "utrdiffusion_mrl9_key_mapping.tsv", sep="\t", index=False)
    write_fasta(OUTPUT_DIR / "utrdiffusion_mrl9_original50.fasta", diffusion[["ID", "Sequence"]].itertuples(index=False, name=None))
    for model, filename in [("LinearDesign", "lineardesign_cds30_candidates_n100.tsv"), ("LinearCDSFold", "linearcdsfold_cds30_candidates_n100.tsv"), ("DERNA", "derna_cds30_candidates_n100.tsv")]:
        cds = pd.read_csv(PANEL_ROOT / filename, sep="\t")
        if set(map(tuple, cds[["peptide_id", "pairing_index"]].to_numpy())) != keys:
            raise ValueError(f"{model}: keys differ from common panel")
        merged = diffusion.merge(cds[["peptide_id", "pairing_index", "raw_sequence"]], on=["peptide_id", "pairing_index"], validate="one_to_one")
        merged["Sequence"] = merged.Sequence.map(normalize).str[-20:] + merged.raw_sequence.map(normalize)
        merged["record_id"] = merged.apply(lambda row: f"utrdiffusion__{model.lower()}__{row.peptide_id}__{int(row.pairing_index):04d}", axis=1)
        merged[["peptide_id", "pairing_index", "ID", "record_id", "Sequence"]].to_csv(OUTPUT_DIR / f"utrdiffusion_terminal20_{model.lower()}_mapping.tsv", sep="\t", index=False)
        write_fasta(OUTPUT_DIR / f"utrdiffusion_terminal20_{model.lower()}.fasta", merged[["record_id", "Sequence"]].itertuples(index=False, name=None))
    manifest = {"pairing_rule": "UTR-Diffusion seq_0..seq_2999 assigned to common panel sorted by peptide_id,pairing_index", "utrgan_rows": len(utrgan), "utrgan_native50_100_rows": int(utrgan.original_length.between(50, 100).sum()), "optimus_rows": len(optimus), "utrdiffusion_rows": len(diffusion)}
    (OUTPUT_DIR / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(length_summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def load_metrics(filename, group, include_mfe=True):
    data = pd.read_csv(OUTPUT_DIR / filename)
    result = data[["ID", "Sequence", "MRL", "MFE"]].copy()
    result.insert(0, "group", group)
    if not include_mfe:
        result["MFE"] = float("nan")
    return result


def summarize():
    frames = [
        load_metrics("utrgan_terminal50_actual3000.csv", GROUP_ORDER[0]),
        load_metrics("utrgan_native50_100.csv", GROUP_ORDER[1], include_mfe=False),
        load_metrics("optimus_original50.csv", GROUP_ORDER[2]),
        load_metrics("utrdiffusion_mrl9_original50.csv", GROUP_ORDER[3]),
        load_metrics("utrdiffusion_terminal20_lineardesign.csv", GROUP_ORDER[4]),
        load_metrics("utrdiffusion_terminal20_linearcdsfold.csv", GROUP_ORDER[5]),
        load_metrics("utrdiffusion_terminal20_derna.csv", GROUP_ORDER[6]),
    ]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(OUTPUT_DIR / "all_baseline_sequence_metrics.tsv", sep="\t", index=False)
    summary = metrics.groupby("group", sort=False).agg(n=("MRL", "size"), length_min=("Sequence", lambda x: x.str.len().min()), length_max=("Sequence", lambda x: x.str.len().max()), mrl_mean=("MRL", "mean"), mrl_std=("MRL", "std"), mrl_median=("MRL", "median"), mrl_min=("MRL", "min"), mrl_max=("MRL", "max"), mfe_count=("MFE", "count"), mfe_mean=("MFE", "mean"), mfe_std=("MFE", "std"), mfe_median=("MFE", "median"), mfe_min=("MFE", "min"), mfe_max=("MFE", "max")).reset_index()
    summary.to_csv(OUTPUT_DIR / "baseline_metric_summary.tsv", sep="\t", index=False)
    supplied = pd.read_csv(UTRDIFFUSION_PATH).set_index("ID")
    reevaluated = frames[3].set_index("ID")
    check = {"n": len(supplied), "max_abs_mrl_delta": float((supplied.MRL - reevaluated.MRL).abs().max()), "max_abs_mfe_delta": float((supplied.MFE - reevaluated.MFE).abs().max())}
    (OUTPUT_DIR / "utrdiffusion_source_reevaluation_check.json").write_text(json.dumps(check, indent=2) + "\n")
    for metric, ylabel, filename in [("MRL", "Predicted MRL", "baseline_MRL_violin.png"), ("MFE", "MFE (kcal/mol)", "baseline_MFE_violin.png")]:
        plot_data = metrics.dropna(subset=[metric])
        order = [group for group in GROUP_ORDER if group in set(plot_data.group)]
        palette = {group: COLORS[GROUP_ORDER.index(group)] for group in order}
        fig, ax = plt.subplots(figsize=(18, 7))
        sns.violinplot(data=plot_data, x="group", y=metric, order=order, palette=palette, inner="quartile", cut=0, linewidth=1, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Exp2 UTR source and UTR-Diffusion CDS assembly: {metric}")
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close(fig)
    print(summary.to_string(index=False))
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "summarize"])
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else summarize()
