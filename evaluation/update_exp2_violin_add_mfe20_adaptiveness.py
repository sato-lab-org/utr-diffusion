#!/usr/bin/env python3
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion")
OUTPUT_DIR = ROOT / "outputs/exp2_utr_source_panel_analysis/mrl9_nan_v1_20260806"
ADAPT_DIR = ROOT / "outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MRL_MFE_CAI_optimization/mrl_9_mfe_-20_cond_4_gamma_0.04/benchmark_evaluation"
ADAPTIVENESS = ["0.85", "0.9", "0.95", "None"]
ADAPT_FILES = {value: f"utr_diffusion_adaptiveness_{value.lower().replace('.', '_')}_mrl_mfe_cai.csv" for value in ADAPTIVENESS}
ADAPT_GROUPS = {value: f"UTR-Diffusion a={value}" for value in ADAPTIVENESS}
ASSEMBLED_FILES = {
    "UTRGAN+LinearDesign": "utrgan_lineardesign__modular_plan_c_mrl_mfe_cai.csv", "UTRGAN+LinearCDSFold": "utrgan_linearcdsfold__modular_plan_c_mrl_mfe_cai.csv", "UTRGAN+DERNA": "utrgan_derna__modular_plan_c_mrl_mfe_cai.csv",
    "Optimus+LinearDesign": "optimus_lineardesign__modular_plan_c_mrl_mfe_cai.csv", "Optimus+LinearCDSFold": "optimus_linearcdsfold__modular_plan_c_mrl_mfe_cai.csv", "Optimus+DERNA": "optimus_derna__modular_plan_c_mrl_mfe_cai.csv",
}
ORDER = [
    "UTRGAN terminal-50", "UTRGAN native 50-100", "Optimus original 50", "UTR-Diffusion MRL=9 original 50",
    "UTRGAN+LinearDesign", "UTRGAN+LinearCDSFold", "UTRGAN+DERNA",
    "Optimus+LinearDesign", "Optimus+LinearCDSFold", "Optimus+DERNA",
    "UTR-Diffusion+LinearDesign", "UTR-Diffusion+LinearCDSFold", "UTR-Diffusion+DERNA",
    *ADAPT_GROUPS.values(),
]
COLORS = {
    "UTRGAN terminal-50": "#4C8DA5", "UTRGAN native 50-100": "#7AAFC1", "Optimus original 50": "#B477A8", "UTR-Diffusion MRL=9 original 50": "#D58A5D",
    "UTRGAN+LinearDesign": "#6FA5B5", "UTRGAN+LinearCDSFold": "#568FA4", "UTRGAN+DERNA": "#3D798F",
    "Optimus+LinearDesign": "#B477A8", "Optimus+LinearCDSFold": "#9A6296", "Optimus+DERNA": "#815184",
    "UTR-Diffusion+LinearDesign": "#708BC1", "UTR-Diffusion+LinearCDSFold": "#5678B8", "UTR-Diffusion+DERNA": "#3E64A7",
    "UTR-Diffusion a=0.85": "#72B7A8", "UTR-Diffusion a=0.9": "#3C9C88", "UTR-Diffusion a=0.95": "#147A6B", "UTR-Diffusion a=None": "#A0A0A0",
}


def plot_violin(data, metric, path):
    plot_data = data.dropna(subset=[metric])
    order = [group for group in ORDER if group in set(plot_data["group"])]
    fig, ax = plt.subplots(figsize=(32, 10))
    sns.violinplot(data=plot_data, x="group", y=metric, order=order, hue="group", palette=COLORS, dodge=False, legend=False, cut=0, width=.94, density_norm="width", common_norm=False, bw_adjust=.9, inner="box", linewidth=1.1, inner_kws={"box_width": 4, "whis_width": 1.2, "color": "black"}, ax=ax)
    ax.set_xlabel("Baseline", fontsize=18)
    ax.set_ylabel("Predicted MRL" if metric == "MRL" else "MFE (kcal/mol)", fontsize=18)
    ax.set_title(f"Exp2 UTR source, CDS assembly, and MFE=-20 UTR-Diffusion conditions: {metric}", fontsize=22)
    ax.tick_params(axis="x", rotation=35, labelsize=12)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    metrics_path = OUTPUT_DIR / "all_baseline_sequence_metrics.tsv"
    summary_path = OUTPUT_DIR / "baseline_metric_summary.tsv"
    metrics_backup = OUTPUT_DIR / "all_baseline_sequence_metrics_without_mfe20_adaptiveness.tsv"
    summary_backup = OUTPUT_DIR / "baseline_metric_summary_without_mfe20_adaptiveness.tsv"
    if not metrics_backup.exists():
        shutil.copy2(metrics_path, metrics_backup)
    if not summary_backup.exists():
        shutil.copy2(summary_path, summary_backup)
    metrics = pd.read_csv(metrics_path, sep="\t")
    metrics = metrics[~metrics["group"].isin([*ASSEMBLED_FILES, *ADAPT_GROUPS.values()])]
    frames = [metrics]
    for group, filename in ASSEMBLED_FILES.items():
        source = pd.read_csv(ADAPT_DIR / filename)
        frames.append(pd.DataFrame({"group": group, "ID": source["candidate_id"], "Sequence": source["Sequence"], "MRL": source["MRL"], "MFE": source["MFE"]}))
    for value in ADAPTIVENESS:
        source = pd.read_csv(ADAPT_DIR / ADAPT_FILES[value])
        frames.append(pd.DataFrame({"group": ADAPT_GROUPS[value], "ID": source["candidate_id"], "Sequence": source["Sequence"], "MRL": source["MRL"], "MFE": source["MFE"]}))
    metrics = pd.concat(frames, ignore_index=True)
    metrics["group"] = pd.Categorical(metrics["group"], categories=ORDER, ordered=True)
    metrics = metrics.sort_values(["group", "ID"]).reset_index(drop=True)
    metrics.to_csv(metrics_path, sep="\t", index=False)

    summary = metrics.groupby("group", observed=True).agg(n=("Sequence", "size"), length_min=("Sequence", lambda x: x.str.len().min()), length_max=("Sequence", lambda x: x.str.len().max()), mrl_mean=("MRL", "mean"), mrl_std=("MRL", "std"), mrl_median=("MRL", "median"), mrl_min=("MRL", "min"), mrl_max=("MRL", "max"), mfe_count=("MFE", "count"), mfe_mean=("MFE", "mean"), mfe_std=("MFE", "std"), mfe_median=("MFE", "median"), mfe_min=("MFE", "min"), mfe_max=("MFE", "max")).reset_index()
    summary.insert(1, "color_hex", summary["group"].map(COLORS))
    summary.to_csv(summary_path, sep="\t", index=False)
    summary.to_csv(OUTPUT_DIR / "baseline_group_statistics_and_colors.tsv", sep="\t", index=False)

    for metric in ["MRL", "MFE"]:
        current = OUTPUT_DIR / f"baseline_{metric}_violin.png"
        backup = OUTPUT_DIR / f"baseline_{metric}_violin_without_mfe20_adaptiveness_failed.png"
        if current.exists() and not backup.exists():
            shutil.copy2(current, backup)
        plot_violin(metrics, metric, current)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
