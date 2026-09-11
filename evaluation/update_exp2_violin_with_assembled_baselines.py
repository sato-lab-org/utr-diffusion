from pathlib import Path
import shutil

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


OUTPUT_DIR = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/exp2_utr_source_panel_analysis/mrl9_nan_v1_20260806")
BENCHMARK_DIR = Path("/gs/bs/tga-satolab-gtex/dai/myscript/utr-diffusion/outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MRL_MFE_CAI_optimization/mrl_9_mfe_-15_cond_4_gamma_0.02/benchmark_evaluation")
ASSEMBLED = {
    "UTRGAN+LinearDesign": "utrgan_lineardesign__modular_plan_c_mrl_mfe_cai.csv",
    "UTRGAN+LinearCDSFold": "utrgan_linearcdsfold__modular_plan_c_mrl_mfe_cai.csv",
    "UTRGAN+DERNA": "utrgan_derna__modular_plan_c_mrl_mfe_cai.csv",
    "Optimus+LinearDesign": "optimus_lineardesign__modular_plan_c_mrl_mfe_cai.csv",
    "Optimus+LinearCDSFold": "optimus_linearcdsfold__modular_plan_c_mrl_mfe_cai.csv",
    "Optimus+DERNA": "optimus_derna__modular_plan_c_mrl_mfe_cai.csv",
}
ORDER = [
    "UTRGAN terminal-50", "UTRGAN native 50-100", "Optimus original 50", "UTR-Diffusion MRL=9 original 50",
    "UTRGAN+LinearDesign", "UTRGAN+LinearCDSFold", "UTRGAN+DERNA",
    "Optimus+LinearDesign", "Optimus+LinearCDSFold", "Optimus+DERNA",
    "UTR-Diffusion+LinearDesign", "UTR-Diffusion+LinearCDSFold", "UTR-Diffusion+DERNA",
]


def plot_violin(data, metric, path):
    plot_data = data.dropna(subset=[metric])
    order = [group for group in ORDER if group in plot_data["group"].unique()]
    palette = {group: "#4C72B0" if group.startswith("UTRGAN") else "#DD8452" if group.startswith("Optimus") else "#55A868" for group in ORDER}
    plt.figure(figsize=(25, 10))
    sns.violinplot(data=plot_data, x="group", y=metric, order=order, palette=palette, hue="group", legend=False, cut=0, inner="box", linewidth=1.2, inner_kws={"box_width": 8, "whis_width": 1.5, "color": "black"})
    plt.xlabel("Baseline", fontsize=18)
    plt.ylabel(metric if metric == "MRL" else "MFE (kcal/mol)", fontsize=18)
    plt.title(f"Exp2 UTR source and assembled baseline {metric} distributions", fontsize=22)
    plt.xticks(rotation=35, ha="right", fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    metrics_path = OUTPUT_DIR / "all_baseline_sequence_metrics.tsv"
    old = pd.read_csv(metrics_path, sep="\t")
    old = old[~old["group"].isin(ASSEMBLED)]
    frames = [old]
    for group, filename in ASSEMBLED.items():
        source = pd.read_csv(BENCHMARK_DIR / filename)
        frames.append(pd.DataFrame({"group": group, "ID": source["candidate_id"], "Sequence": source["Sequence"], "MRL": source["MRL"], "MFE": source["MFE"]}))
    data = pd.concat(frames, ignore_index=True)
    data["group"] = pd.Categorical(data["group"], categories=ORDER, ordered=True)
    data = data.sort_values(["group", "ID"]).reset_index(drop=True)

    for metric in ["MRL", "MFE"]:
        current = OUTPUT_DIR / f"baseline_{metric}_violin.png"
        backup = OUTPUT_DIR / f"baseline_{metric}_violin_without_6_assembled_baselines_failed.png"
        if current.exists() and not backup.exists():
            shutil.copy2(current, backup)
        plot_violin(data, metric, current)

    data.to_csv(metrics_path, sep="\t", index=False)
    summary = data.groupby("group", observed=True).agg(n=("Sequence", "size"), length_min=("Sequence", lambda x: x.str.len().min()), length_max=("Sequence", lambda x: x.str.len().max()), mrl_mean=("MRL", "mean"), mrl_std=("MRL", "std"), mrl_median=("MRL", "median"), mrl_min=("MRL", "min"), mrl_max=("MRL", "max"), mfe_count=("MFE", "count"), mfe_mean=("MFE", "mean"), mfe_std=("MFE", "std"), mfe_median=("MFE", "median"), mfe_min=("MFE", "min"), mfe_max=("MFE", "max")).reset_index()
    summary.to_csv(OUTPUT_DIR / "baseline_metric_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
