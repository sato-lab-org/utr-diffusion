import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import logomaker
from collections import Counter

figure_size = (10, 6)
logo_fig_size = (10, 2) #(8, 2)
label_fontsize = 18
title_fontsize = 18
legend_fontsize = 14
tick_fontsize = 14
text_fontsize = 14 #12
clabel_fontsize = 12 #10

AMINO_TO_CODONS = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],                  # Alanine           アラニン
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],    # Arginine          アルギニン
    'N': ['AAU', 'AAC'],                                # Asparagine        アスパラギン
    'D': ['GAU', 'GAC'],                                # Aspartic acid     アスパラギン酸
    'C': ['UGU', 'UGC'],                                # Cysteine          システイン
    'Q': ['CAA', 'CAG'],                                # Glutamine         グルタミン
    'E': ['GAA', 'GAG'],                                # Glutamic acid     グルタミン酸
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],                  # Glycine           グリシン
    'H': ['CAU', 'CAC'],                                # Histidine         ヒスチジン
    'I': ['AUU', 'AUC', 'AUA'],                         # Isoleucine        イソロイシン
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],    # Leucine           ロイシン
    'K': ['AAA', 'AAG'],                                # Lysine            リシン
    'M': ['AUG'],                                       # Methionine (START)メチオニン
    'F': ['UUU', 'UUC'],                                # Phenylalanine     フェニルアラニン
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],                  # Proline           プロリン
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],    # Serine
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],                  # Threonine         スレオニン
    'W': ['UGG'],                                       # Tryptophan        トリプトファン
    'Y': ['UAU', 'UAC'],                                # Tyrosine          チロシン
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],                  # Valine            バリン
    '*': ['UAA', 'UAG', 'UGA'],                         # Stop codons       終止コドン
}

def dna_to_rna(seq: str) -> str:
    return seq.replace('T', 'U')


def _derived_plot_path(output_path, suffix):
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}{suffix}"))


def _amino_codon_counts_with_invalid(seqs, amino, position):
    codons = AMINO_TO_CODONS[amino]
    counts = [sum(1 for seq in seqs if seq[position:position + 3] == codon) for codon in codons]
    invalid_count = len(seqs) - sum(counts)
    labels = list(codons)
    if invalid_count:
        labels.append("Invalid")
        counts.append(invalid_count)
    return labels, counts, invalid_count


def _expanded_axis_limits(values, baseline):
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    lower = min([float(baseline[0]), *finite_values])
    upper = max([float(baseline[1]), *finite_values])
    span = upper - lower
    padding = 0.03 * span if span > 0 else 1.0
    return lower - padding, upper + padding


def _normalise_target_pairs(args):
    """Return the conditioning target as a single ``[MRL, MFE]`` pair.

    Missing labels are represented by NaN so the plotting layer mirrors the
    masked-label representation used by MCML.
    """

    mrl = getattr(args, "mrl", None)
    mfe = getattr(args, "mfe", None)
    return [[
        float(mrl) if mrl is not None else float("nan"),
        float(mfe) if mfe is not None else float("nan"),
    ]]


def _finite_target_pairs(targets):
    """Select targets that can be drawn as points on the two-label scatter."""

    return [
        [float(target[0]), float(target[1])]
        for target in (targets or [])
        if np.isfinite(float(target[0])) and np.isfinite(float(target[1]))
    ]


def _describe_targets(targets):
    """Create a human-readable conditioning description, including masks."""

    if not targets:
        return "MRL/MFE-unconditioned generation"
    if len(targets) != 1:
        return f"{len(targets)} conditioning targets"

    target_mrl, target_mfe = (float(value) for value in targets[0])
    has_mrl, has_mfe = np.isfinite(target_mrl), np.isfinite(target_mfe)
    if has_mrl and has_mfe:
        return f"target MRL={target_mrl:g}, MFE={target_mfe:g}"
    if has_mrl:
        return f"MRL-only target MRL={target_mrl:g}"
    if has_mfe:
        return f"MFE-only target MFE={target_mfe:g}"
    return "MRL/MFE-unconditioned generation"


def _as_constraint_tokens(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def compute_shannon_entropy_base_and_amino(sequences, aminos, amino_pos_list):
    L = len(sequences[0])

    # Identify codon regions
    codon_positions = set()
    for p in amino_pos_list:
        codon_positions.update([p, p+1, p+2])

    base_positions = [i for i in range(L) if i not in codon_positions]

    # constants
    alpha = ['A', 'C', 'G', 'U']
    max_1mer_entropy = np.log2(4)

    # 1-mer entropy
    counts_1mer = [Counter() for _ in range(L)]
    raw_1mer_entropy = {}
    pos_entropy_1mer = np.array([None] * L, dtype=object)
    norm_entropy_1mer = np.zeros(L)

    for pos in base_positions:
        for seq in sequences:
            b = seq[pos]
            if b in alpha:
                counts_1mer[pos][b] += 1

    for pos in base_positions:
        total = sum(counts_1mer[pos].values())
        if total > 0:
            freq = np.array([counts_1mer[pos][b] / total for b in alpha])
            nz = freq[freq > 1e-12]
            H1 = -(nz * np.log2(nz)).sum()
        else:
            H1 = 0.0
        raw_1mer_entropy[pos] = H1
        pos_entropy_1mer[pos] = H1
        norm_entropy_1mer[pos] = H1 / max_1mer_entropy

    # 3-mer entropy
    raw_3mer_entropy = {}
    pos_entropy_3mer = np.array([None] * L, dtype=object)
    norm_entropy_3mer = np.zeros(L)

    for aa, p in zip(aminos, amino_pos_list):
        codons = AMINO_TO_CODONS[aa]
        k = len(codons)
        maxH3 = np.log2(k)

        codon_counts = {c: 0 for c in codons}
        for seq in sequences:
            codon = seq[p:p+3]
            if codon in codon_counts:
                codon_counts[codon] += 1

        freq = np.array(list(codon_counts.values()), float)
        freq /= freq.sum() + 1e-12
        nz = freq[freq > 1e-12]
        H3 = -(nz * np.log2(nz)).sum()

        raw_3mer_entropy[p] = H3

        # assign same entropy to p,p+1,p+2
        for pp in [p, p+1, p+2]:
            pos_entropy_3mer[pp] = H3
            norm_entropy_3mer[pp] = H3 / maxH3 if maxH3 > 0 else 0.0

    # merged entropy (original behavior)
    merged_entropy = np.zeros(L)
    max_entropy = np.zeros(L)

    for i in range(L):
        if i in base_positions:
            merged_entropy[i] = pos_entropy_1mer[i]
            max_entropy[i] = max_1mer_entropy
        else:
            for aa, p in zip(aminos, amino_pos_list):
                if i in [p, p+1, p+2]:
                    merged_entropy[i] = raw_3mer_entropy[p]
                    max_entropy[i] = np.log2(len(AMINO_TO_CODONS[aa]))
                    break

    normalized_entropy = merged_entropy / (max_entropy + 1e-12)

    return {
        "merged_entropy": merged_entropy,
        "normalized_entropy": normalized_entropy,
        "max_entropy": max_entropy,
        "pos_entropy_1mer": pos_entropy_1mer,
        "pos_entropy_3mer": pos_entropy_3mer,
        "norm_entropy_1mer": norm_entropy_1mer,
        "norm_entropy_3mer": norm_entropy_3mer,
        "raw_1mer_entropy": raw_1mer_entropy,
        "raw_3mer_entropy": raw_3mer_entropy,
    }


def compute_shannon_entropy_per_pos(sequences, alphabet="ACGU", eps=1e-12):
    sequences = [dna_to_rna(seq) for seq in sequences]
    length = len(sequences[0])
    alpha_set = set(alphabet)
    max_entropy = np.log2(len(alpha_set))
    counts_per_position = [Counter() for _ in range(length)]

    for seq in sequences:
        for i, ch in enumerate(seq):
            if ch not in alpha_set:
                continue
            counts_per_position[i][ch] += 1

    perH = np.zeros(length, dtype=float)
    for i in range(length):
        cdict = counts_per_position[i]
        total = sum(cdict.values())
        H = 0.0
        for ch, cnt in cdict.items():
            p = cnt / total
            H -= p * np.log2(p + eps) # Entropy = - sigma_{i}_{log(pi) * pi}
        perH[i] = H

    mean_entropy = float(perH.mean())
    perH_norm = perH / max_entropy if max_entropy > 0 else 0.0
    normalized_mean_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "per_position_entropy": perH,  # np.ndarray [L]
        "norm_position_entropy": perH_norm,  # H₁(pos)/max
        "mean_entropy": mean_entropy,  # 平均位置熵
        "max_entropy": max_entropy,  # log2(K)
        "normalized_mean_entropy": normalized_mean_entropy,  # 0~1
        "counts_per_position": [dict(c) for c in counts_per_position],
        "alphabet": alphabet,
    }


def plot_amino_constraint_tripanel(seqs, amino, amino_pos, savepath=None, title=None):
    seqs = [dna_to_rna(s) for s in seqs]
    L = len(seqs[0])
    bases = ['A','C','G','U']

    freq = np.zeros((L,4))
    for i in range(L):
        for s in seqs:
            b = s[i]
            if b in bases:
                freq[i, bases.index(b)] += 1
    freq /= len(seqs)
    df = pd.DataFrame(freq, columns=bases)

    entropy_result = compute_shannon_entropy_base_and_amino(seqs, amino, amino_pos_list=amino_pos)
    H_final = entropy_result["normalized_entropy"]

    n_amino = len(amino)
    ncols = min(4, max(n_amino, 1))
    nrows = max(1, math.ceil(n_amino / ncols))

    fig = plt.figure(figsize=(figure_size[0], 5.0 + 1.8 * nrows))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)
    # === NEW: 3-row layout: LOGO → PIE → ENTROPY ===
    gs_outer = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.0, max(1.2, 1.15 * nrows), 1.0],
        hspace=0.32,
    )

    gs_logo    = gs_outer[0].subgridspec(1, 1)
    gs_pies    = gs_outer[1].subgridspec(nrows, ncols, wspace=0.25, hspace=0.30)
    gs_entropy = gs_outer[2].subgridspec(1, 1)

    ax_logo = fig.add_subplot(gs_logo[0,0])
    ax_H    = fig.add_subplot(gs_entropy[0,0], sharex=ax_logo)

    # ===== LOGO =====
    logomaker.Logo(df, ax=ax_logo)
    ax_logo.set_ylabel("Probability", fontsize=tick_fontsize)
    ax_logo.set_xlabel("")
    ax_logo.set_xticks([])
    ax_logo.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_logo.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_logo.tick_params(axis="y", labelsize=clabel_fontsize)

    # ===== PIE CHARTS =====
    min_pct_for_label = 10.0

    for idx, (aa, p) in enumerate(zip(amino, amino_pos)):
        ax_pie = fig.add_subplot(gs_pies[idx // ncols, idx % ncols])

        labels, counts, invalid_count = _amino_codon_counts_with_invalid(seqs, aa, p)
        total = sum(counts)
        percentages = [c/total*100 for c in counts]
        wedge_labels = [
            label if label == "Invalid" or pct >= min_pct_for_label else ""
            for label, pct in zip(labels, percentages)
        ]

        wedges, texts, autotexts = ax_pie.pie(
            counts,
            labels=wedge_labels,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= min_pct_for_label else "",
            startangle=90,
            counterclock=False,
            labeldistance=1.1,
            pctdistance=0.7,
            textprops={"fontsize": max(clabel_fontsize-1,6)},
        )
        for t in autotexts:
            t.set_color("white")

        valid_count = total - invalid_count
        ax_pie.set_title(
            f"amino: {aa} (valid {valid_count}/{total})",
            fontsize=clabel_fontsize,
            fontweight="bold",
        )
        ax_pie.axis("equal")

    for empty_idx in range(n_amino, nrows * ncols):
        ax_empty = fig.add_subplot(gs_pies[empty_idx // ncols, empty_idx % ncols])
        ax_empty.axis("off")

    # ===== ENTROPY (moved to bottom) =====
    x = np.arange(L)
    ax_H.plot(x, H_final, color="blue", linewidth=2)

    # highlight codon regions in orange
    for p in amino_pos:
        for pp in [p, p + 1]:
                ax_H.plot([pp, pp + 1],[H_final[pp], H_final[pp + 1]], color="orange", linewidth=2)
    ax_H.set_xlabel("Position", fontsize=tick_fontsize)
    ax_H.set_ylabel("Normalized Entropy", fontsize=tick_fontsize)
    ax_H.set_xlim(0,L)
    ax_H.set_ylim(0,1.0)
    ax_H.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_H.set_xticks(np.arange(0, L+1, 10))
    ax_H.tick_params(labelsize=clabel_fontsize)

    blue_line = mlines.Line2D([], [], color='blue', label='Base (1-mer entropy)')
    orange_line = mlines.Line2D([], [], color='orange', label='Codon (3-mer entropy)')
    ax_H.legend(handles=[blue_line, orange_line], fontsize=clabel_fontsize)
    ax_H.set_title('Position-wise Normalized Shannon Entropy', fontsize=tick_fontsize)

    # Big Title
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize, y=0.99)

    # === Layout Adjustment ===
    plt.savefig(savepath, dpi=300)
    print(f"[saved] {savepath}")
    plt.close()




def plot_codon_constraint_duopanel(
    seqs,
    codon_pos=None,
    savepath=None,
    title=None,
    nucleotide_regions=None,
):
    """
    Duopanel visualization for nucleotide-specific RePaint experiments:
      Panel 1: 1-mer sequence logo
      Panel 2: position-wise 1-mer Shannon entropy

    Args:
        seqs: list of sequences (DNA or RNA)
        codon_pos: legacy list of three-nucleotide constraint starts
        nucleotide_regions: optional ``(start, length)`` regions; preferred for
            arbitrary-length nucleotide constraints
        savepath: output file path
        title: big title for the whole figure
    """

    # ---------- imports ----------
    seqs = [dna_to_rna(s) for s in seqs]
    codon_pos = codon_pos or []
    if nucleotide_regions is None:
        nucleotide_regions = [(position, 3) for position in codon_pos]

    L = len(seqs[0])
    bases = ['A', 'C', 'G', 'U']

    # ======================================================
    # 1. Compute LOGO frequency matrix
    # ======================================================
    freq = np.zeros((L, 4))
    for i in range(L):
        for s in seqs:
            b = s[i]
            if b in bases:
                freq[i, bases.index(b)] += 1
    freq /= len(seqs)

    df = pd.DataFrame(freq, columns=bases)

    # ======================================================
    # 2. Compute 1-mer Shannon entropy
    # ======================================================
    entropy_output = compute_shannon_entropy_per_pos(seqs)
    H = entropy_output["norm_position_entropy"]
    x = np.arange(L)

    # 3. Figure layout: 2 rows → LOGO + ENTROPY
    fig = plt.figure(figsize=figure_size)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.10)

    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.32)

    ax_logo = fig.add_subplot(gs[0, 0])
    ax_H    = fig.add_subplot(gs[1, 0])

    # Panel 1 — LOGO plot
    logomaker.Logo(df, ax=ax_logo)
    ax_logo.set_ylabel("Probability", fontsize=tick_fontsize)
    ax_logo.set_xlabel("")
    ax_logo.set_xticks([])

    ax_logo.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax_logo.tick_params(axis="y", labelsize=clabel_fontsize)
    ax_logo.tick_params(axis="x", bottom=False, labelbottom=False)

    # Panel 2 — 1-mer Shannon entropy (normalized)
    ax_H.plot(x, H, color="blue", linewidth=2, label="1-mer (base positions)")
    for start, length in nucleotide_regions:
        region_start = max(0, int(start))
        region_stop = min(L, region_start + int(length))
        if region_stop <= region_start:
            continue
        ax_H.axvspan(
            region_start - 0.5,
            region_stop - 0.5,
            color="orange",
            alpha=0.18,
        )
        region_x = x[region_start:region_stop]
        ax_H.plot(region_x, H[region_start:region_stop], color="orange", linewidth=5)

    ax_H.set_xlabel("Position", fontsize=tick_fontsize)
    ax_H.set_ylabel("Normalized Entropy", fontsize=tick_fontsize)
    ax_H.set_xlim(0, L)
    ax_H.set_ylim(0, 1.0)

    ax_H.set_xticks(np.arange(0, L+1, 10))
    ax_H.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax_H.tick_params(labelsize=clabel_fontsize)

    # Legend
    blue_line = mlines.Line2D([], [], color='blue', label='Base (1-mer entropy)')
    orange_line = mlines.Line2D([], [], color='orange', label='Constrained nucleotides')
    ax_H.legend(handles=[blue_line, orange_line], fontsize=clabel_fontsize)
    ax_H.set_title('Position-wise Normalized Shannon Entropy', fontsize=tick_fontsize)

    fig.suptitle(title, fontsize=title_fontsize, y=0.99) if title is not None else None

    plt.savefig(savepath, dpi=300)
    print(f"[saved] {savepath}")
    plt.close()


def plot_MRL_MFE_scatter(mrls, mfes, savepath=None, title=None, targets=None):
    plt.figure(figsize=figure_size)
    plt.scatter(mrls, mfes, s=6, alpha=0.35, label="Generated")
    plt.scatter([mrls.mean()], [mfes.mean()], s=80, marker='*', label="Mean")
    finite_targets = _finite_target_pairs(targets)
    if finite_targets:
        plt.scatter(
            [target[0] for target in finite_targets],
            [target[1] for target in finite_targets],
            s=70,
            marker='x',
            linewidths=2,
            color='black',
            label="Targets",
        )
    plt.xlabel('Predicted MRL', fontsize=label_fontsize)
    plt.ylabel('Predicted MFE', fontsize=label_fontsize)
    target_mrls = [target[0] for target in targets] if targets else []
    target_mfes = [target[1] for target in targets] if targets else []
    x_limits = _expanded_axis_limits([*mrls, *target_mrls], (2.0, 9.0))
    y_limits = _expanded_axis_limits([*mfes, *target_mfes], (-30.0, 0.0))
    plt.xlim(*x_limits)
    plt.ylim(*y_limits)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=True)
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.close()


def plot_cai_response(records, target_adaptiveness, savepath, effective_reference=None):
    """Plot achieved sequence CAI against the specified adaptiveness alpha."""

    all_data = pd.DataFrame(records).copy()
    all_data["CAI"] = pd.to_numeric(all_data["CAI"], errors="coerce")
    if "peptide_valid" in all_data:
        valid_mask = all_data["peptide_valid"].map(
            lambda value: value is True or str(value).strip().lower() == "true"
        )
    else:
        valid_mask = np.ones(len(all_data), dtype=bool)
    data = all_data[valid_mask & np.isfinite(all_data["CAI"])]

    fig, ax = plt.subplots(figsize=figure_size)
    if data.empty:
        ax.text(
            0.0,
            0.5,
            "No peptide-valid sequences with finite CAI",
            ha="center",
            va="center",
            fontsize=text_fontsize,
        )
    else:
        achieved_mean = float(data["CAI"].mean())
        achieved_std = float(data["CAI"].std(ddof=1)) if len(data) > 1 else 0.0
        grouped = (
            data.groupby(
                ["target_MRL", "target_MFE"], sort=False, dropna=False
            )["CAI"]
            .mean()
            .reset_index()
        )
        ax.bar(
            [0.0],
            [achieved_mean],
            yerr=[achieved_std],
            width=0.42,
            color="#4C8DA5",
            alpha=0.88,
            capsize=6,
            edgecolor="white",
            label="Peptide-valid sequences (mean +/- SD)",
            zorder=2,
        )
        offsets = np.linspace(-0.13, 0.13, len(grouped)) if len(grouped) > 1 else np.array([0.0])
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(grouped), 1)))
        for offset, color, (_, row) in zip(offsets, colors, grouped.iterrows()):
            ax.scatter(
                [offset],
                [row["CAI"]],
                s=65,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                label=_describe_targets(
                    [[row["target_MRL"], row["target_MFE"]]]
                ),
                zorder=4,
            )
    ax.scatter(
        [0.0],
        [target_adaptiveness],
        marker="_",
        s=320,
        linewidth=2.2,
        color="black",
        label="Specified adaptiveness alpha",
        zorder=5,
    )
    if effective_reference is not None and not math.isclose(
        float(effective_reference), float(target_adaptiveness), abs_tol=1e-9
    ):
        ax.scatter(
            [0.0],
            [effective_reference],
            marker="_",
            s=320,
            linewidth=2.2,
            color="#D1495B",
            label="Geometric mean of feasible per-position alpha values",
            zorder=5,
        )

    reference_values = [target_adaptiveness]
    if effective_reference is not None:
        reference_values.append(effective_reference)
    y_values = np.concatenate((data["CAI"].to_numpy(), np.asarray(reference_values)))
    y_low, y_high = float(y_values.min()), float(y_values.max())
    padding = max((y_high - y_low) * 0.18, 0.025)
    ax.set_ylim(max(0.0, y_low - padding), min(1.02, y_high + padding))
    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0.0], [f"alpha = {target_adaptiveness:g}"])
    ax.set_xlabel("Specified Codon Relative Adaptiveness", fontsize=label_fontsize)
    ax.set_ylabel("Generated Sequence CAI", fontsize=label_fontsize)
    ax.set_title(
        "CAI Response to Specified Codon Relative Adaptiveness\n"
        f"peptide-valid sequences: {len(data)}/{len(all_data)}",
        fontsize=title_fontsize,
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, frameon=True)
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    plt.close(fig)

def read_csv_and_plot(csv_file, args):
    data = pd.read_csv(csv_file)
    mrls, mfes = data['MRL'].to_numpy(), data['MFE'].to_numpy()
    target_pairs = _normalise_target_pairs(args)
    target_description = _describe_targets(target_pairs)
    plot_MRL_MFE_scatter(
        mrls=mrls,
        mfes=mfes,
        savepath=_derived_plot_path(args.out, '_dist.jpg'),
        title=(
            f"MRL-MFE Distribution of Generated Sequences(n={len(data)})\n"
            f"Condition: {target_description}"
        ),
        targets=target_pairs,
    )

    seqs = data['Sequence'].astype(str).tolist()
    nucleotide = _as_constraint_tokens(getattr(args, "nucleotide", None))
    amino = _as_constraint_tokens(getattr(args, "amino", None))
    cds_amino = getattr(args, "cds_amino", None)
    active_constraints = sum(bool(value) for value in (nucleotide, amino, cds_amino))
    if active_constraints > 1:
        raise ValueError(
            "--nucleotide, --amino, and --cds-amino are mutually exclusive"
        )

    if nucleotide:
        nucleotide_regions = []
        for token in nucleotide:
            try:
                position_text, sequence = token.split(":", 1)
                position = int(position_text)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid nucleotide constraint {token!r}; expected POSITION:SEQUENCE"
                ) from exc
            nucleotide_regions.append((position, len(sequence.strip())))
        plot_codon_constraint_duopanel(
            seqs=seqs,
            nucleotide_regions=nucleotide_regions,
            savepath=_derived_plot_path(args.out, '_constraint.jpg'),
            title=f"Nucleotide Constraints: {' '.join(nucleotide)}",
        )
    elif amino:
        amino_pos = [int(token.split(":", 1)[0]) for token in amino]
        amino_list = [token.split(":", 1)[1].upper() for token in amino]
        plot_amino_constraint_tripanel(
            seqs=seqs,
            amino=amino_list,
            amino_pos=amino_pos,
            savepath=_derived_plot_path(args.out, '_constraint.jpg'),
            title=f"Amino-acid Constraints: {' '.join(amino)}",
        )
    elif cds_amino:
        peptide = str(cds_amino).strip().upper()
        amino_list = list(peptide)
        amino_start = 50 - 3 * len(amino_list)
        if not amino_list or amino_start < 0:
            raise ValueError(
                "--cds-amino must contain 1-16 amino acids for a 50-nt sequence"
            )
        amino_pos = list(range(amino_start, 50, 3))
        cai = getattr(args, "cai", None)
        cai_description = (
            f"\nRequested codon relative adaptiveness alpha={float(cai):g}"
            if cai is not None
            else ""
        )
        plot_amino_constraint_tripanel(
            seqs=seqs,
            amino=amino_list,
            amino_pos=amino_pos,
            savepath=_derived_plot_path(args.out, '_constraint.jpg'),
            title=f"CDS Amino-acid Constraint: {peptide}{cai_description}",
        )
