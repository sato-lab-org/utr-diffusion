import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
import numpy as np
import pandas as pd
import logomaker
from collections import Counter
from pathlib import Path

from src.models.repaint.amino_codon_table import AA_TO_CODON_USAGE_HUMAN_RNA, CODON_TO_AMINO

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

CODON_TO_RELATIVE_ADAPTIVENESS = {}
for usage_by_codon in AA_TO_CODON_USAGE_HUMAN_RNA.values():
    maximum = max(usage_by_codon.values())
    for codon, usage in usage_by_codon.items():
        CODON_TO_RELATIVE_ADAPTIVENESS[codon] = usage / maximum

def dna_to_rna(seq: str) -> str:
    return seq.upper().replace('T', 'U')


def calculate_cai(cds: str) -> float:
    """Human CAI over the complete CDS suffix, excluding the initiating AUG."""
    cds = dna_to_rna(cds)
    if len(cds) < 6 or len(cds) % 3 != 0 or not cds.startswith('AUG'):
        raise ValueError("CAI input must be a complete CDS suffix beginning with AUG")
    weights = []
    for offset in range(3, len(cds), 3):
        codon = cds[offset:offset + 3]
        if codon not in CODON_TO_RELATIVE_ADAPTIVENESS:
            raise ValueError(f"codon is absent from the human usage table: {codon}")
        weight = CODON_TO_RELATIVE_ADAPTIVENESS[codon]
        if weight <= 0:
            raise ValueError(f"codon has non-positive relative adaptiveness: {codon}")
        weights.append(weight)
    return math.exp(sum(math.log(weight) for weight in weights) / len(weights))


def translate_rna(cds: str) -> str:
    cds = dna_to_rna(cds)
    if len(cds) % 3:
        raise ValueError(f"CDS length is not divisible by three: {len(cds)}")
    amino_acids = []
    for offset in range(0, len(cds), 3):
        codon = cds[offset:offset + 3]
        if codon not in CODON_TO_AMINO:
            raise ValueError(f"codon is absent from the human usage table: {codon}")
        amino_acids.append(CODON_TO_AMINO[codon])
    return "".join(amino_acids)


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
    figure_height = figure_size[1] if nrows == 1 else 5.0 + 1.8 * nrows

    fig = plt.figure(figsize=(figure_size[0], figure_height))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.82 if title and "\n" in title else 0.92, bottom=0.08)
    # === NEW: 3-row layout: LOGO → PIE → ENTROPY ===
    gs_outer = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, max(1.2, 1.15 * nrows), 1.0], hspace=0.32)

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

        codons = AMINO_TO_CODONS[aa]
        counts = [sum(1 for s in seqs if s[p:p+3] == c) for c in codons]
        total = sum(counts)
        if total == 0:
            counts = [1]*len(counts)
            total = sum(counts)

        percentages = [c/total*100 for c in counts]
        wedge_labels = [c if pct >= min_pct_for_label else "" for c,pct in zip(codons, percentages)]

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

        ax_pie.set_title(f"amino: {aa}", fontsize=clabel_fontsize, fontweight="bold")
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




def plot_codon_constraint_duopanel(seqs, codon_pos=None, savepath=None, title=None):
    """
    Duopanel visualization for CODON-specific RePaint experiments:
      Panel 1: 1-mer sequence logo
      Panel 2: position-wise 1-mer Shannon entropy

    Args:
        seqs: list of sequences (DNA or RNA)
        codon_pos: list of codon start positions (optional; used only for annotation highlight)
        savepath: output file path
        title: big title for the whole figure
    """

    # ---------- imports ----------
    seqs = [dna_to_rna(s) for s in seqs]

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
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84 if title and "\n" in title else 0.92, bottom=0.10)

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
    for p in codon_pos:
        for pp in [p, p + 1]:
            ax_H.plot([pp, pp + 1], [H[pp], H[pp + 1]], color="orange", linewidth=5)

    ax_H.set_xlabel("Position", fontsize=tick_fontsize)
    ax_H.set_ylabel("Normalized Entropy", fontsize=tick_fontsize)
    ax_H.set_xlim(0, L)
    ax_H.set_ylim(0, 1.0)

    ax_H.set_xticks(np.arange(0, L+1, 10))
    ax_H.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax_H.tick_params(labelsize=clabel_fontsize)

    # Legend
    blue_line = mlines.Line2D([], [], color='blue', label='Base (1-mer entropy)')
    orange_line = mlines.Line2D([], [], color='orange', label='Codon (3-mer entropy)')
    ax_H.legend(handles=[blue_line, orange_line], fontsize=clabel_fontsize)
    ax_H.set_title('Position-wise Normalized Shannon Entropy', fontsize=tick_fontsize)

    fig.suptitle(title, fontsize=title_fontsize, y=0.99) if title is not None else None

    plt.savefig(savepath, dpi=300)
    print(f"[saved] {savepath}")
    plt.close()


def plot_MRL_MFE_scatter(mrls, mfes, savepath=None, title=None):
    plt.figure(figsize=figure_size)
    plt.scatter(mrls, mfes, s=6, alpha=0.35, label="Generated")
    plt.scatter([mrls.mean()], [mfes.mean()], s=80, marker='*', label="Mean")
    plt.xlabel('Predicted MRL', fontsize=label_fontsize)
    plt.ylabel('Predicted MFE', fontsize=label_fontsize)
    plt.axis([2.0, 9, -30.0, 0])
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=True)
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    print(f"[saved] {savepath}")
    plt.close()


def plot_violin(values, target_value, label, savepath=None, title=None):
    values = np.asarray(values, dtype=float)
    target_value = float(target_value)
    if values.size == 0:
        raise ValueError(f"CSV contains no predicted {label} values")
    if not np.all(np.isfinite(values)) or not math.isfinite(target_value):
        raise ValueError(f"{label} predictions and target must be finite")
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0

    plt.figure(figsize=figure_size)
    if len(values) > 1 and np.ptp(values) > 0:
        violin = plt.violinplot([values], positions=[0], widths=0.7, showextrema=False)
        for body in violin['bodies']:
            body.set_facecolor('steelblue')
            body.set_edgecolor('steelblue')
            body.set_alpha(0.45)
    offsets = np.random.default_rng(0).uniform(-0.14, 0.14, len(values)) if len(values) > 1 else np.asarray([0.0])
    plt.scatter(offsets, values, s=14, color='navy', alpha=0.3, label='Generated sequences')
    plt.errorbar([0], [mean], yerr=[std], fmt='*', markersize=12, color='darkorange', capsize=6,
                 label=f"Mean={mean:.3f}, SD={std:.3f}", zorder=3)
    plt.axhline(target_value, color='crimson', linewidth=2, linestyle='--', label=f"Target {label}={target_value:g}")

    plt.ylabel(f"Predicted {label}", fontsize=label_fontsize)
    plt.xticks([0], ['Generated sequences'])
    plt.xlim(-0.5, 0.5)
    plt.ylim((2.0, 9.0) if label == 'MRL' else (-30.0, 0.0))
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=True)
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    print(f"[saved] {savepath}")
    plt.close()


def plot_cai_distribution(cais, target_cai=None, savepath=None, title=None):
    cais = np.asarray(cais, dtype=float)
    mean = cais.mean()
    std = cais.std(ddof=1) if len(cais) > 1 else 0.0
    plt.figure(figsize=figure_size)
    plt.bar([0], [mean], yerr=[std], width=0.5, color="steelblue", alpha=0.8, capsize=8, label=f"Mean={mean:.3f}, SD={std:.3f}")
    offsets = np.linspace(-0.18, 0.18, len(cais))
    plt.scatter(offsets, cais, s=16, color="navy", alpha=0.35, label="Generated sequences")
    if target_cai is not None:
        plt.axhline(target_cai, color="crimson", linewidth=2, linestyle="--", label=f"Specified codon adaptiveness={target_cai:.3f}")
    plt.ylabel("CAI", fontsize=label_fontsize)
    plt.xticks([0], ["Generated sequences"])
    plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1.02)
    plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    plt.legend(fontsize=legend_fontsize, frameon=True)
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    print(f"[saved] {savepath}")
    plt.close()

def read_csv_and_plot(csv_file, args):
    data = pd.read_csv(csv_file)
    mrls, mfes = data['MRL'].to_numpy(), data['MFE'].to_numpy()
    conditions = []
    if args.mrl is not None:
        conditions.append(f"MRL={args.mrl:g}")
    if args.mfe is not None:
        conditions.append(f"MFE={args.mfe:g}")
    if args.cai is not None:
        conditions.append(f"specified codon adaptiveness={args.cai:g}")
    condition_text = ", ".join(conditions) if conditions else "unconditional"
    out_path = Path(args.out)
    dist_path = out_path.with_name(f"{out_path.stem}_dist.jpg")
    constraint_path = out_path.with_name(f"{out_path.stem}_constraint.jpg")
    cai_path = out_path.with_name(f"{out_path.stem}_cai.jpg")
    if (args.mrl is not None) != (args.mfe is not None):
        label = 'MRL' if args.mrl is not None else 'MFE'
        values = mrls if args.mrl is not None else mfes
        target_value = args.mrl if args.mrl is not None else args.mfe
        plot_violin(values=values, target_value=target_value, label=label, savepath=dist_path,
                    title=f"{label} Distribution of Generated Sequences(n={len(data)})\nCondition: {condition_text}")
    else:
        plot_MRL_MFE_scatter(mrls=mrls, mfes=mfes, savepath=dist_path,
                             title=f"MRL-MFE Distribution of Generated Sequences(n={len(data)})\nCondition: {condition_text}")

    seqs = data['Sequence'].astype(str).tolist()
    if args.base:
        codon_pos = [int(x.split(":")[0]) for x in args.base]
        plot_codon_constraint_duopanel(seqs=seqs, codon_pos=codon_pos, savepath=constraint_path,
                                       title=f"Codon Constraints: {' '.join(args.base)}\nCondition: {condition_text}")
    elif args.amino:
        amino_pos  = [int(x.split(":")[0]) for x in args.amino]
        amino_list = [x.split(":")[1].upper() for x in args.amino]
        plot_amino_constraint_tripanel(seqs=seqs, amino=amino_list, amino_pos=amino_pos, savepath=constraint_path,
                                       title=f"Amino-acid Constraints: {' '.join(args.amino)}\nCondition: {condition_text}")
    elif args.cds_amino:
        amino_pos = [int(x.split(":")[0]) for x in args.cds_amino]
        amino_list = [x.split(":")[1].upper() for x in args.cds_amino]
        plot_amino_constraint_tripanel(seqs=seqs, amino=amino_list, amino_pos=amino_pos, savepath=constraint_path,
                                       title=f"CDS Amino-acid Constraint: {''.join(amino_list)} at positions {amino_pos[0]}-{amino_pos[-1]} (step 3)\nCondition: {condition_text}")
        cds_start = amino_pos[0]
        translated = [translate_rna(sequence[cds_start:]) for sequence in seqs]
        expected = "".join(amino_list)
        mismatches = [index for index, peptide in enumerate(translated) if peptide != expected]
        if mismatches:
            raise ValueError(f"{len(mismatches)} generated CDS sequences do not encode {expected}; first row: {mismatches[0]}")
        data["CAI"] = [calculate_cai(sequence[cds_start:]) for sequence in seqs]
        data.to_csv(csv_file, index=False)
        cai_stats = data["CAI"].agg(["count", "mean", "std", "min", "max"])
        print("[CAI] " + ", ".join(f"{name}={value:.4f}" for name, value in cai_stats.items()))
        plot_cai_distribution(cais=data["CAI"].to_numpy(), target_cai=args.cai,
                              savepath=cai_path, title=f"CAI Distribution of Generated Sequences(n={len(data)})\nCondition: {condition_text}")
