import os
import pickle
import random
from os.path import split
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from numpy.random import gumbel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.utils.utils import one_hot_encode, one_hot_encode_zero_to_neg
from sklearn.model_selection import train_test_split

nucleotides = ["A", "C", "G", "T"]

def load_data(data_path: str = "", split_ratio: float = 0.0, label_type="MRL_label"):
    # Preprocessing data
    data = pd.read_csv(data_path) if data_path.endswith(".csv") else torch.load(data_path)

    if split_ratio == 0:
        train_data = data
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])

        encode_data_dict = {
            "Train": X_train,
            "Train_label": torch.tensor(train_data[label_type].to_numpy(), dtype=torch.int8),
            'Classes': data[label_type].nunique()
        }

    else:
        train_data, valid_data = train_test_split(data , test_size=split_ratio, random_state=42)
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        x_valid_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in valid_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        X_valid = np.array([x.T.tolist() for x in x_valid_encode])

        encode_data_dict = {
            "Train": X_train,
            "Valid": X_valid,
            "Train_label": torch.tensor(train_data[label_type].to_numpy(), dtype=torch.int8),
            "Valid_label": torch.tensor(valid_data[label_type].to_numpy(), dtype=torch.int8),
            'Classes': data[label_type].nunique()
        }

    print_data_info(encode_data_dict)

    return encode_data_dict


def load_data_continues(data_path: str = "", split_ratio: float = 0.0, label: str="MRL"):
    # Preprocessing data
    data = pd.read_csv(data_path) if data_path.endswith(".csv") else torch.load(data_path)

    if split_ratio == 0:
        train_data = data
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])

        encode_data_dict = {
            "Train": X_train,
            "Train_label": torch.tensor(train_data[label].to_numpy(), dtype=torch.float32),
            "Classes": 1
        }

    else:
        train_data, valid_data = train_test_split(data , test_size=split_ratio, random_state=42)
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        x_valid_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in valid_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        X_valid = np.array([x.T.tolist() for x in x_valid_encode])

        encode_data_dict = {
            "Train": X_train,
            "Valid": X_valid,
            "Train_label": torch.tensor(train_data[label].to_numpy(), dtype=torch.float32),
            "Valid_label": torch.tensor(valid_data[label].to_numpy(), dtype=torch.float32),
            "Classes": 1
        }

    print_data_info_c(encode_data_dict)

    return encode_data_dict

def load_data_MRL_MFE_double_label(data_path: str = "", split_ratio: float = 0.0):
    data = pd.read_csv(data_path) if data_path.endswith(".csv") else torch.load(data_path)

    # 获取 label 数目（用于 model 初始化）
    n_class_mrl = data['MRL_label'].nunique()
    n_class_mfe = data['MFE_label'].nunique()

    if split_ratio == 0:
        train_data = data
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        labels = train_data[['MRL_label', 'MFE_label']].to_numpy(dtype=np.int64)  # shape [B, 2]

        encode_data_dict = {
            "Train": X_train,
            "Train_label": torch.tensor(labels, dtype=torch.long),
            "Classes": [n_class_mrl, n_class_mfe]
        }

    else:
        train_data, valid_data = train_test_split(data, test_size=split_ratio, random_state=42)

        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x ])
        x_valid_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in valid_data['Sequence'] if "N" not in x ])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        X_valid = np.array([x.T.tolist() for x in x_valid_encode])

        train_labels = train_data[['MRL_label', 'MFE_label']].to_numpy(dtype=np.int64)
        valid_labels = valid_data[['MRL_label', 'MFE_label']].to_numpy(dtype=np.int64)

        encode_data_dict = {
            "Train": X_train,
            "Valid": X_valid,
            "Train_label": torch.tensor(train_labels, dtype=torch.long),   # [B, 2]
            "Valid_label": torch.tensor(valid_labels, dtype=torch.long),   # [B, 2]
            "Classes": [n_class_mrl, n_class_mfe]
        }

    print_data_info_double_label(encode_data_dict)

    return encode_data_dict

def load_data_continues_MRL_MFE_double_label(data_path: str = "", label_cols:list=None, split_ratio: float = 0.0):
    # Preprocessing data
    label_cols = ['MRL', 'MFE'] if label_cols is None else label_cols
    data = pd.read_csv(data_path) if data_path.endswith(".csv") else torch.load(data_path)

    if split_ratio == 0:
        train_data = data
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])

        labels = data[label_cols].to_numpy(dtype=np.float32)

        encode_data_dict = {
            "Train": X_train,
            "Train_label": torch.tensor(labels, dtype=torch.float32),
            "Classes": 1
        }

    else:
        train_data, valid_data = train_test_split(data , test_size=split_ratio, random_state=42)
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        x_valid_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in valid_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        X_valid = np.array([x.T.tolist() for x in x_valid_encode])

        train_labels = train_data[label_cols].to_numpy(dtype=np.float32)
        valid_labels = valid_data[label_cols].to_numpy(dtype=np.float32)

        encode_data_dict = {
            "Train": X_train,
            "Valid": X_valid,
            "Train_label": torch.tensor(train_labels, dtype=torch.float32),
            "Valid_label": torch.tensor(valid_labels, dtype=torch.float32),
            "Classes": 1
        }

    print_data_info_c(encode_data_dict)

    return encode_data_dict


def load_multiple_dataset_continuous_labels(
    data_paths: list[str],
    label_cols: list = None,
    split_ratio: float = 0.0,
    seq_len: int = 50,
):
    """
    Load and concatenate multiple sequence datasets with continuous partial labels.

    For each dataset:
        1. Read CSV or torch file.
        2. Keep only 'Sequence' and label_cols.
        3. Add missing label columns as NaN.

    After concatenation:
        4. Remove sequences containing 'N'.
        5. Split into training and validation sets.
        6. One-hot encode sequences.

    Args:
        data_paths:
            List of dataset paths.

        label_cols:
            Complete label column list expected by the model.
            Missing label columns in each dataset are automatically filled with NaN.

        split_ratio:
            Validation split ratio. Set to 0 for training data only.

        seq_len:
            Sequence length used for one-hot encoding.

    Returns:
        Dictionary containing encoded sequences and label tensors.
    """

    label_cols = ["MRL", "MFE"] if label_cols is None else label_cols
    datasets = []

    # Continuously read each dataset
    for data_path in data_paths:
        data = pd.read_csv(data_path) if data_path.endswith(".csv") else torch.load(data_path)
        for label_col in label_cols:
            if label_col not in data.columns:
                data[label_col] = np.nan
        data = data[["Sequence"] + label_cols].copy()
        datasets.append(data)

    # Concatenate all datasets
    data = pd.concat(datasets, axis=0, ignore_index=True,)

    # Remove invalid sequences before splitting and encoding
    data = data[~data["Sequence"].str.contains("N", na=True)].reset_index(drop=True)

    if split_ratio == 0:
        train_data = data
        x_train_encode = np.array([one_hot_encode_zero_to_neg(sequence, nucleotides, seq_len,) for sequence in train_data["Sequence"]])
        X_train = np.array([encoded_sequence.T.tolist() for encoded_sequence in x_train_encode])

        train_labels = train_data[label_cols].to_numpy(dtype=np.float32)

        encode_data_dict = {
            "Train": X_train,
            "Train_label": torch.tensor(train_labels, dtype=torch.float32,),
            "Classes": 1,
        }

    else:
        train_data, valid_data = train_test_split(data, test_size=split_ratio, random_state=42,)
        train_data = train_data.reset_index(drop=True)
        valid_data = valid_data.reset_index(drop=True)
        x_train_encode = np.array([one_hot_encode_zero_to_neg(sequence, nucleotides, seq_len,) for sequence in train_data["Sequence"]])
        x_valid_encode = np.array([one_hot_encode_zero_to_neg(sequence, nucleotides, seq_len,) for sequence in valid_data["Sequence"]])
        X_train = np.array([encoded_sequence.T.tolist() for encoded_sequence in x_train_encode])
        X_valid = np.array([encoded_sequence.T.tolist() for encoded_sequence in x_valid_encode])

        train_labels = train_data[label_cols].to_numpy(dtype=np.float32)
        valid_labels = valid_data[label_cols].to_numpy(dtype=np.float32)

        encode_data_dict = {
            "Train": X_train,
            "Valid": X_valid,
            "Train_label": torch.tensor(train_labels, dtype=torch.float32,),
            "Valid_label": torch.tensor(valid_labels, dtype=torch.float32,),
            "Classes": 1,
        }

    print_data_info_c(encode_data_dict)

    return encode_data_dict


def load_data_without_dummy_label(data_path: str = "", split_ratio: float = 0.0, seq_length: int = 50):
    data = pd.read_csv(data_path) if data_path.endswith(".csv") else torch.load(data_path)

    # Train only (no split)
    if split_ratio == 0:
        train_data = data
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, seq_length) for x in train_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        dummy_train_labels = torch.zeros(len(X_train), dtype=torch.float32)

        encode_data_dict = {
            "Train": X_train,
            "Train_label": dummy_train_labels,
            "Classes": 1
        }

    # Train + Validation split
    else:
        train_data, valid_data = train_test_split(data, test_size=split_ratio, random_state=42)
        x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in train_data['Sequence'] if "N" not in x])
        x_valid_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, 50) for x in valid_data['Sequence'] if "N" not in x])
        X_train = np.array([x.T.tolist() for x in x_train_encode])
        X_valid = np.array([x.T.tolist() for x in x_valid_encode])

        dummy_train_labels = torch.zeros(len(X_train), dtype=torch.float32)
        dummy_valid_labels = torch.zeros(len(X_valid), dtype=torch.float32)

        encode_data_dict = {
            "Train": X_train,
            "Valid": X_valid,
            "Train_label": dummy_train_labels,
            "Valid_label": dummy_valid_labels,
            "Classes": 1
        }

    print_data_info(encode_data_dict)
    return encode_data_dict

def load_data_with_label_name(
        datapath: str = "",
        seq_col: str= 'gs.sequence',
        labels:list[str] = None,
        max_seq_len:int = None,
        split_ratio: float = 0.05,
        do_normalize: bool = False,
        log2_transform_labels: list[str] = None, # ['ss.rna.dna.mean']
):
    # Preprocessing data
    data = pd.read_csv(datapath)
    data = data[[seq_col] + labels].copy()
    data[seq_col] = data[seq_col].str.upper()
    max_seq_len = data[seq_col].str.len().max() if max_seq_len is None else max_seq_len
    check_and_print_missing_labels(data, labels)

    if do_normalize:
        for label in labels:
            data[label] = _z_score_normalize(data, label, do_log2=label in log2_transform_labels)

    train_data, valid_data = train_test_split(data , test_size=split_ratio, random_state=42)
    x_train_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, max_seq_len) for x in train_data[seq_col] if "N" not in x])
    x_valid_encode = np.array([one_hot_encode_zero_to_neg(x, nucleotides, max_seq_len) for x in valid_data[seq_col] if "N" not in x])
    X_train = np.array([x.T.tolist() for x in x_train_encode])
    X_valid = np.array([x.T.tolist() for x in x_valid_encode])

    train_labels = train_data[labels].to_numpy(dtype=np.float32)
    valid_labels = valid_data[labels].to_numpy(dtype=np.float32)

    encode_data_dict = {
        "Train": X_train,
        "Valid": X_valid,
        "Train_label": torch.tensor(train_labels, dtype=torch.float32),
        "Valid_label": torch.tensor(valid_labels, dtype=torch.float32),
    }

    print_data_info_c(encode_data_dict)
    return encode_data_dict


def print_data_info(data):
    print('Training data:')
    for i in range(data['Classes']):
        count = (data['Train_label'] == i + 1).sum()
        print(f'Class{i + 1}: {count}')
    if 'Valid' in data:
        print('\rValidation data:')
        for i in range(data['Classes']):
            count = (data['Valid_label'] == i + 1).sum()
            print(f'Class{i + 1}: {count}')


def print_data_info_c(data):
    print(f'Training data: {len(data["Train"])}')
    print(f'Validation data: {len(data["Valid"])}')


def print_data_info_double_label(data):
    print(f"[Info] Training size: {len(data['Train'])}")
    if "Valid" in data:
        print(f"[Info] Validation size: {len(data['Valid'])}")

    def print_joint_distribution(label_tensor, name):
        print(f"\n[Info] {name} label combination distribution:")
        unique_pairs, counts = torch.unique(label_tensor, dim=0, return_counts=True)
        for pair, count in zip(unique_pairs, counts):
            label_str = f"({pair[0].item()}, {pair[1].item()})"
            print(f"    Label={label_str}: {count.item()} samples")

    print_joint_distribution(data['Train_label'], name="Train")
    if "Valid_label" in data:
        print_joint_distribution(data['Valid_label'], name="Valid")


def motifs_from_fasta(fasta: str):
    print("Computing Motifs....")
    os.system(f"gimme scan {fasta} -p  JASPAR2020_vertebrates -g hg38 -n 20> train_results_motifs.bed")
    df_results_seq_guime = pd.read_csv("train_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_seq_guime["motifs"] = df_results_seq_guime[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])

    df_results_seq_guime[0] = df_results_seq_guime[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_results_seq_guime_count_out = df_results_seq_guime[[0, "motifs"]].groupby("motifs").count()
    return df_results_seq_guime_count_out


def save_fasta(df: pd.DataFrame, name: str, num_sequences: int, seq_to_subset_comp: bool = False) -> str:
    fasta_path = f"{name}.fasta"
    save_fasta_file = open(fasta_path, "w")
    num_to_sample = df.shape[0]

    # Subsetting sequences
    if num_sequences and seq_to_subset_comp:
        num_to_sample = num_sequences

    # Sampling sequences
    print(f"Sampling {num_to_sample} sequences")
    write_fasta_component = "\n".join(
        df[["dhs_id", "sequence", "TAG"]]
        .head(num_to_sample)
        .apply(lambda x: f">{x[0]}_TAG_{x[2]}\n{x[1]}", axis=1)
        .values.tolist()
    )
    save_fasta_file.write(write_fasta_component)
    save_fasta_file.close()

    return fasta_path


def generate_motifs_and_fastas(
    df: pd.DataFrame, name: str, num_sequences: int, subset_list: list | None = None
) -> dict[str, Any]:
    print("Generating Motifs and Fastas...", name)
    print("---" * 10)

    # Saving fasta
    if subset_list:
        fasta_path = save_fasta(df, f"{name}_{'_'.join([str(c) for c in subset_list])}", num_sequences)
    else:
        fasta_path = save_fasta(df, name, num_sequences)

    # Computing motifs
    motifs = motifs_from_fasta(fasta_path)

    # Generating subset specific motifs
    final_subset_motifs = {}
    for comp, v_comp in df.groupby("TAG"):
        print(comp)
        c_fasta = save_fasta(v_comp, f"{name}_{comp}", num_sequences, seq_to_subset_comp=True)
        final_subset_motifs[comp] = motifs_from_fasta(c_fasta)

    return {
        "fasta_path": fasta_path,
        "motifs": motifs,
        "final_subset_motifs": final_subset_motifs,
        "df": df,
    }


def preprocess_data(
    input_csv: str,
    subset_list: list | None = None,
    limit_total_sequences: int | None = None,
    number_of_sequences_to_motif_creation: int = 1000,
    save_output: bool = True,
):
    # Reading the csv file
    df = pd.read_csv(input_csv, sep="\t")

    # Subsetting the dataframe
    if subset_list:
        print(" or ".join([f"TAG == {c}" for c in subset_list]))
        df = df.query(" or ".join([f'TAG == "{c}" ' for c in subset_list]))
        print("Subsetting...")

    # Limiting the total number of sequences
    if limit_total_sequences > 0:
        print(f"Limiting total sequences to {limit_total_sequences}")
        df = df.sample(limit_total_sequences)

    # Creating train/test/shuffle groups
    df_test = df[df["chr"] == "chr1"].reset_index(drop=True)
    df_train_shuffled = df[df["chr"] == "chr2"].reset_index(drop=True)
    df_train = df_train = df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(drop=True)

    df_train_shuffled["sequence"] = df_train_shuffled["sequence"].apply(
        lambda x: "".join(random.sample(list(x), len(x)))
    )

    # Getting motif information from the sequences
    train = generate_motifs_and_fastas(df_train, "train", number_of_sequences_to_motif_creation, subset_list)
    test = generate_motifs_and_fastas(df_test, "test", number_of_sequences_to_motif_creation, subset_list)
    train_shuffled = generate_motifs_and_fastas(
        df_train_shuffled,
        "train_shuffled",
        number_of_sequences_to_motif_creation,
        subset_list,
    )

    combined_dict = {"train": train, "test": test, "train_shuffled": train_shuffled}

    # Writing to pickle
    if save_output:
        # Saving all train, test, train_shuffled dictionaries to pickle
        with open("src/src/data/encode_data.pkl", "wb") as f:
            pickle.dump(combined_dict, f)

    return combined_dict

def gumbel_softmax(logits, scale=1.0, tau=1.0, hard=False):
    """
    :param logits: normalized data
    :param scale: make it larger
    :param tau: softmax smoothness and similiarity
    :param hard: inferencing
    """
    # sample Gumbel noise
    gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    y_soft = F.softmax((logits * scale + gumbels) / tau, dim=-1)  # 计算 softmax

    if hard:
        y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        return (y_hard - y_soft).detach() + y_soft  # y_hardの勾配を切り離す
    else:
        return y_soft

class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor,
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image = self.seqs[index]

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y


def check_and_print_missing_labels(data, labels):
    """
    Check and print:
    1. missing count for each label column
    2. row-wise missing distribution
    3. number of rows with exactly 1 / 2 / 3 missing labels
    4. missing label combinations for rows with exactly 1 / 2 / 3 missing labels

    Args:
        data: pandas DataFrame
        labels: list[str]
    """
    import pandas as pd

    print("=" * 60)
    print("CHECK MISSING LABELS")
    print("=" * 60)

    # 1) per-label missing count
    print("\n[1] Missing count for each label:")
    for label in labels:
        num_missing = data[label].isna().sum()
        num_total = len(data[label])
        ratio = num_missing / num_total if num_total > 0 else 0.0
        print(f"{label}: {num_missing}/{num_total} missing ({ratio:.2%})")

    # 2) row-wise missing mask and missing count
    missing_mask = data[labels].isna()              # [num_samples, num_labels]
    missing_count_per_row = missing_mask.sum(axis=1)

    print("\n[2] Row-wise missing-count distribution:")
    print(missing_count_per_row.value_counts().sort_index())

    # 3) exactly k missing
    for k in [1, 2, 3]:
        num_k_missing = (missing_count_per_row == k).sum()
        print(f"\n[3.{k}] Rows with exactly {k} missing label(s): {num_k_missing}")

        # 4) show which label(s) are missing
        rows_k = missing_mask[missing_count_per_row == k]

        if len(rows_k) == 0:
            print(f"No rows with exactly {k} missing label(s).")
            continue

        combo_counts = rows_k.apply(
            lambda row: tuple(row.index[row].tolist()),
            axis=1
        ).value_counts()

        print(f"Missing combination counts for exactly {k} missing:")
        print(combo_counts)

    print("\nDone.")

import os
from pathlib import Path
import pandas as pd
import yaml


def _print_label_stats(x: pd.Series, label_name: str, stage: str):
    """
    Print summary statistics of a label series.
    """
    valid = x.dropna()
    print(
        f"[{stage}] {label_name} | "
        f"count={len(valid)}, "
        f"min={valid.min():.6f}, "
        f"max={valid.max():.6f}, "
        f"mean={valid.mean():.6f}, "
        f"std={valid.std():.6f}"
    )


def _save_violin_plot(
    raw_values: pd.Series,
    processed_values: pd.Series,
    label_name: str,
    save_dir: str | Path,
    suffix: str = ""
):
    """
    Save violin plots for raw and normalized distributions.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_valid = raw_values.dropna().values
    processed_valid = processed_values.dropna().values

    # raw
    plt.figure(figsize=(10, 6))
    plt.violinplot(raw_valid, showmeans=True, showextrema=True)
    plt.xticks([1], [label_name])
    plt.ylabel("value")
    plt.title(f"{label_name} raw distribution")
    raw_path = save_dir / f"{label_name}_raw_dist{suffix}.png"
    plt.tight_layout()
    plt.savefig(raw_path, dpi=300)
    plt.close()

    # normalized
    plt.figure(figsize=(10, 6))
    plt.violinplot(processed_valid, showmeans=True, showextrema=True)
    plt.xticks([1], [label_name])
    plt.ylabel("value")
    plt.title(f"{label_name} normalized distribution")
    norm_path = save_dir / f"{label_name}_normalized_dist{suffix}.png"
    plt.tight_layout()
    plt.savefig(norm_path, dpi=300)
    plt.close()

    print(f"Saved violin plots:\n  raw: {raw_path}\n  normalized: {norm_path}")


def _z_score_normalize(
    data: pd.Series,
    label: str,
    do_log2: bool = False,
    save_dir: str | Path = "data/E_Coli/label_dist",
    eps: float = 1e-12,
):
    """
    Normalize a label series with optional log2 transform.

    Steps:
    1. print raw stats
    2. optional log2 transform
    3. z-score normalization
    4. print normalized stats
    5. optional save violin plots

    Args:
        x: pd.Series of one label column
        label: label name for logging / plotting
        do_log2: whether to apply log2 transform before normalization
        save_plot: whether to save violin plots
        save_dir: directory to save plots
        eps: small value to avoid division by zero

    Returns:
        normalized_x: pd.Series
        stats: dict containing transform parameters
    """
    x = data[label].copy()

    print("=" * 80)
    print(f"Processing label: {label}")
    _print_label_stats(x, label, stage="raw")

    raw_x_for_plot = x.copy()

    # log2 transform if needed
    if do_log2:
        non_nan_mask = x.notna()

        # safety check
        if (x[non_nan_mask] <= 0).any():
            raise ValueError(f"Label '{label}' contains non-positive values, cannot apply log2.")

        x.loc[non_nan_mask] = np.log2(x.loc[non_nan_mask])

    # z-score normalization using valid values only
    valid_mask = x.notna()
    valid_x = x.loc[valid_mask]

    mean_val = valid_x.mean()
    std_val = valid_x.std()
    min_before_norm, max_before_norm = valid_x.min(), valid_x.max()
    if std_val < eps:
        print(
            f"[Warning] std of label '{label}' is too small ({std_val:.6e}). "
            f"Using std=1.0 to avoid division by zero."
        )
        std_val = 1.0

    x.loc[valid_mask] = (x.loc[valid_mask] - mean_val) / std_val
    min_after_norm, max_after_norm = x.loc[valid_mask].min(), x.loc[valid_mask].max()
    _print_label_stats(x, label, stage="normalized")

    stats = {
        "label_name": label,
        "do_log2": do_log2,
        "min_before_norm": round(float(min_before_norm), 4),
        "max_before_norm": round(float(max_before_norm), 4),
        "mean": round(float(mean_val), 4),
        "std": round(float(std_val), 4),
        "min_after_norm": round(float(min_after_norm), 4),
        "max_after_norm": round(float(max_after_norm), 4),
        "count_valid": int(valid_mask.sum()),
        "count_nan": int((~valid_mask).sum()),
    }

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save stats to yaml
    norm_info_path = save_dir / "norm_info.yaml"
    if norm_info_path.exists():
        with open(norm_info_path, "r", encoding="utf-8") as f:
            norm_info = yaml.safe_load(f) or {}
    else:
        norm_info = {}

    norm_info[label] = stats

    with open(norm_info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(norm_info, f, allow_unicode=True, sort_keys=False)

    suffix = "_log2" if do_log2 else ""

    raw_fig_path = save_dir / f"{label}_raw_dist{suffix}.png"
    norm_fig_path = save_dir / f"{label}_normalized_dist{suffix}.png"

    # save plots only if one of them does not exist
    if not (raw_fig_path.exists() and norm_fig_path.exists()):
        _save_violin_plot(
            raw_values=raw_x_for_plot,
            processed_values=x,
            label_name=label,
            save_dir=save_dir,
            suffix=suffix,
        )

    return x

if __name__ == '__main__':
    nucleotides = ["A", "C", "G", "T"]
    base = 'ACCGTGAACG'+'ACCGTGAACG'+'ACCGTGAACG'+'ACCGTGAACG'+'ACCGTGAACG'

    x = torch.tensor(one_hot_encode_zero_to_neg(base, nucleotides, max_seq_len=50))
    tau = 1
    total_number = 1000
    print(f'Gumbel softmax with tau = {tau}')

    # print(f'Input:\nBase: {base}, One-hot: {x}\n')
    # print('Output:')
    # np.set_printoptions(suppress=True, precision=6)
    x0 = x
    x = x.unsqueeze(0).expand(total_number, -1, -1).to('cuda')
    tau = torch.tensor([tau]).to('cuda')
    x_G = gumbel_softmax(x, scale=3, tau=tau, hard=False)
    idx = torch.argmax(x_G, dim=-1)
    re_base = np.array(nucleotides)[idx.cpu().numpy()]
    # print(f'base: {re_base}, Onehot: {np.array(x_G)}')
    pred_idx = idx.cpu().numpy()
    true_idx = x.argmax(dim=-1).cpu().numpy()
    correct = pred_idx == true_idx
    correct = correct.all(axis=1)
    correct = correct.mean()
    print(f'Correct rate: {correct}')

    #trial hard mode
    G_0 = x_G[0]
    x_re = gumbel_softmax(G_0, tau=tau, hard=True)
