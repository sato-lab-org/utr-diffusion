import os
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
import subprocess


# Your project imports
from Scripts.esm import Alphabet, FastaBatchedDataset
from Scripts.esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS


# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    seed: int = 1337
    layers: int = 6
    heads: int = 16
    embed_dim: int = 128
    inp_len: int = 50
    batch_toks: int = 4096 * 8
    device: str = "cuda"


def set_seed(seed: int) -> None:

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Model
# ---------------------------

def build_alphabet() -> Alphabet:

    alphabet = Alphabet(mask_prob=0.0, standard_toks='AGCT')

    assert alphabet.tok_to_idx == {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'A': 3, 'G': 4, 'C': 5, 'T': 6, '<cls>': 7, '<mask>': 8, '<sep>': 9}

    return alphabet


class CNN_linear(nn.Module):

    def __init__(
            self,
            alphabet: Alphabet,
            layers: int,
            heads: int,
            embed_dim: int,
            inp_len: int,
            border_mode: str = "same",
            filter_len: int = 8,
            nbr_filters: int = 120,
            dropout1: float = 0.0,
            dropout2: float = 0.0,
    ):
        super(CNN_linear, self).__init__()

        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = 40
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self._repr_layer = layers

        self.esm2 = ESM2_SISS(
            num_layers=layers,
            embed_dim=embed_dim,
            attention_heads=heads,
            alphabet=alphabet,
        )

        self.conv1 = nn.Conv1d(
            in_channels=self.embedding_size,
            out_channels=self.nbr_filters,
            kernel_size=self.filter_len,
            padding=self.border_mode,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.nbr_filters,
            out_channels=self.nbr_filters,
            kernel_size=self.filter_len,
            padding=self.border_mode,
        )

        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.dropout3 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
        self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
        self.output = nn.Linear(in_features=self.nodes, out_features=1)

    def forward(self, tokens, need_head_weights=True, return_contacts=False, return_representation=True):

        x = self.esm2(tokens, [self._repr_layer], need_head_weights, return_contacts, return_representation)
        x = x["representations"][self._repr_layer][:, 0]
        x_o = x.unsqueeze(2)

        x = self.flatten(x_o)
        o_linear = self.fc(x)
        o_relu = self.relu(o_linear)
        o_dropout = self.dropout3(o_relu)
        o = self.output(o_dropout)

        return o


def load_model(model_path: str, cfg: Config, alphabet: Alphabet, device: torch.device) -> nn.Module:

    model = CNN_linear(
        alphabet=alphabet,
        layers=cfg.layers,
        heads=cfg.heads,
        embed_dim=cfg.embed_dim,
        inp_len=cfg.inp_len,
    ).to(device)

    state = torch.load(model_path, map_location=device)

    # Handle DDP checkpoints: module.xxx
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


# ---------------------------
# Data IO
# ---------------------------

def normalize_seq(seq: str) -> str:

    seq = seq.upper().replace("U", "T")

    return seq


def read_fasta_file(fasta_path: str, inp_len: int) -> Tuple[List[str], List[str]]:

    ids: List[str] = []
    seqs: List[str] = []

    with open(fasta_path, "r", encoding="utf-8") as f:

        for record in SeqIO.parse(f, "fasta"):

            seq = normalize_seq(str(record.seq))
            seq = seq[-inp_len:]

            if not set(seq).issubset(set("AGCT")):
                continue

            ids.append(record.id)
            seqs.append(seq)

    return ids, seqs


def build_dataloader(ids: List[str], seqs: List[str], alphabet: Alphabet, batch_toks: int):

    dataset = FastaBatchedDataset(ids, seqs, mask_prob=0.0)
    batches = dataset.get_batch_indices(toks_per_batch=batch_toks, extra_toks_per_seq=1)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(),
        batch_sampler=batches,
        shuffle=False,
    )

    return dataloader


# ---------------------------
# Prediction
# ---------------------------

@torch.no_grad()
def predict_mrl(dataloader, model: nn.Module, device: torch.device) -> pd.DataFrame:

    ids_all: List[str] = []
    seqs_all: List[str] = []
    preds_all: List[float] = []

    for ids, strs, _, toks, _, _ in dataloader:

        ids_all.extend(ids)
        seqs_all.extend(strs)

        toks = toks.to(device)
        logits = model(toks, return_representation=True, return_contacts=True)
        logits = logits.reshape(-1).detach().cpu().tolist()

        preds_all.extend(logits)

    df = pd.DataFrame({"ID": ids_all, "Sequence": seqs_all, "MRL": preds_all})

    return df


def get_mfe_batch(sequences, rnafold_path: str) -> List[float]:

    input_bytes = ("\n".join(sequences) + "\n").encode("utf-8")

    res = subprocess.run(
        [rnafold_path],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    lines = res.stdout.decode("utf-8").strip().splitlines()

    mfe_list: List[float] = []

    for i in range(1, len(lines), 2):

        mfe_line = lines[i]
        mfe = float(mfe_line.split("(")[-1].split(")")[0])
        mfe_list.append(mfe)

    if len(mfe_list) != len(sequences):

        raise RuntimeError(
            f"RNAfold parsed {len(mfe_list)} MFEs but got {len(sequences)} sequences. "
            f"stdout lines={len(lines)}"
        )

    return mfe_list


def predict_mfe(df: pd.DataFrame, rnafold_path: str, batch_size: int = 100) -> pd.DataFrame:

    if "Sequence" not in df.columns:
        raise KeyError("df must contain 'Sequence' column to compute MFE.")

    sequences = df["Sequence"].astype(str).tolist()
    mfe_all: List[float] = []

    for i in range(0, len(sequences), batch_size):

        seq_batch = sequences[i : i + batch_size]
        mfe_batch = get_mfe_batch(seq_batch, rnafold_path=rnafold_path)
        mfe_all.extend(mfe_batch)

        print(f"[MFE] {min(i + batch_size, len(sequences))}/{len(sequences)}", flush=True)

    df = df.copy()
    df["MFE"] = mfe_all

    return df


# ---------------------------
# FASTA collection
# ---------------------------

def collect_fasta_files(input_path: str, mode: str):

    input_path = Path(input_path)
    fasta_suffixes = [".fasta", ".fa", ".fna"]

    if mode == "file":

        if not input_path.exists():
            raise FileNotFoundError(f"Input fasta file not found: {input_path}")

        if input_path.suffix.lower() not in fasta_suffixes:
            raise ValueError(f"Input file is not fasta/fa/fna: {input_path}")

        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_path}")

    fasta_files = []

    if mode == "folder":

        for f in input_path.iterdir():

            if f.is_file() and f.suffix.lower() in fasta_suffixes:
                fasta_files.append(f)

    elif mode == "subfolders":

        for subdir in sorted(input_path.iterdir()):

            if not subdir.is_dir():
                continue

            for f in subdir.iterdir():

                if f.is_file() and f.suffix.lower() in fasta_suffixes:
                    fasta_files.append(f)

    elif mode == "recursive":

        for f in input_path.rglob("*"):

            if f.is_file() and f.suffix.lower() in fasta_suffixes:
                fasta_files.append(f)

    else:

        raise ValueError(f"Unknown mode: {mode}")

    fasta_files = sorted(fasta_files)

    return fasta_files


# ---------------------------
# Processing
# ---------------------------

def process_single_fasta(
        fasta_path: str,
        model: nn.Module,
        cfg: Config,
        alphabet: Alphabet,
        device: torch.device,
        rnafold_path: str,
        mfe_batch: int,
        overwrite: bool = False,
):

    fasta_path = Path(fasta_path)
    out_file = fasta_path.with_suffix(".csv")

    if out_file.exists() and not overwrite:

        print(f"[SKIP] Output already exists: {out_file}")
        return

    print(f"\n[INFO] Processing: {fasta_path}")

    ids, seqs = read_fasta_file(str(fasta_path), inp_len=cfg.inp_len)

    if len(seqs) == 0:

        print(f"[WARNING] No valid sequences found in {fasta_path}, skipped.")
        return

    dataloader = build_dataloader(ids, seqs, alphabet, batch_toks=cfg.batch_toks)

    df = predict_mrl(dataloader, model, device=device)
    df = predict_mfe(df, rnafold_path=rnafold_path, batch_size=mfe_batch)

    df.to_csv(out_file, index=False)

    print(f"[OK] Predicted {len(df)} sequences -> {out_file}")


def process_fasta_files(
        fasta_files,
        model: nn.Module,
        cfg: Config,
        alphabet: Alphabet,
        device: torch.device,
        rnafold_path: str,
        mfe_batch: int,
        overwrite: bool = False,
):

    print(f"[INFO] Found {len(fasta_files)} fasta files")

    if len(fasta_files) == 0:

        print("[WARNING] No fasta files found.")
        return

    for idx, fasta_file in enumerate(fasta_files, start=1):

        print(f"\n========== [{idx}/{len(fasta_files)}] ==========")

        try:

            process_single_fasta(
                fasta_path=str(fasta_file),
                model=model,
                cfg=cfg,
                alphabet=alphabet,
                device=device,
                rnafold_path=rnafold_path,
                mfe_batch=mfe_batch,
                overwrite=overwrite,
            )

        except Exception as e:

            print(f"[ERROR] Failed on {fasta_file}")
            print(e)


# ---------------------------
# CLI
# ---------------------------

def parse_args():

    p = argparse.ArgumentParser(description="UTRLM predictor for UTR-Diffusion evaluation: MRL + MFE.")

    p.add_argument("--input", required=True, help="Input fasta file or folder.")
    p.add_argument("--mode", required=True, choices=["file", "folder", "subfolders", "recursive"])

    p.add_argument("--model", default="Model/model.pt", help="Model checkpoint path.")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--inp-len", type=int, default=50, help="Sequence length used for inference.")
    p.add_argument("--batch-toks", type=int, default=4096 * 8, help="Tokens per batch.")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--rnafold-path", default="RNAfold", help="Path to RNAfold binary.")
    p.add_argument("--mfe-batch", type=int, default=100, help="Batch size for RNAfold MFE computation.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing csv files.")

    return p.parse_args()


if __name__ == "__main__":

    args = parse_args()

    cfg = Config(
        seed=args.seed,
        inp_len=args.inp_len,
        batch_toks=args.batch_toks,
        device=args.device,
    )

    set_seed(cfg.seed)

    if "cuda" in cfg.device and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    print(f"[INFO] Device: {device}")

    alphabet = build_alphabet()
    model = load_model(args.model, cfg, alphabet, device=device)

    fasta_files = collect_fasta_files(args.input, mode=args.mode)

    process_fasta_files(
        fasta_files=fasta_files,
        model=model,
        cfg=cfg,
        alphabet=alphabet,
        device=device,
        rnafold_path=args.rnafold_path,
        mfe_batch=args.mfe_batch,
        overwrite=args.overwrite,
    )