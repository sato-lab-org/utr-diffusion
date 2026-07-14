import math
import os
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
import shutil

def exists(x):
    return x is not None


def cycle(dl):
    while True:
        yield from dl


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def l2norm(t):
    return F.normalize(t, dim=-1)


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape, device=None):
    batch_size = t.shape[0]
    if device:
        a = a.to(device)
        t = t.to(device)

    out = a.gather(-1, t)
    result = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    if device:
        result.to(device)
    return result


def one_hot_encode(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array

def one_hot_encode_zero_to_neg(seq, alphabet, max_seq_len):
    """One-hot encode a sequence."""
    seq_len = len(seq)
    seq_array = -1 * np.ones((max_seq_len, len(alphabet)))
    for i in range(seq_len):
        seq_array[i, alphabet.index(seq[i])] = 1
    return seq_array


def encode(seq, alphabet):
    """Encode a sequence."""
    seq_len = len(seq)
    seq_array = np.zeros(len(alphabet))
    for i in range(seq_len):
        seq_array[alphabet.index(seq[i])] = 1

    return seq_array


class EMA:
    # https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py
    def __init__(self, beta: float = 0.995) -> None:
        super().__init__()
        self.beta = beta
        self.step = 0


    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        device = new.device
        old = old.to(device)
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 500) -> None:
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def linear_beta_schedule(timesteps: int, beta_end: float = 0.005):
    beta_start = 0.0001
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.001, 0.999)


def quadratic_beta_schedule(timesteps: int):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int):
    beta_start = 0.001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def convert_to_seq(x, alphabet):
    return "".join([alphabet[i] for i in np.argmax(x.reshape(4, 200), axis=0)])


def get_warmup_flatten_cosine_schedule(optimizer, num_training_steps, warmup_rate, flatten_rate, min_lambda: float=0.001):
    num_warmup_steps  = num_training_steps * warmup_rate
    num_flatten_steps = num_training_steps * flatten_rate
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(num_warmup_steps)
        if num_warmup_steps <= current_step and current_step < num_warmup_steps + num_flatten_steps:
            return 1.0
        else:
            progress = float(current_step - num_warmup_steps - num_flatten_steps) / \
                       float(max(1, num_training_steps - num_warmup_steps - num_flatten_steps))
            return max(min_lambda, math.cos(0.5 * math.pi * progress))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def write_to_fasta(sequences, folder_name, epoch=None, trial_name=None):
    os.makedirs(folder_name, exist_ok=True)
    if epoch is not None:
        filename = f'{trial_name}_epoch_{epoch}.fasta' if trial_name is not None else f'epoch_{epoch}.fasta'
    else:
        filename = f'{trial_name}.fasta' if trial_name is not None else f'sample.fasta'
    with open(os.path.join(folder_name, filename), 'w') as f:
        for seq in sequences:
            f.write(seq)


def collect_fasta_from_subdirs(target_dir):

    target_dir = Path(target_dir)

    for subdir in target_dir.iterdir():
        if not subdir.is_dir():
            continue

        fasta_files = list(subdir.glob("*.fasta"))

        if len(fasta_files) != 1:
            raise ValueError(
                f"Expected exactly one .fasta file in {subdir}, "
                f"but found {len(fasta_files)}"
            )

        fasta_path = fasta_files[0]
        new_fasta_path = target_dir / f"{subdir.name}.fasta"

        # move and rename
        shutil.move(str(fasta_path), str(new_fasta_path))

        # remove the now-empty subdirectory
        shutil.rmtree(subdir)

        print(f"Moved {fasta_path.name} -> {new_fasta_path.name}")


def rename_fasta_cond_to_condition_weight(
        target_dir,
        old="cond",
        new="condition_weight",
        dry_run=False,
):

    target_dir = Path(target_dir).resolve()

    if not target_dir.exists():
        raise FileNotFoundError(f"Target dir does not exist: {target_dir}")

    fasta_files = list(target_dir.glob("*.fasta"))

    if len(fasta_files) == 0:
        print("No .fasta files found.")
        return

    for fasta in fasta_files:
        if old not in fasta.name:
            continue

        new_name = fasta.name.replace(old, new)
        new_path = fasta.with_name(new_name)

        if dry_run:
            print(f"[DRY-RUN] {fasta.name} -> {new_name}")
        else:
            fasta.rename(new_path)
            print(f"Renamed: {fasta.name} -> {new_name}")

