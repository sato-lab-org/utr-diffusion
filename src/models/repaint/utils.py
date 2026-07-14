# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license
from os import replace

import yaml
import os
from PIL import Image
import random
from .amino_codon_table import get_codons_for_amino, get_amino_for_codon, rna_to_dna
import numpy as np
import torch

from src.models.repaint.amino_codon_table import AMINO_TO_CODONS

nucleotides = ['A', 'C', 'G', 'T']

base2vec = {"A": [1, -1, -1, -1],
            "C": [-1, 1, -1, -1],
            "G": [-1, -1, 1, -1],
            "T": [-1, -1, -1, 1],
            "N": [-1, -1, -1, -1]}

inf_base2vec = {
    "A": [1, -1, -1, -1],
    "C": [-1, 1, -1, -1],
    "G": [-1, -1, 1, -1],
    "T": [-1, -1, -1, 1],
    "N": [-float('inf'), -float('inf'), -float('inf'), -float('inf')],
}

def txtread(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))

def imwrite(path=None, img=None):
    Image.fromarray(img).save(path)


def get_codon_sequence(codon='ACG', pos=0, seq_length: int =50):
    seq = ['N'] * seq_length
    seq[pos:pos+len(codon)] = list(codon)
    return ''.join(seq)

def get_codon_image(codon, pos):
    sequence = get_codon_sequence(codon=codon, pos=pos)
    image = torch.tensor([nucleotides.get(base) for base in sequence], dtype=torch.float).T # [50, 4] -> [4, 50]
    return image.unsqueeze(0) # add channel dim

def get_mask(length, pos):
    mask = torch.zeros(4, 50, dtype=torch.float)
    mask[:, pos : pos + length] = 1.0
    return mask.unsqueeze(0) # add channel dim

def aminos_to_amino_images(tgt_aminos, with_padding=True):
    # amino_images [num_amino, num_codons_per_amino, 4, 3]
    # N: The number of specified amino acid
    # num_codons_per_amino: 1 amino -> 2~6 codons, here we change it to 1 amino -> 6 codons using 'NNN' padding
    all_amino_images =[]
    for amino in tgt_aminos:
        alter_codons = get_codons_for_amino(amino)
        alter_codons = alter_codons + (6 -len(alter_codons)) * ['NNN'] if with_padding else alter_codons
        amino_image = []
        for codon in alter_codons:
            codon_image = torch.tensor([inf_base2vec[base] for base in codon], dtype=torch.float).T
            amino_image.append(codon_image)
        amino_image = torch.stack(amino_image, dim=0)
        all_amino_images.append(amino_image)
    return torch.stack(all_amino_images, dim=0)


def amino_images_to_aminos(amino_images: torch.Tensor) -> str:
    """
    reverse of aminos_to_amino_images, but we only return the most likely amino acid for each position, without considering the codon degeneracy
    """


def prepare_amino_context(amino_pattern:dict):
    amino_context_list = []

    for amino in amino_pattern['amino']:
        amino = amino
    return amino_context_list


def bulid_gt_and_mask_from_codons(codon_list: list[str], pos_list: list[int], total_length: int=50):
    seq = ['N'] * total_length
    mask = torch.zeros(4, total_length, dtype=torch.float)

    prev_pos, codon_length = 0, 0
    for codon, pos in zip(codon_list, pos_list):
        if prev_pos + codon_length > pos:
            raise ValueError("codons overlap.")
        codon_length = len(codon)
        seq[pos : pos + codon_length] = list(codon)
        mask[:, pos : pos + codon_length] = 1.0
        prev_pos = pos
    seq = rna_to_dna(''.join(seq))
    image = torch.tensor([base2vec.get(base) for base in seq], dtype=torch.float).T # [50, 4] -> [4, 50]
    return image.unsqueeze(0), mask.unsqueeze(0) # add channel dim

# this func is to generate new gt image for codon image
def build_gt_from_image_and_pos(codon_images:torch.Tensor, pos_list:list, total_length: int=50, device: torch.device = None) -> torch.Tensor:
    # images: [B, N, 3, 4]
    # amino_pos [N]
    # return new ground truth image with new codon in fix amino_pos
    B, N, _, _ = codon_images.shape
    if device == None:
        gt = torch.zeros(B, 1, 4, total_length, dtype=torch.float) # [B, 1, 4, 50]
    else:
        gt = torch.zeros(B, 1, 4, total_length, dtype=torch.float, device=device)

    for i, pos in enumerate(pos_list):
        gt[:, 0, :, pos : pos + 3] = codon_images[:, i, :, :]
    return gt



def build_gt_mask_from_aminos(amino_list: list[str], pos_list: list[int], total_length: int=50):
    seq = ['N'] * total_length
    mask = torch.zeros(4, total_length, dtype=torch.float)

    prev_pos, amino_length = -3, 3
    for amino, pos in zip(amino_list, pos_list):
        if prev_pos + amino_length > pos:
            raise ValueError("aminos overlap.")
        alter_codons = get_codons_for_amino(amino)
        seq[pos : pos + amino_length] = random.choice(alter_codons)
        mask[:, pos : pos + amino_length] = 1.0
        prev_pos = pos
    seq = ''.join(seq)
    image = torch.tensor([base2vec.get(base) for base in seq], dtype=torch.float).T
    return image.unsqueeze(0), mask.unsqueeze(0)


def build_codon_usage_table_for_specific_CAI(
        amino_to_codons,
        codon_usage_table,
        target_cai: float,
        max_num_codons: int = 6,
):
    """
    Return:
        cai_prob_table: dict
            aa -> torch.Tensor [6]
    """

    cai_prob_table = {}

    for aa, codons in amino_to_codons.items():

        if aa == '*':
            continue

        usage = torch.tensor(
            [codon_usage_table[aa][codon] for codon in codons],
            dtype=torch.float32
        )

        q = usage / usage.sum()

        # CAI relative adaptiveness
        w = usage / usage.max()

        lambda_value, clipped_target_cai = calculate_lambda_for_target_cai(
            q=q,
            w=w,
            target_cai=target_cai
        )

        numerator = q * torch.exp(lambda_value * w)
        p = numerator / numerator.sum()

        logits = torch.log(q + 1e-12) + lambda_value * w
        p = torch.softmax(logits, dim=0)

        padded_p = torch.zeros(max_num_codons, dtype=torch.float32)
        padded_p[:len(codons)] = p

        cai_prob_table[aa] = padded_p

    return cai_prob_table


def calculate_lambda_for_target_cai(q, w, target_cai, max_iter=100, tol=1e-6):
    """
    q: codon usage frequency, [K]
    w: relative adaptiveness / CAI value, [K]
    target_cai: desired expected CAI for this amino acid

    p_i = q_i * exp(lambda * w_i) / sum_j q_j * exp(lambda * w_j)
    """

    q = torch.as_tensor(q, dtype=torch.float32)
    w = torch.as_tensor(w, dtype=torch.float32)

    target_cai = float(target_cai)

    min_w = float(w.min())
    max_w = float(w.max())

    # clip target CAI to feasible range for this amino acid
    if target_cai < min_w:
        target_cai = min_w

    if target_cai > max_w:
        target_cai = max_w

    left, right = -50.0, 50.0

    for _ in range(max_iter):
        mid = (left + right) / 2.0

        numerator = q * torch.exp(mid * w)
        p = numerator / numerator.sum()

        expected_cai = torch.sum(p * w).item()

        if abs(expected_cai - target_cai) < tol:
            return mid, target_cai

        if expected_cai < target_cai:
            left = mid
        else:
            right = mid

    return (left + right) / 2.0, target_cai

def write_fasta(matrices, save_path, num_class: int = 3, tgt_values=None, batch_bs: int = 100):
    lines = []
    if isinstance(matrices, list):  # multi-frame
        iterable = enumerate(matrices)
    else:  # single-frame, last frame
        iterable = [('last', matrices)]

    for step, seqs in iterable:
        # do class-wise generation
        if tgt_values is None:
            for class_idx in range(0, num_class):
                for n in range(batch_bs):  # seqs: [1000, 4, 50]
                    seq = seqs[class_idx * batch_bs + n, 0, :, :] # [4, 50]
                    sequence = ''.join([nucleotides[n] for n in seq.argmax(axis=0)])
                    header = f">_{step}_{class_idx + 1}_{n}"
                    lines.append(header)
                    lines.append(sequence)
        # do target values-wise generation
        else:
            for class_idx, tgt in enumerate(tgt_values):
                for n in range(batch_bs):  # seqs: [1000, 4, 50]
                    seq = seqs[class_idx * batch_bs + n, 0, :, :] # [4, 50]
                    sequence = ''.join([nucleotides[n] for n in seq.argmax(axis=0)])
                    if isinstance(tgt, (list, tuple)):
                        v1, v2 = tgt
                        header = f">_step_{step}_mrl_{v1}_mfe_{v2}_idx_{n}" if step != 'last' else f'>_mrl_{v1}_mfe_{v2}_idx_{n}'
                    else:
                        header = f">_step_{step}_tgt_{tgt}_idx_{n}" if step != 'last' else f'>_tgt_{tgt}_idx_{n}'
                    lines.append(header)
                    lines.append(sequence)

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print("fasta saved")
