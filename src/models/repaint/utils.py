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
import inspect
import math
import numbers
from collections.abc import Sequence

import yaml
import os
from PIL import Image
import random
from .amino_codon_table import (
    AA_TO_CODON_USAGE_HUMAN_RNA,
    AMINO_TO_CODONS,
    dna_to_rna,
    get_codons_for_amino,
    get_amino_for_codon,
)
import torch

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
    image = torch.tensor([base2vec[base] for base in sequence], dtype=torch.float).T # [50, 4] -> [4, 50]
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


def get_amino_images_for_alter_codons(tgt_aminos, with_padding=True):
    """Compatibility alias for :func:`aminos_to_amino_images`."""

    return aminos_to_amino_images(tgt_aminos, with_padding=with_padding)


def validate_amino_constraints(amino_list, pos_list, total_length=50):
    """Validate and normalize amino-acid constraints.

    Positions are zero-based codon starts.  They must be strictly ordered so
    that the flattened masked region has the same order as ``amino_list``.
    Stop codons and ambiguous amino-acid symbols are deliberately rejected.
    """

    if isinstance(amino_list, (str, bytes)) or not isinstance(amino_list, Sequence):
        raise TypeError("amino_list must be a sequence of one-letter amino-acid codes")
    if isinstance(pos_list, (str, bytes)) or not isinstance(pos_list, Sequence):
        raise TypeError("pos_list must be a sequence of zero-based codon starts")
    if len(amino_list) == 0:
        raise ValueError("at least one amino-acid constraint is required")
    if len(amino_list) != len(pos_list):
        raise ValueError("amino_list and pos_list must have the same length")
    if isinstance(total_length, bool) or not isinstance(total_length, numbers.Integral):
        raise TypeError("total_length must be an integer")
    if total_length < 3:
        raise ValueError("total_length must be at least 3")

    normalized_aminos = []
    normalized_positions = []
    previous_end = 0

    for index, (amino, pos) in enumerate(zip(amino_list, pos_list)):
        if not isinstance(amino, str) or len(amino.strip()) != 1:
            raise ValueError(f"amino acid at index {index} must be a one-letter code")
        amino = amino.strip().upper()
        if amino == '*':
            raise ValueError("stop-codon constraints are not supported")
        if amino not in AMINO_TO_CODONS:
            raise ValueError(f"unsupported amino-acid code {amino!r} at index {index}")

        if isinstance(pos, bool) or not isinstance(pos, numbers.Integral):
            raise TypeError(f"position at index {index} must be an integer")
        pos = int(pos)
        if pos < 0 or pos + 3 > total_length:
            raise ValueError(
                f"codon at position {pos} exceeds sequence bounds [0, {total_length})"
            )
        if index > 0 and pos < previous_end:
            raise ValueError("amino-acid positions must be sorted and cannot overlap")

        normalized_aminos.append(amino)
        normalized_positions.append(pos)
        previous_end = pos + 3

    return normalized_aminos, normalized_positions


def validate_target_adaptiveness(target_adaptiveness):
    """Return a finite codon relative-adaptiveness target in ``(0, 1]``."""

    if isinstance(target_adaptiveness, bool) or not isinstance(
        target_adaptiveness, numbers.Real
    ):
        raise TypeError("target_adaptiveness must be a real number")
    target_adaptiveness = float(target_adaptiveness)
    if not math.isfinite(target_adaptiveness):
        raise ValueError("target_adaptiveness must be finite")
    if not 0.0 < target_adaptiveness <= 1.0:
        raise ValueError("target_adaptiveness must be in the interval (0, 1]")
    return target_adaptiveness


def normalize_target_labels(tgt_labels):
    """Validate scalar or fixed-width continuous conditioning targets.

    NaN is preserved because MCML uses it to denote an absent label.  Infinite
    values and mixed scalar/vector target layouts are rejected.
    """

    if isinstance(tgt_labels, (str, bytes)) or not isinstance(tgt_labels, Sequence):
        raise TypeError("tgt_labels must be a non-empty sequence")
    if len(tgt_labels) == 0:
        raise ValueError("tgt_labels cannot be empty")

    first_is_scalar = isinstance(tgt_labels[0], numbers.Real) and not isinstance(
        tgt_labels[0], bool
    )
    normalized = []
    expected_width = None

    for target_index, target in enumerate(tgt_labels):
        if first_is_scalar:
            if isinstance(target, bool) or not isinstance(target, numbers.Real):
                raise ValueError("tgt_labels cannot mix scalar and vector targets")
            value = float(target)
            if math.isinf(value):
                raise ValueError(f"target label at index {target_index} cannot be infinite")
            normalized.append(value)
            continue

        if isinstance(target, (str, bytes)) or not isinstance(target, Sequence):
            raise ValueError("tgt_labels cannot mix scalar and vector targets")
        if expected_width is None:
            expected_width = len(target)
            if expected_width == 0:
                raise ValueError("target label vectors cannot be empty")
        elif len(target) != expected_width:
            raise ValueError("all target label vectors must have the same length")

        row = []
        for label_index, value in enumerate(target):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(
                    f"target label ({target_index}, {label_index}) must be a real number"
                )
            value = float(value)
            if math.isinf(value):
                raise ValueError(
                    f"target label ({target_index}, {label_index}) cannot be infinite"
                )
            row.append(value)
        normalized.append(tuple(row))

    return normalized


def get_guidance_mask_parameter(diffusion):
    """Select the guidance-mask keyword supported by ``diffusion``.

    MCML diffusion uses ``uncond_mask`` while the earlier CML implementation
    uses ``context_mask``.  Inspecting the public signature avoids masking a
    genuine ``TypeError`` raised from inside the sampler implementation.
    """

    sample_fn = getattr(diffusion, 'p_sample_guided', None)
    if sample_fn is None or not callable(sample_fn):
        raise TypeError("diffusion must provide a callable p_sample_guided method")

    parameters = inspect.signature(sample_fn).parameters
    if 'uncond_mask' in parameters:
        return 'uncond_mask'
    if 'context_mask' in parameters:
        return 'context_mask'
    raise TypeError(
        "diffusion.p_sample_guided must accept either 'uncond_mask' or "
        "'context_mask'"
    )


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
    seq = ''.join(seq)
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
    amino_list, pos_list = validate_amino_constraints(
        amino_list,
        pos_list,
        total_length=total_length,
    )
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


def _validate_probability_inputs(q, w):
    q = torch.as_tensor(q, dtype=torch.float64)
    w = torch.as_tensor(w, dtype=torch.float64, device=q.device)

    if q.ndim != 1 or w.ndim != 1 or q.numel() == 0 or q.shape != w.shape:
        raise ValueError("q and w must be non-empty one-dimensional tensors of equal size")
    if not torch.isfinite(q).all() or (q <= 0).any():
        raise ValueError("q must contain finite, strictly positive weights")
    if not torch.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError("w must contain finite relative-adaptiveness values in [0, 1]")

    return q / q.sum(), w


def solve_lambda_for_target_adaptiveness(
        q,
        w,
        target_adaptiveness,
        max_iter=200,
        tol=1e-9,
):
    """Solve the exponential-tilt parameter for an expected adaptiveness.

    The tilted probabilities are ``softmax(log(q) + lambda * w)``.  The
    requested alpha is clipped to the feasible range for the amino acid.  The
    returned target is that effective (possibly clipped) alpha, not a sequence
    CAI value.
    """

    if isinstance(max_iter, bool) or not isinstance(max_iter, numbers.Integral):
        raise TypeError("max_iter must be an integer")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    tol = float(tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    q, w = _validate_probability_inputs(q, w)
    target_adaptiveness = validate_target_adaptiveness(target_adaptiveness)
    effective_target = min(max(target_adaptiveness, float(w.min())), float(w.max()))

    min_w = float(w.min())
    max_w = float(w.max())
    if math.isclose(min_w, max_w, abs_tol=tol):
        return 0.0, effective_target
    if math.isclose(effective_target, min_w, abs_tol=tol):
        return -math.inf, effective_target
    if math.isclose(effective_target, max_w, abs_tol=tol):
        return math.inf, effective_target

    log_q = torch.log(q)

    def expected_adaptiveness(lambda_value):
        probability = torch.softmax(log_q + lambda_value * w, dim=0)
        return float(torch.dot(probability, w))

    left, right = -1.0, 1.0
    max_abs_lambda = 2.0 ** 40
    while expected_adaptiveness(left) > effective_target:
        left *= 2.0
        if abs(left) > max_abs_lambda:
            raise RuntimeError("could not bracket target adaptiveness from below")
    while expected_adaptiveness(right) < effective_target:
        right *= 2.0
        if right > max_abs_lambda:
            raise RuntimeError("could not bracket target adaptiveness from above")

    mid = 0.0
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        expected = expected_adaptiveness(mid)
        if abs(expected - effective_target) <= tol:
            break
        if expected < effective_target:
            left = mid
        else:
            right = mid

    return mid, effective_target


def probabilities_for_target_adaptiveness(q, w, target_adaptiveness, tol=1e-9):
    """Construct stable probabilities with the requested expected alpha.

    Endpoint targets are handled explicitly rather than exponentiating an
    infinite tilt.  This keeps probabilities finite and normalized even for
    alpha values at, or outside, an amino acid's feasible range.

    Returns ``(probabilities, effective_target_adaptiveness)``.
    """

    tol = float(tol)
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    q, w = _validate_probability_inputs(q, w)
    lambda_value, effective_target = solve_lambda_for_target_adaptiveness(
        q=q,
        w=w,
        target_adaptiveness=target_adaptiveness,
        tol=tol,
    )

    if math.isinf(lambda_value):
        endpoint = w.max() if lambda_value > 0 else w.min()
        endpoint_mask = torch.isclose(w, endpoint, rtol=0.0, atol=tol)
        probability = q * endpoint_mask
        probability = probability / probability.sum()
    else:
        probability = torch.softmax(torch.log(q) + lambda_value * w, dim=0)

    return probability, effective_target


def build_target_adaptiveness_probability_table(
        target_adaptiveness,
        amino_to_codons=AMINO_TO_CODONS,
        codon_usage_table=AA_TO_CODON_USAGE_HUMAN_RNA,
        max_num_codons=6,
        return_effective_targets=False,
):
    """Build padded codon probabilities for a target adaptiveness alpha.

    ``target_adaptiveness`` controls the expected *per-codon relative
    adaptiveness*.  It is not the CAI of a generated multi-codon sequence.
    """

    target_adaptiveness = validate_target_adaptiveness(target_adaptiveness)
    if isinstance(max_num_codons, bool) or not isinstance(max_num_codons, numbers.Integral):
        raise TypeError("max_num_codons must be an integer")
    if max_num_codons < 1:
        raise ValueError("max_num_codons must be positive")

    probability_table = {}
    effective_targets = {}
    for amino, codons_for_amino in amino_to_codons.items():
        if amino == '*':
            continue

        codons = list(codons_for_amino)
        if len(codons) > max_num_codons:
            raise ValueError(
                f"amino acid {amino!r} has {len(codons)} codons, exceeding "
                f"max_num_codons={max_num_codons}"
            )

        try:
            usage = torch.tensor(
                [codon_usage_table[amino][dna_to_rna(codon)] for codon in codons],
                dtype=torch.float64,
            )
        except KeyError as exc:
            raise ValueError(
                f"missing codon-usage value for amino acid {amino!r}: {exc.args[0]!r}"
            ) from exc

        q = usage / usage.sum()
        relative_adaptiveness = usage / usage.max()
        probability, effective_target = probabilities_for_target_adaptiveness(
            q=q,
            w=relative_adaptiveness,
            target_adaptiveness=target_adaptiveness,
        )

        padded_probability = torch.zeros(max_num_codons, dtype=torch.float32)
        padded_probability[:len(codons)] = probability.to(torch.float32)
        probability_table[amino] = padded_probability
        effective_targets[amino] = effective_target

    if return_effective_targets:
        return probability_table, effective_targets
    return probability_table


def calculate_lambda_for_target_cai(q, w, target_cai, max_iter=200, tol=1e-9):
    """Compatibility wrapper; ``target_cai`` historically meant alpha.

    Despite the legacy parameter name, the target is expected codon relative
    adaptiveness, not sequence CAI.
    """

    return solve_lambda_for_target_adaptiveness(
        q=q,
        w=w,
        target_adaptiveness=target_cai,
        max_iter=max_iter,
        tol=tol,
    )


def build_codon_usage_table_for_specific_CAI(
        amino_to_codons,
        codon_usage_table,
        target_cai,
        max_num_codons=6,
):
    """Compatibility wrapper for the historical, misleading CAI name.

    ``target_cai`` is treated as a target codon relative adaptiveness alpha;
    this helper does not target or calculate sequence CAI.
    """

    return build_target_adaptiveness_probability_table(
        target_adaptiveness=target_cai,
        amino_to_codons=amino_to_codons,
        codon_usage_table=codon_usage_table,
        max_num_codons=max_num_codons,
    )



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
