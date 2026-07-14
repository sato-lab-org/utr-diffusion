import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import itertools
from src.utils.utils import convert_to_seq

nucleotides = ["A", "C", "G", "T"]

def create_sample(
        diffusion_model,
        cell_types: list,
        conditional_numeric_to_tag: dict,
        number_of_samples: int = 1000,
        group_number: list | None = None,
        cond_weight_to_metric: int = 0,
        save_timesteps: bool = False,
        save_dataframe: bool = True,
        generate_attention_maps: bool = False,
    ):

    final_sequences = []
    for n_a in tqdm(range(number_of_samples)):
        sample_bs = 10
        
        if group_number:
            sampled = torch.from_numpy(np.array([group_number] * sample_bs))
        else:
            sampled = torch.from_numpy(np.random.choice(cell_types, sample_bs))

        classes = sampled.float().to(diffusion_model.device)

        if generate_attention_maps:
            sampled_images, cross_att_values = diffusion_model.sample_cross(
                classes, (sample_bs, 1, 4, 200), cond_weight_to_metric
            )
            # save cross attention maps in a numpy array
            np.save(f"cross_att_values_{conditional_numeric_to_tag[group_number]}.npy", cross_att_values)

        else:
            sampled_images = diffusion_model.sample(classes, (sample_bs, 1, 4, 200), cond_weight_to_metric)
        
        if save_timesteps:
            seqs_to_df = {}
            for en, step in enumerate(sampled_images):
                seqs_to_df[en] = [convert_to_seq(x, nucleotides) for x in step]
            final_sequences.append(pd.DataFrame(seqs_to_df))
        
        if save_dataframe:
            # Only using the last timestep
            for en, step in enumerate(sampled_images[-1]):
                final_sequences.append(convert_to_seq(step, nucleotides))
        else:
            for n_b, x in enumerate(sampled_images[-1]):
                seq_final = f">seq_test_{n_a}_{n_b}\n" + "".join(
                    [nucleotides[s] for s in np.argmax(x.reshape(4, 200), axis=0)]
                )
                final_sequences.append(seq_final)
    
    if save_timesteps:
        # Saving dataframe containing sequences for each timestep
        pd.concat(final_sequences, ignore_index=True).to_csv(
            f"final_{conditional_numeric_to_tag[group_number]}.txt",
            header=True,
            sep="\t",
            index=False,
        )
        return
    
    if save_dataframe:
        # Saving list of sequences to txt file
        print("Running Save Dataframe block")
        with open(f"final_{conditional_numeric_to_tag[group_number]}.txt", "w") as f:
            f.write("\n".join(final_sequences))
        return
    
    df_motifs_count_syn = extract_motifs(final_sequences)
    return df_motifs_count_syn

def inference_step(
    diffusion_model,
    labels,
    sample_bs: int,
    seq_len: int,
    cond_weight: float = 0.0,
    output_all_steps: bool = False,
):
    if output_all_steps:
        all_sampled = diffusion_model.sample(
            classes=labels,
            shape=(sample_bs, 1, 4, seq_len),
            cond_weight=cond_weight,
            output_all_steps=True,
        )
        sampled_image = all_sampled[-1]
        all_sampled = torch.stack(all_sampled, dim=0).squeeze(2)
        return sampled_image, all_sampled

    sampled_image = diffusion_model.sample(
        classes=labels,
        shape=(sample_bs, 1, 4, seq_len),
        cond_weight=cond_weight,
        output_all_steps=False,
    )
    return sampled_image, None

def inference(
        diffusion_model,
        class_num,             # int or list/tuple
        cond_weight: float = 0.0,
        sample_bs: int = 1000,
        seq_len: int = 50,
        output_all_steps: bool = False,
        label_names:list[str] = None,
        target_values = None, # for continuous value generation
        device=None,
        with_condition=False,
        fast_gen=True,
):
    if target_values is None: # discrete value
        if isinstance(class_num, int):  # single-label
            return inference_single_label(
                diffusion_model=diffusion_model,
                class_num=class_num,
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
                with_condition=with_condition,
            )
        elif isinstance(class_num, (list, tuple)): # multi-label
            return inference_double_label(
                diffusion_model=diffusion_model,
                num_labels=len(class_num),
                num_classes=class_num[0],
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported class_num type: {type(class_num)}")
    else: # continuous value
        if torch.tensor(target_values, dtype=torch.float32).ndim == 1: #single label
            return inference_continuous_single_label(
                diffusion_model=diffusion_model,
                target_values=target_values,
                cond_weight=cond_weight,
                sample_bs=sample_bs,
                seq_len=seq_len,
                output_all_steps=output_all_steps,
                device=device,
            )
        else: # multi label
            if fast_gen:
                return inference_continuous_multi_label_batched(
                    diffusion_model=diffusion_model,
                    target_values=target_values,
                    label_names=label_names,
                    cond_weight=cond_weight,
                    target_batch_size=9,
                    sample_bs=sample_bs,
                    seq_len=seq_len,
                    output_all_steps=output_all_steps,
                    device=device,
                )
            else:
                return inference_continuous_multi_label(
                        diffusion_model=diffusion_model,
                        target_values=target_values,
                        label_names=label_names,
                        cond_weight=cond_weight,
                        sample_bs=sample_bs,
                        seq_len=seq_len,
                        output_all_steps=output_all_steps,
                        device=device,
                    )
                # return inference_continuous_double_label(
                #     diffusion_model=diffusion_model,
                #     target_values=target_values,
                #     cond_weight=cond_weight,
                #     sample_bs=sample_bs,
                #     seq_len=seq_len,
                #     output_all_steps=output_all_steps,
                #     device=device,
                # )


def inference_single_label(
        diffusion_model,
        class_num: int = 3,
        cond_weight: float = 0.0,
        sample_bs = 1000,
        seq_len = 50,
        output_all_steps = False,
        device=None,
        with_condition=True
    ):
    final_sequences = []
    all_sampled_images = {}

    for label in range(1, class_num+1):
        labels = torch.full((sample_bs,),label, dtype=torch.float).to(device) if with_condition else None
        sampled_image, all_sampled = inference_step(diffusion_model, labels, sample_bs, seq_len, cond_weight, output_all_steps)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            seq = f">class_{label}_seq_{n}\n" + "".join(seq) + "\n"
            final_sequences.append(seq)

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def inference_double_label(
    diffusion_model,
    num_labels: int = 2,
    num_classes: int = 3,
    cond_weight: float = 0.0,
    sample_bs: int = 1000,
    seq_len: int = 50,
    output_all_steps: bool = False,
    device=None,
):

    final_sequences = []
    all_sampled_images = {}

    # make all label combination e.g., [(1,1), (1,2), (1,3), ..., (3,3)]
    label_combinations = list(itertools.product(range(1, num_classes + 1), repeat=num_labels))
    for label_tuple in label_combinations:
        # 构建 label tensor, shape: [sample_bs, num_labels]
        labels = torch.tensor([label_tuple] * sample_bs, dtype=torch.float32).to(device)
        sampled_image, all_sampled = inference_step(diffusion_model, labels, sample_bs, seq_len, cond_weight, output_all_steps)

        # decode to sequence
        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            seq = f">class_{label_tuple}_seq_{n}\n" + "".join(seq) + "\n"
            final_sequences.append(seq)

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def inference_continuous_single_label(
        diffusion_model,
        target_values: list = [4.0, 6.0, 8.0],
        cond_weight: float = 0.0,
        sample_bs = 1000,
        seq_len = 50,
        output_all_steps = False,
        device=None,
    ):
    final_sequences = []
    all_sampled_images = {}

    for idx, tgt in enumerate(target_values):
        labels = torch.full((sample_bs,),tgt, dtype=torch.float).to(device)
        sampled_image, all_sampled = inference_step(diffusion_model, labels, sample_bs, seq_len, cond_weight, output_all_steps)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            seq = f">target_{tgt}_seq_{n}\n" + "".join(seq) + "\n"
            final_sequences.append(seq)

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences


def inference_continuous_double_label(
        diffusion_model,
        target_values: list = [[4.0, -10.0], [6.0, -5.0], [8.0, -5.0]],
        cond_weight: float = 0.0,
        sample_bs = 1000,
        seq_len = 50,
        output_all_steps: bool = False,
        device=None,
    ):
    final_sequences = []
    all_sampled_images = {}

    for idx, tgt in enumerate(target_values):
        labels = torch.tensor([tgt] * sample_bs, dtype=torch.float32).to(device)
        sampled_image, all_sampled = inference_step(diffusion_model, labels, sample_bs, seq_len, cond_weight, output_all_steps)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            header = f">target_MRL{tgt[0]:.1f}_MFE{tgt[1]:.1f}_seq_{n}"
            final_sequences.append(header + "\n" + "".join(seq) + "\n")

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences

def inference_continuous_multi_label(
        diffusion_model,
        target_values,
        label_names=None,
        cond_weight: float = 0.0,
        sample_bs: int = 1000,
        seq_len: int = 50,
        output_all_steps: bool = False,
        device=None,
        header_decimals: int = 2,
):
    final_sequences = []
    all_sampled_images = {}

    for idx, tgt in enumerate(target_values):
        labels = torch.tensor([tgt] * sample_bs, dtype=torch.float32).to(device)
        sampled_image, all_sampled = inference_step(diffusion_model, labels, sample_bs, seq_len, cond_weight, output_all_steps)

        for n, x in enumerate(sampled_image):
            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0)]
            label_part = "_".join([f"{name}_{val:.{header_decimals}f}" for name, val in zip(label_names, tgt)])
            header = f">target_{label_part}_seq_{n}"
            final_sequences.append(header + "\n" + "".join(seq) + "\n")

    return (final_sequences, all_sampled_images) if output_all_steps else final_sequences

import torch
import numpy as np


def inference_continuous_multi_label_batched(
        diffusion_model,
        target_values,
        label_names=None,
        cond_weight: float = 0.0,
        sample_bs: int = 100,
        target_batch_size: int = 10,
        seq_len: int = 50,
        output_all_steps: bool = False,
        device=None,
        header_decimals: int = 2,
):
    """
    Generate sequences for multiple target conditions in batches.

    Args:
        diffusion_model:
            Trained diffusion model.
        target_values:
            list[list[float]] / np.ndarray / torch.Tensor
            Shape: [num_targets, num_labels]
        label_names:
            list[str], label names used in FASTA header.
        cond_weight:
            Classifier-free guidance weight.
        sample_bs:
            Number of sequences generated per target condition.
        target_batch_size:
            Number of target conditions processed in one inference call.
            Actual batch size = sample_bs * target_batch_size.
        seq_len:
            Sequence length.
        output_all_steps:
            Whether to return all sampled diffusion steps.
        device:
            torch device.
        header_decimals:
            Decimal digits in FASTA header.

    Returns:
        final_sequences:
            list[str], FASTA formatted sequences.
        all_sampled_images:
            dict, only returned when output_all_steps=True.
    """

    if device is None:
        device = next(diffusion_model.parameters()).device

    if not torch.is_tensor(target_values):
        target_values = torch.tensor(target_values, dtype=torch.float32)

    target_values = target_values.to(device)

    if target_values.ndim == 1:
        target_values = target_values.unsqueeze(0)

    num_targets, num_labels = target_values.shape

    if label_names is None:
        label_names = [f"label{i}" for i in range(num_labels)]

    if len(label_names) != num_labels:
        raise ValueError(f"len(label_names)={len(label_names)} does not match! num_labels={num_labels}")

    final_sequences = []
    all_sampled_images = {}

    for start in range(0, num_targets, target_batch_size):
        end = min(start + target_batch_size, num_targets)

        # [target_batch_size, num_labels]
        target_batch = target_values[start:end]

        # [target_batch_size, num_labels] -> [target_batch_size * sample_bs, num_labels]
        labels = target_batch.repeat_interleave(sample_bs, dim=0)

        total_bs = labels.shape[0]

        sampled_image, all_sampled = inference_step(diffusion_model, labels, total_bs, seq_len, cond_weight, output_all_steps,)

        if output_all_steps:
            all_sampled_images[f"target_batch_{start}_{end}"] = all_sampled

        for global_idx, x in enumerate(sampled_image):
            local_target_idx = global_idx // sample_bs
            seq_idx = global_idx % sample_bs

            target_idx = start + local_target_idx
            tgt = target_values[target_idx].detach().cpu().tolist()

            seq = [nucleotides[s] for s in torch.argmax(x.squeeze(0), dim=0).detach().cpu().tolist()]
            label_part = "_".join([f"{name}_{val:.{header_decimals}f}" for name, val in zip(label_names, tgt)])
            header = f">target_{label_part}_seq_{seq_idx}"
            final_sequences.append(header + "\n" + "".join(seq) + "\n")

    if output_all_steps:
        return final_sequences, all_sampled_images

    return final_sequences


def build_continuous_multilabel_fasta_header(
    target_values,
    label_names,
    seq_idx,
    prefix="target",
    decimals=2,
    label_aliases=None,
):
    """
    Build FASTA header for continuous multi-label generation.

    Example:
        >target_poly-1.00_prot-1.00_ssrna0.00_mfe2.00_seq_15
    """

    if label_aliases is None:
        label_aliases = {}

    if len(target_values) != len(label_names):
        raise ValueError(
            f"target_values length ({len(target_values)}) != "
            f"label_names length ({len(label_names)})"
        )

    parts = []
    for label, value in zip(label_names, target_values):
        short_label = label_aliases.get(label, label)
        short_label = (
            short_label
            .replace(".", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )

        parts.append(f"{short_label}{value:.{decimals}f}")

    condition_str = "_".join(parts)

    return f">{prefix}_{condition_str}_seq_{seq_idx}"

def extract_motifs(sequence_list: list):
    """Extract motifs from a list of sequences"""
    motifs = open("synthetic_motifs.fasta", "w")
    motifs.write("\n".join(sequence_list))
    motifs.close()
    os.system("gimme scan synthetic_motifs.fasta -p JASPAR2020_vertebrates -g hg38 -n 20 > syn_results_motifs.bed")
    df_results_syn = pd.read_csv("syn_results_motifs.bed", sep="\t", skiprows=5, header=None)
    df_results_syn["motifs"] = df_results_syn[8].apply(lambda x: x.split('motif_name "')[1].split('"')[0])
    df_results_syn[0] = df_results_syn[0].apply(lambda x: "_".join(x.split("_")[:-1]))
    df_motifs_count_syn = df_results_syn[[0, "motifs"]].groupby("motifs").count()
    return df_motifs_count_syn


def convert_sample_to_fasta(sample_path: list):
    """Convert cell specific samples to a fasta format"""
    sequences = []
    samples = pd.read_csv(sample_path, sep="\t", header=None)
    # Extract each line of the dataframe into a list
    samples_list = samples[0].tolist()
    # Convert into a fasta format
    for i, seq in enumerate(samples_list):
        sequences.append(f">sequence_{i}\n" + seq)
    return sequences
