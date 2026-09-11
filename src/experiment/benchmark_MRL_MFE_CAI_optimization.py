from accelerate import Accelerator
from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels as Diffusion_MCML
from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels as UNet_MCML
from src.models.repaint.repaint_amino_nucleo_cml import RePaint_Amino_Nucleotide_Continuous_Multi_Labels as Repaint_Amino_CML
import torch
import pandas as pd
import numpy as np
import warnings
import os
from src.models.repaint.utils import write_fasta
from src.experiment.exp_configuration import save_experiment_config
from copy import deepcopy

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def get_amino_context_from_csv(csv_path, column_amino='downstream_peptide_16aa', column_ID='candidate_id', padding='UG',):
    df = pd.read_csv(csv_path)
    amino_patterns = []
    for _, row in df.iterrows():
        amino_sequence = row[column_amino]
        pattern_id = row[column_ID]

        pattern = {
            'name': pattern_id,
            'amino': [padding] + list(amino_sequence),
            'pos': [0] + [len(padding) + 3 * i for i in range(len(amino_sequence))],
        }

        amino_patterns.append(pattern)

    return amino_patterns

def extend_utr_by_shortening_cds(amino_pattern, remove_tail_aa:int = 0, complete_nt:str = 'A',):
    """
    Shorten the CDS constraint by removing the final N amino acids, and use the released 3*N nucleotides as an unconstrained UTR region.
    Example
    Original: UG + 16 aa
    remove_tail_aa = 1: 2-nt UTR + AUG + 15 aa
    remove_tail_aa = 2: 5-nt UTR + AUG + 14 aa
    """
    modified_pattern = deepcopy(amino_pattern)

    if remove_tail_aa == 0:
        return modified_pattern

    shift_nt = 3 * remove_tail_aa

    # Remove the final N amino acids and shift pos
    modified_pattern['amino'] = modified_pattern['amino'][:-remove_tail_aa]
    modified_pattern['pos'] = modified_pattern['pos'][:-remove_tail_aa]
    modified_pattern['pos'] = [pos + shift_nt for pos in modified_pattern['pos']]

    # Complete the first partial nucleotide constraint: UG -> AUG
    modified_pattern['amino'][0] = (complete_nt + modified_pattern['amino'][0])
    modified_pattern['pos'][0] -= len(complete_nt)
    if modified_pattern['amino'][0] == 'AUG': # we only have the situation that "AUG" without "A", so
        modified_pattern['amino'][0] = 'M'
    return modified_pattern

csv_Path = 'exp2_30_peptide_candidates/exp2_30_human_peptide_candidates.csv'
AMINO_PATTERNS = get_amino_context_from_csv(csv_Path, column_amino='downstream_peptide_16aa', column_ID='candidate_id', padding='UG')
STRATEGY = 'euclidean'#'specific_adaptiveness' #'usage_weighted_distance'
GAMMA = 0.04
#target_adaptiveness = [0.85, 0.9, 0.95]
cond_weight = 4
adap = None

def benchmark_MFE_CAI_optimization(trial_path):
    accelerator = Accelerator()

    unet = UNet_MCML(
        dim=200,  # 200
        channels=1,
        dim_mults=(1, 2, 4),  # (1,2,4)
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2,
        num_label=2,
        label_emb_mode='per_label'
    )

    diffusion = Diffusion_MCML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=cond_weight,
        uncondition_prop=0.2,
        label_wise_mask=False,  # True,
    )

    ckpt_path = os.path.join(trial_path, 'checkpoints/epoch_2000.pt')
    checkpoint_dict = torch.load(ckpt_path, map_location='cpu')
    diffusion.load_state_dict(checkpoint_dict['model'])
    diffusion = accelerator.prepare(diffusion)

    target_mrl, target_mfe = 9, 0
    target_labels = [[target_mrl, target_mfe]]

    repaint = Repaint_Amino_CML(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=cond_weight,
        return_all=False,
        tgt_labels=target_labels,
        strategy=STRATEGY,
        gamma_for_usage_frequency=GAMMA,
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)

    # we need to run for several times, so today we just run for Quadra_Amino pattern as it has 81 variants
    #for adap in target_adaptiveness:
    save_path = os.path.join(trial_path, 'benchmark_MRL_MFE_CAI_optimization', f'mrl_{target_mrl}_mfe_{target_mfe}_cond_{cond_weight}_gamma_{GAMMA}',f'adaptiveness_euclidean')

    os.makedirs(save_path, exist_ok=True)
    save_experiment_config(
        save_dir=save_path,
        checkpoint_path=ckpt_path,
        repaint=repaint,
        target_labels=target_labels,
        family_name='benchmark_MFE_CAI_optimization',
        patterns=AMINO_PATTERNS,
        extra_config={
            "experiment_describe": "benchmark of MRL-MFE-CAI optimization on AUG-proximal CDS region",
            "strategy": STRATEGY,
            "adaptiveness": adap,
            "gamma_for_usage_frequency": GAMMA,
        }
    )
    for original_pattern in AMINO_PATTERNS:
        pattern = extend_utr_by_shortening_cds(original_pattern, remove_tail_aa=7)
        savename = os.path.join(save_path, pattern['name'] + f'_UTR_20nt.fasta')
        if os.path.exists(savename):
            print(f'[INFO] Skip existing pattern: {pattern["name"]}')
            continue
        print(f'[INFO] Sampling for pattern: {pattern["name"]} with amino {pattern["amino"]} at positions {pattern["pos"]} with 5 prime UTR 20nt')
        repaint.setup(constraint_list=pattern['amino'], pos_list=pattern['pos'], adaptiveness=adap)
        result = repaint.p_resample()

        write_fasta(result, savename, tgt_values=target_labels, batch_bs=100)


if __name__ == "__main__":
    benchmark_MFE_CAI_optimization('../../outputs/real_MRL_pred_MFE_260k_mcml')
