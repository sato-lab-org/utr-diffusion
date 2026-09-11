from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
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

csv_Path = 'exp2_30_peptide_candidates/exp2_30_human_peptide_candidates.csv'
AMINO_PATTERNS = get_amino_context_from_csv(csv_Path, column_amino='downstream_peptide_16aa', column_ID='candidate_id', padding='UG')
STRATEGY = 'specific_adaptiveness'
GAMMA = 0.04
target_adaptiveness =  [0.85, 0.9, 0.95]

def benchmark_MFE_CAI_optimization(trial_path):
    accelerator = Accelerator()

    unet = UNet_CML(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2,
        num_label=2,
    )

    diffusion = Diffusion_CML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=2,
        uncondition_prop=0.2,
    )

    ckpt_path = os.path.join(trial_path, 'checkpoints/epoch_2000.pt')
    checkpoint_dict = torch.load(ckpt_path, map_location='cpu')
    diffusion.load_state_dict(checkpoint_dict['model'])
    diffusion = accelerator.prepare(diffusion)

    target_labels = [[8.0, -25.0]]
    cond_weight = 4.0
    repaint = Repaint_Amino_CML(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        return_all=False,
        tgt_labels=target_labels,
        strategy=STRATEGY,
        gamma_for_usage_frequency=GAMMA,
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)

    # we need to run for several times, so today we just run for Quadra_Amino pattern as it has 81 variants
    for adap in target_adaptiveness:
        save_path = os.path.join(trial_path, 'benchmark_MFE_CAI_optimization', f'cond_weight_{cond_weight}_adaptiveness_{adap}')

        os.makedirs(save_path, exist_ok=True)
        save_experiment_config(
            save_dir=save_path,
            checkpoint_path=ckpt_path,
            repaint=repaint,
            target_labels=target_labels,
            family_name='benchmark_MFE_CAI_optimization',
            patterns=AMINO_PATTERNS,
            extra_config={
                "experiment_describe": "benchmark of MFE-CAI optimization on AUG-proximal CDS region",
                "strategy": STRATEGY,
                "adaptiveness": adap,
                "gamma_for_usage_frequency": GAMMA,
            }
        )
        for pattern in AMINO_PATTERNS:
            if os.path.exists(os.path.join(save_path, pattern['name'] + '.fasta')):
                print(f'[INFO] Skip existing pattern: {pattern["name"]}')
                continue
            print(f'[INFO] Sampling for pattern: {pattern["name"]} with amino {pattern["amino"]} at positions {pattern["pos"]}')
            repaint.setup(constraint_list=pattern['amino'], pos_list=pattern['pos'], adaptiveness=adap)
            result = repaint.p_resample()

            write_fasta(result, os.path.join(save_path, pattern['name'] + '.fasta'), tgt_values=target_labels, batch_bs=100)


if __name__ == "__main__":
    benchmark_MFE_CAI_optimization('../../outputs/real_MRL_pred_MFE_260k_mcml')
