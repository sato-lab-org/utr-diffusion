from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.models.repaint.repaint_amino_cml import RePaint_Amino_Continuous_Multi_Labels as Repaint_Amino_CML
import torch
import warnings
import os
import yaml
from src.models.repaint.utils import build_gt_mask_from_aminos, write_fasta
from src.experiment.exp_configuration import save_experiment_config

from src.experiment.exp_codon_pattern import build_amino_constraint_patterns
from src.experiment.exp_target_labels import joint_target_values_sweep

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def get_amino_context_from_path(save_path):
    if os.path.exists(save_path):
        amino_blueprint = yaml.safe_load(open(save_path, 'r'))
    else:
        amino_blueprint = build_amino_constraint_patterns(
            pos = [29, 32, 35, 38, 41, 44, 47],
            num_amino_choices = [1, 1, 1, 1, 1, 1, 1],
            start_codon_pos = 26,
            repeat_pick=False,
        )
        with open(save_path, 'w') as f:
            yaml.dump(amino_blueprint, f)
    return amino_blueprint

trial_dir = '../../outputs/real_MRL_pred_MFE_260k'
AMINO_PATTERNS = get_amino_context_from_path(os.path.join(trial_dir, 'repaint_CAI_specification_redo', '7_amino_pattern_for_cai.yaml'))
STRATEGY = 'specific_adaptiveness'
GAMMA = 0.04
target_adaptiveness =  [0.65, 0.675, 0.7, 0.725, 0.75, 0.775]
                        #[0.5, 0.525, 0.55, 0.575, 0.6, 0.625]
                       # 0.8, 0.825, 0.85, 0.875, 0.9]

def refined_amino_constraint_experiment(trial_path):
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
        condition_weight=1,
        uncondition_prop=0.2,
    )

    ckpt_path = os.path.join(trial_path, 'checkpoints/epoch_2000.pt')
    checkpoint_dict = torch.load(ckpt_path, map_location='cpu')
    diffusion.load_state_dict(checkpoint_dict['model'])
    diffusion = accelerator.prepare(diffusion)

    target_labels = joint_target_values_sweep
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
        save_path = os.path.join(trial_path, 'repaint_CAI_specification_redo', STRATEGY + f'_adaptiveness_{adap}')

        os.makedirs(save_path, exist_ok=True)
        save_experiment_config(
            save_dir=save_path,
            checkpoint_path=ckpt_path,
            repaint=repaint,
            target_labels=target_labels,
            family_name='synonymous_codon_selection_strategy_benchmark',
            patterns=AMINO_PATTERNS,
            extra_config={
                "experiment_describe": "refined amino constraint benchmark using random amino-acid",
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
            repaint.setup(amino_list=pattern['amino'], pos_list=pattern['pos'], adaptiveness=adap)
            result = repaint.p_resample()

            write_fasta(result, os.path.join(save_path, pattern['name'] + '.fasta'), tgt_values=target_labels, batch_bs=100)


if __name__ == "__main__":
    refined_amino_constraint_experiment(trial_dir)
