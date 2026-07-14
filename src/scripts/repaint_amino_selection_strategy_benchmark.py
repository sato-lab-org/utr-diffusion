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

from src.experiment.exp_codon_pattern import build_random_quadra_amino_patterns
from src.experiment.exp_target_labels import joint_target_values_sweep

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def get_amino_context_from_path(save_path):
    if os.path.exists(save_path):
        quadra_amino_patterns = yaml.safe_load(open(save_path, 'r'))
    else:
        quadra_amino_patterns = build_random_quadra_amino_patterns(N=4, pos = [29, 35, 41, 47],)
        with open(save_path, 'w') as f:
            yaml.dump(quadra_amino_patterns, f)
    return quadra_amino_patterns

trial_dir = '../../outputs/real_MRL_pred_MFE_260k'
AMINO_PATTERNS = get_amino_context_from_path(os.path.join(trial_dir, 'repaint_strategy_benchmark', 'synonymous_codon_selection_strategy_benchmark.yaml'))
#STRATEGIES = ['init_random', 'init_usage', 'euclidean', 'wasserstein'] + ['usage_weighted_distance'] * 6
STRATEGIES = ['usage_weighted_distance'] #['init_random_fixed'] #+
GAMMAS = [0.04] #[None] #+

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

    # we need to run for several times, so today we just run for Quadra_Amino pattern as it has 81 variants
    for strategy, gamma in zip(STRATEGIES, GAMMAS):
        repaint = Repaint_Amino_CML(
            diffusion=diffusion,
            sample_bs=100,
            seq_len=50,
            cond_weight=2,
            return_all=False,
            tgt_labels=target_labels,
            strategy=strategy,
            gamma_for_usage_frequency=gamma,
            skip_frames=10,
        )
        repaint = accelerator.prepare(repaint)

        if gamma is None:
            save_path = os.path.join(trial_path, 'repaint_strategy_benchmark', strategy)
        else:
            save_path = os.path.join(trial_path, 'repaint_strategy_benchmark', strategy + f'_gamma_{gamma}')

        os.makedirs(save_path, exist_ok=True)
        save_experiment_config(
            save_dir=save_path,
            checkpoint_path=ckpt_path,
            repaint=repaint,
            target_labels=target_labels,
            family_name = 'synonymous_codon_selection_strategy_benchmark',
            patterns=AMINO_PATTERNS,
            extra_config={
                "experiment_describe": "refined amino constraint benchmark using random amino-acid",
                "strategy": strategy,
                "gamma": gamma,
            }
        )
        for pattern in AMINO_PATTERNS[:2]:
            if os.path.exists(os.path.join(save_path, pattern['name'] + '.fasta')):
                print(f'[INFO] Skip existing pattern: {pattern["name"]}')
                continue
            print(f'[INFO] Sampling for pattern: {pattern["name"]} with amino {pattern["amino"]} at positions {pattern["pos"]}')
            repaint.setup(amino_list=pattern['amino'], pos_list=pattern['pos'])
            result = repaint.p_resample()

            write_fasta(result, os.path.join(save_path, pattern['name'] + '.fasta'), tgt_values=target_labels, batch_bs=100)


if __name__ == "__main__":
    refined_amino_constraint_experiment(trial_dir)
