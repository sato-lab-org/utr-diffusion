from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.models.repaint.repaint_codon_cml import RePaint_Codon_Continuous_Multi_Labels as Repaint_Codon_CML
import torch
import warnings
import os
from src.models.repaint.utils import bulid_gt_and_mask_from_codons, write_fasta
from src.experiment.exp_configuration import save_experiment_config
from src.experiment.exp_target_labels import joint_target_values_sweep
from src.experiment.exp_codon_pattern import (
    Kozak,
    build_DRACH_at_pos,
    build_DRACH_at_pos_with_Kozak,
    build_Kozak_DRACH_with_triplet,
)


warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


CODON_PATTERNS = {
    'Kozak': Kozak,
    'DRACH': build_DRACH_at_pos(),
    'Kozak+DRACH': build_DRACH_at_pos_with_Kozak(),
    'Kozak+DRACH+triplet': build_Kozak_DRACH_with_triplet()

}


def refined_codon_constraint_experiment(checkpoint_path):
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

    checkpoint_dict = torch.load(checkpoint_path)
    diffusion.load_state_dict(checkpoint_dict['model'])
    diffusion = accelerator.prepare(diffusion)

    target_labels = joint_target_values_sweep

    repaint = Repaint_Codon_CML(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        tgt_labels=target_labels,
        return_all=False,
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)
    # Kozak
    for family_name, patterns in CODON_PATTERNS.items():
        save_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), 'repaint', family_name)
        os.makedirs(save_path, exist_ok=True)

        save_experiment_config(
            save_dir=save_path,
            checkpoint_path=checkpoint_path,
            repaint=repaint,
            target_labels=target_labels,
            patterns=patterns,
            family_name = family_name,
            extra_config={
                "experiment_describe": "refined codon constraint benchmark using exact motif",
                "pattern_family": family_name,
            }
        )

        for pattern in patterns:
            if os.path.exists(os.path.join(save_path, pattern['name'] + '.fasta')):
                print(f'[INFO] Skip existing pattern: {pattern["name"]}')
                continue
            print(f'[INFO] Sampling for pattern: {pattern["name"]} with codons {pattern["codon"]} at positions {pattern["pos"]}')
            gt, mask = bulid_gt_and_mask_from_codons(codon_list=pattern['codon'] , pos_list=pattern['pos'])
            result = repaint.p_resample(gt=gt, mask=mask)
            write_fasta(result, os.path.join(save_path, pattern['name'] + '.fasta'), tgt_values=target_labels, batch_bs=100)


if __name__ == "__main__":
    checkpoint_path = '../../outputs/real_MRL_pred_MFE_260k/checkpoints/epoch_2000.pt'
    refined_codon_constraint_experiment(checkpoint_path)
