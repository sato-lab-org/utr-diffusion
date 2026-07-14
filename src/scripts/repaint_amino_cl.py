from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.models.repaint.repaint_amino_cml import RePaint_Amino_Continuous_Multi_Labels as Repaint_Amino_CML
from src.models.repaint.utils import build_gt_mask_from_aminos, write_fasta
import torch
import warnings
import os
from src.experiment.exp_codon_pattern import Amino_Patterns
from src.experiment.exp_target_labels import joint_target_values_sweep, joint_target_values_3x3, mrl_target_values_3, mrl_target_extreme

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample(checkpoint_path):
    accelerator = Accelerator()

    unet = UNet_CML(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2,
        num_label=1, # single label set num_label=1
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

    target_labels = mrl_target_extreme #joint_target_values_3x3 #joint_target_values_sweep
    repaint = Repaint_Amino_CML(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        return_all=False,
        tgt_labels=target_labels,
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)

    ## experiment of 5 fixed CDS pattern with various length
    import numpy as np
    from src.experiment.exp_CDS_codon import build_amino_patterns_from_CDS_list
    CDS_length_range = np.arange(6, 50, 3)
    for cds_length in CDS_length_range:
        patterns = build_amino_patterns_from_CDS_list(n=cds_length)
        for pattern in patterns:
            gt_image, mask = build_gt_mask_from_aminos(amino_list=pattern['amino'], pos_list=pattern['pos'])
            result = repaint.p_resample(gt=gt_image, mask=mask, tgt_aminos=pattern['amino'], pos_list=pattern['pos'])

            save_path = os.path.join('../../repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'amino_CDS_exp')
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, pattern['name'])
            # function Repaint_Codon_CML arg return_all must be set as False
            write_fasta(result, save_name + '.fasta', tgt_values=target_labels, batch_bs=100)

if __name__ == "__main__":
    checkpoint_path = '../../checkpoints/MRL_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_2000epoch.pt'
    sample(checkpoint_path)
