from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.models.repaint.repaint_codon_cml import RePaint_Codon_Continuous_Multi_Labels as Repaint_Codon_CML
import torch
import warnings
import os
from src.experiment.exp_codon_pattern import Codon_Patterns, Codon_Patterns_Refined
from src.experiment.exp_target_labels import joint_target_values_3x3, joint_target_values_2x2, joint_target_values_sweep
from src.models.repaint.utils import bulid_gt_and_mask_from_codons, write_fasta
from src.models.repaint.scheduler import schedule_jump_params, schedule_no_jump_params
import numpy as np
from src.experiment.exp_CDS_codon import build_codon_patterns_from_CDS_list

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample(checkpoint_path, exp_mode = 'codon_pattern'): #exp_mode = 'codon_pattern' or 'various_length'
    accelerator = Accelerator()

    unet = UNet_CML(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2,
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

    target_labels = joint_target_values_3x3
    #target_labels = joint_target_values_sweep

    repaint = Repaint_Codon_CML(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        tgt_labels= target_labels,
        return_all=False,
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)
    ### experiment of various patterns
    if exp_mode == 'codon_pattern':
        for pattern in Codon_Patterns:
            gt, mask = bulid_gt_and_mask_from_codons(codon_list=pattern['codon'] , pos_list=pattern['pos'])
            result = repaint.p_resample(gt=gt, mask=mask)
            save_path = os.path.join('repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'codon')
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, 'pattern_' + pattern['name'])

            if isinstance(result, dict) and 'samples' in result:  # return all
                torch.save(result, save_name + '.pt')
                write_fasta(result['samples'], save_name + '.fasta', tgt_values=target_labels, batch_bs=100)
            else:  # return last
                write_fasta(result, save_name + '.fasta', tgt_values=target_labels, batch_bs=100)

    # ### experiment of 5 fixed CDS pattern with various length
    if exp_mode == 'various_length':
        CDS_length_range = np.arange(6, 50, 3)
        for cds_length in CDS_length_range:
            patterns = build_codon_patterns_from_CDS_list(n=cds_length)
            for pattern in patterns:
                gt, mask = bulid_gt_and_mask_from_codons(codon_list=pattern['codon'], pos_list=pattern['pos'])
                result = repaint.p_resample(gt=gt, mask=mask)

                save_path = os.path.join('repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'codon_CDS_exp', f'len_{cds_length}nt')
                os.makedirs(save_path, exist_ok=True)
                save_name = os.path.join(save_path, pattern['name'])
                # function Repaint_Codon_CML arg return_all must be set as False
                write_fasta(result, save_name + '.fasta', tgt_values=target_labels, batch_bs=100)

        # ### experiment of 5 fixed CDS pattern with various length, using repaint without renoising
        # CDS_length_range = np.arange(6, 50, 3)
        # for cds_length in CDS_length_range:
        #     patterns = build_codon_patterns_from_CDS_list(n=cds_length)
        #     for pattern in patterns:
        #         gt, mask = bulid_gt_and_mask_from_codons(codon_list=pattern['codon'], pos_list=pattern['pos'])
        #         result = repaint.p_resample(gt=gt, mask=mask, schedule=schedule_no_jump_params)
        #
        #         save_path = os.path.join('../../repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'codon_CDS_exp_no_renoising', f'len_{cds_length}nt')
        #         os.makedirs(save_path, exist_ok=True)
        #         save_name = os.path.join(save_path, pattern['name'])
        #         # function Repaint_Codon_CML arg return_all must be set as False
        #         write_fasta(result, save_name + '.fasta', tgt_values=target_labels, batch_bs=100)



def sample_refined(checkpoint_path, exp_mode = 'codon_pattern'):
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

    # target_labels = joint_target_values_3x3
    target_labels = joint_target_values_sweep

    repaint = Repaint_Codon_CML(
        diffusion=diffusion,
        sample_bs=25,
        seq_len=50,
        cond_weight=2,
        tgt_labels=target_labels,
        return_all=False,
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)
    if exp_mode == 'codon_pattern':
        for pattern in Codon_Patterns_Refined:
            gt, mask = bulid_gt_and_mask_from_codons(codon_list=pattern['codon'] , pos_list=pattern['pos'])
            result = repaint.p_resample(gt=gt, mask=mask)

            save_path = os.path.join('repaint_save', checkpoint_path.split('/')[-3] + '_at_' + checkpoint_path.split('/')[-1].replace('.pt', ''), 'refined_codon_combination')
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, 'pattern_' + pattern['name'])
            write_fasta(result, save_name + '.fasta', tgt_values=target_labels, batch_bs=25)



if __name__ == "__main__":
    checkpoint_path = '../../outputs/MRL_MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4/checkpoints/epoch_2000.pt'
    sample_refined(checkpoint_path)
