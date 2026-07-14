from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.models.repaint.repaint_amino_relax_constraint import RePaint_Amino_Relaxed
from src.models.repaint.utils import build_gt_mask_from_aminos, write_fasta
import torch
import warnings
import os
from src.experiment.exp_codon_pattern import Amino_Patterns
from src.experiment.exp_target_labels import joint_target_values_sweep, joint_target_values_3x3
import numpy as np
from src.experiment.exp_CDS_codon import build_amino_patterns_from_CDS_list

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def sample(checkpoint_path):
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

    target_labels = joint_target_values_3x3 #joint_target_values_sweep
    sweep_count = 0
    save_path = os.path.join(ROOT, 'repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''),'amino_relax_sweep')
    os.makedirs(save_path, exist_ok=True)
    cds_length = 30
    patterns = build_amino_patterns_from_CDS_list(n=cds_length)

    for reanchor_threshold in np.round(np.arange(0.9, 0.4, -0.1),1):
        for reanchor_interval in range(1, 11):
            repaint = RePaint_Amino_Relaxed(
                diffusion=diffusion,
                sample_bs=100,
                seq_len=50,
                cond_weight=2,
                return_all=False,
                tgt_labels=target_labels,
                skip_frames=1,
                reanchor_interval = reanchor_interval,
                reanchor_threshold = reanchor_threshold
            )
            repaint = accelerator.prepare(repaint)

            ## experiment of 5 fixed CDS pattern with various length
            for pattern in patterns:
                gt_image, mask = build_gt_mask_from_aminos(amino_list=pattern['amino'], pos_list=pattern['pos'])
                result = repaint.p_resample(gt=gt_image, mask=mask, tgt_aminos=pattern['amino'], pos_list=pattern['pos'])
                save_name = os.path.join(save_path, pattern['name'] + f'_interval_{reanchor_interval}_thres_{reanchor_threshold}')
                if isinstance(result, dict) and 'samples' in result:  # return all
                    torch.save(result, save_name + '.pt')
                    write_fasta(result['samples'], save_name + '.fasta', num_class=len(target_labels), batch_bs=100)
                else:  # return last
                    write_fasta(result, save_name + '.fasta', tgt_values=target_labels, batch_bs=100)
                sweep_count += 1
                print(f'Total Sweeping count: {sweep_count}/250')
            del repaint
            torch.cuda.empty_cache()

if __name__ == "__main__":
    checkpoint_path = os.path.join(ROOT, "checkpoints", "MRL_MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_epoch_2000.pt")
    sample(checkpoint_path)
