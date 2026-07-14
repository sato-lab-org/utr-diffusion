from accelerate import Accelerator
from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.models.repaint.repaint_codon import RePaintSampler
import torch
import warnings
import os
from src.experiment.exp_codon_pattern import Codon_Patterns, Codon_Patterns_Refined
from src.models.repaint.utils import bulid_gt_and_mask_from_codons, write_fasta

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample(checkpoint_path):
    accelerator = Accelerator()

    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2
    )

    diffusion = Diffusion(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=1,
        uncondition_prop=0.2,
    )
    checkpoint_dict = torch.load(checkpoint_path)
    diffusion.load_state_dict(checkpoint_dict['model'])
    diffusion = accelerator.prepare(diffusion)

    repaint = RePaintSampler(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        num_class=3,
        return_all=True
    )
    repaint = accelerator.prepare(repaint)

    for pattern in Codon_Patterns:
        gt, mask = bulid_gt_and_mask_from_codons(codon_list=pattern['codon'] , pos_list=pattern['pos'])
        result = repaint.p_resample(gt=gt, mask=mask)

        save_path = 'repaint_save/' + checkpoint_path.split('/')[-3] + '_at_' + checkpoint_path.split('/')[-1].replace('.pt', '')
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, 'pattern_' + pattern['name'])
        torch.save(result,  save_name + '.pt')
        write_fasta(result['samples'], save_name + '.fasta', num_class=3, batch_bs=100)


if __name__ == "__main__":
    checkpoint_path = '../../outputs/MRL_MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4/checkpoints/epoch_2000.pt'
    sample(checkpoint_path)
