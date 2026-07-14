from accelerate import Accelerator
from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.models.repaint.repaint_amino import RePaint_Amino
import torch
import warnings
import os
from src.experiment.exp_codon_pattern import Amino_Patterns, pattern_dozen_amino
from src.models.repaint.utils import build_gt_mask_from_aminos, write_fasta

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# 3 experiments for amino with flexable codon using discrete single label diffusion
def experiment1(checkpoint_path):
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

    repaint = RePaint_Amino(
        diffusion=diffusion,
        sample_bs=100,
        seq_len=50,
        cond_weight=2,
        num_class=3,
        return_all=True
    )
    repaint = accelerator.prepare(repaint)
    for pattern in Amino_Patterns:
        gt_image, mask = build_gt_mask_from_aminos(amino_list=pattern['amino'] , pos_list=pattern['pos'])
        result = repaint.p_resample(gt=gt_image, mask=mask, tgt_aminos=pattern['amino'], pos_list=pattern['pos'])

        save_path = os.path.join('repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'amino')
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, 'pattern_' + pattern['name'])
        torch.save(result,  save_name + '.pt')
        write_fasta(result['samples'], save_name + '.fasta', num_class=3, batch_bs=100)


def experiment2(checkpoint_path):
    accelerator = Accelerator()

    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2
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

    strategies = ['init_random', 'wasserstein', 'euclidean']
    for strategy in strategies:
        repaint = RePaint_Amino(
            diffusion=diffusion,
            sample_bs=100,
            seq_len=50,
            cond_weight=2,
            num_class=3,
            strategy=strategy,
            return_all=True
        )
        repaint = accelerator.prepare(repaint)
        pattern = pattern_dozen_amino
        gt_image, mask = build_gt_mask_from_aminos(amino_list=pattern['amino'], pos_list=pattern['pos'])
        result = repaint.p_resample(gt=gt_image, mask=mask, tgt_aminos=pattern['amino'], pos_list=pattern['pos'])

        save_path = os.path.join('repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'amino_exp2')
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, strategy)
        torch.save(result, save_name + '.pt')
        write_fasta(result['samples'], save_name + '.fasta', num_class=3, batch_bs=100)


def experiment3(checkpoint_path):
    # experiment 3
    accelerator = Accelerator()

    unet = UNet(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2
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

    stop_points = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for stop_point in stop_points:
        repaint = RePaint_Amino(
            diffusion=diffusion,
            sample_bs=100,
            seq_len=50,
            cond_weight=2,
            num_class=3,
            strategy='euclidean',
            stop_point=stop_point,
            return_all=True
        )
        repaint = accelerator.prepare(repaint)
        pattern = pattern_dozen_amino
        gt_image, mask = build_gt_mask_from_aminos(amino_list=pattern['amino'], pos_list=pattern['pos'])
        result = repaint.p_resample(gt=gt_image, mask=mask, tgt_aminos=pattern['amino'], pos_list=pattern['pos'])

        save_path = os.path.join('repaint_save', checkpoint_path.split('/')[-1].replace('.pt', ''), 'amino_exp3')
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, f'stop_{stop_point}_all_frame')
        torch.save(result, save_name + '.pt')
        samples = result['samples'] if isinstance(result, dict) else result
        write_fasta(samples, save_name + '.fasta', num_class=3, batch_bs=100)

if __name__ == "__main__":
    checkpoint_path = 'checkpoints/20k_class_3_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_epoch_2000.pt'
    experiments(checkpoint_path)
