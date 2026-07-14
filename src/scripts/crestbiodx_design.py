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

from src.experiment.crestbiodx_3seqs import get_CREST_amino_patterns

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

trial_path = '../../outputs/real_MRL_pred_MFE_260k'

def CREST_bioDX_sequence_design():
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

    #target_labels = [[8.0, -20.0],[8.0, -25.0], [9.0, -20.0], [9.0, -25.0]] # we try a very high MRL and high MFE
    target_labels = [[8.0, 0.0], [8.0, -2.0], [9.0, 0.0], [9.0, -2.0]]  # I made a mistake in the previous line, High MFE is not high absolute value

    repaint = Repaint_Amino_CML(
        diffusion=diffusion,
        sample_bs=1000,
        seq_len=50,
        cond_weight=2,
        return_all=False,
        tgt_labels=target_labels,
        strategy='specific_adaptiveness',
        skip_frames=10,
    )
    repaint = accelerator.prepare(repaint)

    # we need to run for several times, so today we just run for Quadra_Amino pattern as it has 81 variants
    amino_patterns = get_CREST_amino_patterns()

    save_path = os.path.join(trial_path, 'CREST_bioDX_request')
    os.makedirs(save_path, exist_ok=True)

    save_experiment_config(
        save_dir=save_path,
        checkpoint_path=ckpt_path,
        repaint=repaint,
        target_labels=target_labels,
        patterns=amino_patterns,
        family_name = 'CREST_bioDX',
        extra_config={
            "experiment_describe": "CREST-bioDX request, 3 reference sequences from Terai-sensei",
        }
    )

    for pattern in amino_patterns:
        if os.path.exists(os.path.join(save_path, pattern['name'] + '.fasta')):
            print(f'[INFO] Skip existing pattern: {pattern["name"]}')
            continue
        print(f'[INFO] Sampling for pattern: {pattern["name"]} with amino {pattern["amino"]} at positions {pattern["pos"]}')
        repaint.setup(amino_list=pattern['amino'], pos_list=pattern['pos'], adaptiveness=0.9)
        result = repaint.p_resample()

        write_fasta(result, os.path.join(save_path, pattern['name'] + '.fasta'), tgt_values=target_labels, batch_bs=1000)


if __name__ == "__main__":
    CREST_bioDX_sequence_design()
