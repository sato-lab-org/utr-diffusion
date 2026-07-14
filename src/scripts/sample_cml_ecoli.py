from accelerate import Accelerator
from src.data.dataloader_diy_data import load_data_with_label_name
from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels as Diffusion_MCML
from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels as UNet_MCML
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
#from src.utils.train_multi_gpu import TrainLoop_multi_gpu as TrainLoop
from src.utils.create_targets import get_targets_from_data_with_HandL_values, get_targets_from_data_with_sigma_grid, get_targets_extrapolation

import warnings
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample_continuous_multi_label():
    cfg_mode = 'per_label'
    label_wise_mask = False
    uncondition_prop = 0.2

    datapath = "data/E_Coli/Supplementary_Data_15_Integrated_Phenotypic_Measurements_with_MFE.csv"
    label_names = ['polysome.mean.load', 'clean.lin.prot.mean', 'ss.rna.dna.mean', 'MFE']
    data = load_data_with_label_name(
        datapath=datapath,
        max_seq_len=96,
        seq_col='gs.sequence',
        labels=label_names,
        do_normalize=True,
        log2_transform_labels=['ss.rna.dna.mean'],
        split_ratio=0.1)
    # I need a better way to set target values
    #targets = get_targets_from_data_with_HandL_values(data, label_names=label_names, H_val=0.95, L_val=0.05)
    targets_sweep = get_targets_from_data_with_sigma_grid(data, label_names, sigma_levels=(-2, -1, 0, 1, 2),
        coupled_value_groups=[
            ["polysome.mean.load", "clean.lin.prot.mean"]
        ],
    )
    # targets_extrapolation = get_targets_extrapolation(data, label_names,
    #     controlled_labels=(1, 2, 3),
    #     other_label=np.nan,
    #     target_levels=(-4.5, 4.5),
    #     coupled_value_groups=[
    #         ["polysome.mean.load", "clean.lin.prot.mean"]
    #     ],
    #     deduplicate_coupled_groups=True,
    # )

    # targets_interpolation = get_targets_extrapolation(data, label_names,
    #     controlled_labels=(1,),
    #     other_label=np.nan,
    #     target_levels=np.arange(-4.0, 4.01, 0.1),
    #     coupled_value_groups=[
    #     ["polysome.mean.load", "clean.lin.prot.mean"]
    #     ],
    #     deduplicate_coupled_groups=True,
    # )

    unet = UNet_MCML(
        dim=200, # 200
        channels=1,
        dim_mults=(1, 2, 4), # (1,2,4)
        resnet_block_groups=4, # 4
        seq_len = 96,
        dropout = 0.2,
        num_label = len(label_names),
        label_emb_mode = cfg_mode
    )

    diffusion = Diffusion_MCML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=4,
        uncondition_prop=uncondition_prop,  # True,
        label_wise_mask=label_wise_mask,  # True,
    )

    accelerator = Accelerator()
    save_name = "outputs/Ecoli_4labels_cfg_perlabel_uncond0.2_global_lr2e-4_fixed_sample_sweep"
    checkpoint_path = "checkpoints/Ecoli_4labels_cfg_perlabel_uncond0.2_global_lr2e-4_fixed_epoch_500.pt"

    TrainLoop(
        data={},
        model=diffusion,
        accelerator=accelerator,
        end_epoch=4000,
        log_step=10,   # how many steps to show the log トレーニングログを表示するステップ数
        valid_epoch=10,
        sample_epoch=500,
        save_epoch=1000,
        save_name=save_name,
        batch_size=3000,
        num_workers = 16,
        learning_rate=2e-4,
        label_names=label_names,
        tgt_values = targets_sweep,
        seq_len = 96,
    ).load_checkpoint_then_do_sample(checkpoint_path, 'sample_sweep_labelwise', sample_bs=1000)

if __name__ == "__main__":
    sample_continuous_multi_label()