# This script is used for testing the conditional fidelity of masked continuous multi label generation
from src.experiment.exp_target_labels import joint_target_values_3x3, joint_target_values_sweep
from accelerate import Accelerator
from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels as Diffusion_MCML
from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels as UNet_MCML
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample_masked_continuous_multi_label():
    unet = UNet_MCML(
        dim=200, # 200
        channels=1,
        dim_mults=(1, 2, 4), # (1,2,4)
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2,
        num_label = 2,
        label_emb_mode = 'per_label'
    )

    diffusion = Diffusion_MCML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=4,
        uncondition_prop=0.2,
        label_wise_mask=False
    )
    accelerator = Accelerator(mixed_precision='fp16')

    model_save_name = "outputs/real_MRL_pred_MFE_260k_mcml"
    checkpoint_path = model_save_name + '/checkpoints/epoch_2000.pt'
    # joint_tgt_values = joint_target_values_3x3  # joint_target_values
    # MRL_tgt_values =  [[4.0, np.nan], [6.0, np.nan], [8.0, np.nan]]  # MRL_values
    # MFE_tgt_values = [[np.nan, -20.0], [np.nan, -10], [np.nan, -2]]  # MFE_values
    #
    # # joint multi-label
    TrainLoop(
        data={},
        model=diffusion,
        accelerator=accelerator,
        end_epoch=2000,
        log_step=8,  # how many steps to show the log トレーニングログを表示するステップ数
        valid_epoch=5,
        sample_epoch=200,
        save_epoch=2000,
        save_name=model_save_name,
        batch_size=4500,
        num_workers=16,
        learning_rate=1e-4,
        tgt_values=joint_target_values_sweep#joint_tgt_values
    ).load_checkpoint_then_do_sample(checkpoint_path, trial_name='joint_target_values_sweep', sample_bs=100)
    #
    # # MRL
    # TrainLoop(
    #     data={},
    #     model=diffusion,
    #     accelerator=accelerator,
    #     end_epoch=2000,
    #     log_step=8,  # how many steps to show the log トレーニングログを表示するステップ数
    #     valid_epoch=5,
    #     sample_epoch=200,
    #     save_epoch=2000,
    #     save_name=model_save_name,
    #     batch_size=4500,
    #     num_workers=16,
    #     learning_rate=1e-4,
    #     tgt_values=MRL_tgt_values
    # ).load_checkpoint_then_do_sample(checkpoint_path, trial_name='MRL_[4.0, 6.0, 8.0]')
    #
    # # MFE
    # TrainLoop(
    #     data={},
    #     model=diffusion,
    #     accelerator=accelerator,
    #     end_epoch=2000,
    #     log_step=8,  # how many steps to show the log トレーニングログを表示するステップ数
    #     valid_epoch=5,
    #     sample_epoch=200,
    #     save_epoch=2000,
    #     save_name=model_save_name,
    #     batch_size=4500,
    #     num_workers=16,
    #     learning_rate=1e-4,
    #     tgt_values=MFE_tgt_values
    # ).load_checkpoint_then_do_sample(checkpoint_path, trial_name='MFE_[-20, -10, -2]')
    #

    # try to sample a maximum single target value
    # tgt_values = [[9.0, np.nan]]
    # TrainLoop(
    #     data={},
    #     model=diffusion,
    #     accelerator=accelerator,
    #     end_epoch=2000,
    #     log_step=8,  # how many steps to show the log トレーニングログを表示するステップ数
    #     valid_epoch=5,
    #     sample_epoch=200,
    #     save_epoch=2000,
    #     save_name=model_save_name,
    #     batch_size=4500,
    #     num_workers=16,
    #     learning_rate=1e-4,
    #     tgt_values=tgt_values
    # ).load_checkpoint_then_do_sample(checkpoint_path, trial_name='MRL_9.0_3k_samples', sample_bs=3000)

if __name__ == "__main__":
    sample_masked_continuous_multi_label()