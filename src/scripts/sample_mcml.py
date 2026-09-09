# This script is used for testing the conditional fidelity of masked continuous multi label generation
from src.experiment.exp_target_labels import joint_target_values_3x3, joint_target_values_sweep
from accelerate import Accelerator
from src.models.mcml_config import (
    MCML_LABEL_NAMES,
    MCML_LOCAL_CHECKPOINT_PATH,
    MCML_TRAIN_CHECKPOINT_DIR,
    MCML_TRAIN_OUTPUT_DIR,
    MCML_UNET_KWARGS,
    build_mcml_diffusion,
)
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample_masked_continuous_multi_label():
    diffusion = build_mcml_diffusion()
    accelerator = Accelerator(mixed_precision='fp16')

    model_save_name = MCML_TRAIN_OUTPUT_DIR
    checkpoint_path = MCML_LOCAL_CHECKPOINT_PATH
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
        checkpoint_dir=MCML_TRAIN_CHECKPOINT_DIR,
        batch_size=4500,
        num_workers=16,
        learning_rate=1e-4,
        label_names=list(MCML_LABEL_NAMES),
        tgt_values=joint_target_values_sweep,#joint_tgt_values
        seq_len=MCML_UNET_KWARGS["seq_len"],
    ).load_checkpoint_then_sample_offline(
        checkpoint_path,
        trial_name='joint_target_values_sweep',
        sample_bs=100,
    )
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
    # ).load_checkpoint_then_sample_offline(checkpoint_path, trial_name='MRL_[4.0, 6.0, 8.0]')
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
    # ).load_checkpoint_then_sample_offline(checkpoint_path, trial_name='MFE_[-20, -10, -2]')
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
    # ).load_checkpoint_then_sample_offline(checkpoint_path, trial_name='MRL_9.0_3k_samples', sample_bs=3000)

if __name__ == "__main__":
    sample_masked_continuous_multi_label()
