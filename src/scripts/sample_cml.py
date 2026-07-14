from src.experiment.exp_target_labels import joint_target_values_sweep, joint_target_values_3x3, CML_Crest_request
from accelerate import Accelerator
from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def sample_continuous_multi_label():
    unet = UNet_CML(
        dim=200,  # 200
        channels=1,
        dim_mults=(1, 2, 4),  # (1,2,4)
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=0.2,
        num_label=2,
    )

    diffusion = Diffusion_CML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=4,
        uncondition_prop=0.2,
    )
    accelerator = Accelerator(mixed_precision='fp16')
    
    tgt_values = joint_target_values_3x3 #joint_target_values_sweep
    model_save_name = "outputs/real_MRL_pred_MFE_601k"
    checkpoint_path = model_save_name + '/checkpoints/epoch_2000.pt'
    TrainLoop(
        data={},
        model=diffusion,
        accelerator=accelerator,
        end_epoch=2000,
        log_step=8,   # how many steps to show the log トレーニングログを表示するステップ数
        valid_epoch=5,
        sample_epoch=200,
        save_epoch=2000,
        save_name=model_save_name,
        batch_size=5200,
        num_workers = 16,
        learning_rate=1e-4,
        tgt_values = tgt_values
    ).load_checkpoint_then_do_sample(checkpoint_path, trial_name='cond_4')

if __name__ == "__main__":
    sample_continuous_multi_label()