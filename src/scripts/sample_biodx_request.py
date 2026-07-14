from src.experiment.exp_target_labels import joint_target_values_sweep, CML_Crest_request
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
        condition_weight=1,
        uncondition_prop=0.2,
    )
    accelerator = Accelerator()

    # Crest BioDX team work Request
    tgt_values = CML_Crest_request
    model_save_name = "MRL_MFE_260k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_sample_CREST_Request"
    checkpoint_path = ("checkpoints/MRL_MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_epoch_2000.pt")
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
        do_gumbel_softmax=False,
        tgt_values = tgt_values
    ).load_checkpoint_then_do_sample(checkpoint_path)

if __name__ == "__main__":
    sample_continuous_multi_label()