from accelerate import Accelerator
from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def sample():
    unet = UNet(
        dim=200, # 200
        channels=1,
        dim_mults=(1, 2, 4), # (1,2,4)
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

    accelerator = Accelerator()
    model_save_name = "N_20k_class_3_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4"
    checkpoint_path = "../../checkpoints/20k_class_3_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_epoch_2000.pt"

    TrainLoop(
        data={},
        model=diffusion,
        accelerator=accelerator,
        end_epoch=2000,
        log_step=3,   # how many steps to show the log トレーニングログを表示するステップ数
        valid_epoch=2,
        sample_epoch=100,
        save_epoch=2000,
        save_name=model_save_name,
        batch_size=3000,
        num_workers = 4,
        learning_rate=1e-4,
        do_gumbel_softmax=False,
    ).load_checkpoint_then_do_sample(checkpoint_path)

if __name__ == "__main__":
    sample()