from accelerate import Accelerator, DataLoaderConfiguration
from src.data.dataloader_diy_data import load_data_MRL_MFE_double_label
from src.models.diffusion import Diffusion
from src.models.unet_ml import UNet_Multi_Labels as UNet_ML
#from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
from src.utils.train_multi_gpu import TrainLoop_multi_gpu as TrainLoop

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def train():
    data = load_data_MRL_MFE_double_label(data_path="../../data/HEK293/MRL_MFE_5k_in_967k_multiclass_3x3.csv", split_ratio=0.05)

    unet = UNet_ML(
        dim=200, # 200
        channels=1,
        dim_mults=(1, 2, 4), # (1,2,4)
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2,
        num_labels = 2,
        num_classes = 3,
    )


    diffusion = Diffusion(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=4,
        uncondition_prop=0.2,
    )

    accelerator = Accelerator(log_with=["wandb"], mixed_precision='fp16')
    model_save_name = "3x3_5k(967k)_ep_2k_ts_200_beta_0.01_con_4_uncon_0.2_drop_0.2_lr_1e-4"
    accelerator.init_trackers(
        project_name="dnadiffusion_m",
        init_kwargs={"wandb": {"notes": model_save_name}},
    )

    TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        end_epoch=2000,
        log_step=5,   # how many steps to show the log トレーニングログを表示するステップ数
        valid_epoch=5,
        sample_epoch=200,
        save_epoch=2000,
        save_name=model_save_name,
        batch_size=6000,
        num_workers = 24,
        learning_rate=1e-4,
    ).train_loop()

if __name__ == "__main__":
    train()