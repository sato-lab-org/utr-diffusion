from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from tensorflow.python.layers.core import dropout
from src.data.dataloader_diy_data import load_data, load_data_without_dummy_label
from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.models.hytrans.hybrid_transformer_v3_0 import hybrid_transformer_v3_0 as HyTrans
#from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
from src.utils.train_multi_gpu import TrainLoop_multi_gpu as TrainLoop

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


def train():
    #data = load_data(data_path="data/MFE_30k_class_3_[-15, -7.5, 0.0].csv", split_ratio=0.05, label_type="MFE_label")
    data = load_data(data_path="../../data/HEK293/MRL_100k_class_3_[4, 6, 8].csv", split_ratio=0.05, label_type="MRL_label")
    #data = load_data_without_dummy_label(data_path="data/MRL_MFE_continuous_967k.csv", split_ratio=0.05)
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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # for unconditional training
    accelerator = Accelerator(log_with=["wandb"], mixed_precision='fp16', kwargs_handlers=[ddp_kwargs])
    model_save_name = "unconditional_967k_ep_2k_ts_200_beta_0.01_con_1_uncon_0.2_drop_0.2"
    accelerator.init_trackers(
        project_name="dnadiffusion",
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
        with_condition= False, # unconditional training
    ).train_loop()

if __name__ == "__main__":
    train()