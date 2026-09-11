from accelerate import Accelerator
from src.data.dataloader_diy_data import load_multiple_dataset_continuous_labels
from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels as Diffusion_MCML
from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels as UNet_MCML
#from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
from src.utils.train_multi_gpu import TrainLoop_multi_gpu as TrainLoop
from src.experiment.exp_target_labels import joint_target_values_3x3

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def train_masked_continuous_multi_label():
    datapaths = [
        'data/HEK293/real_MRL_pred_MFE_260k.csv',
        'data/HEK293/real_MRL_pred_MFE_260k_missing_MFE.csv',
        'data/HEK293/real_MRL_pred_MFE_260k_missing_MRL.csv'
    ]
    label_names = ['real_MRL', 'pred_MFE']  # real_MRL, pred_MFE


    label_emb_mode = 'per_label'
    label_wise_mask = False
    uncondition_prop = 0.2
    LR = 2e-4

    data = load_multiple_dataset_continuous_labels(data_paths=datapaths, label_cols=label_names, split_ratio=0.05)

    targets = joint_target_values_3x3

    unet = UNet_MCML(
        dim=200, # 200
        channels=1,
        dim_mults=(1, 2, 4), # (1,2,4)
        resnet_block_groups=4, # 4
        seq_len = 50,
        dropout = 0.2,
        num_label = len(label_names),
        label_emb_mode = label_emb_mode, # 'per_label'
    )


    diffusion = Diffusion_MCML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=4,
        uncondition_prop=uncondition_prop,
        label_wise_mask=label_wise_mask, #True,
    )

    accelerator = Accelerator(log_with=["wandb"], mixed_precision='fp16',)
    #model_save_name = f"Ecoli_{len(label_names)}labels_cfg_perlabel_uncond{uncondition_prop}_global_lr{LR}_fixed"
    model_save_name = "real_MRL_pred_MFE_260k"
    label_names_str = "-".join(label_names)
    notes = (
    f"label_emb_mode={label_emb_mode}_"
    f"label_wise_mask={label_wise_mask}_"
    f"uncond={uncondition_prop}_"
    f"global_lr={LR}"
    )
    accelerator.init_trackers(
        project_name="UTR-Diffusion",
        init_kwargs={
            "wandb": {
                "name": model_save_name,
                "notes": notes,
                "tags":label_names
            },
        }
    )

    TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        end_epoch=2000,
        log_step=10,   # how many steps to show the log トレーニングログを表示するステップ数
        valid_epoch=20,
        sample_epoch=4001,
        save_epoch=2000,
        save_name=model_save_name,
        batch_size=3000,
        num_workers = 16,
        learning_rate=LR,
        tgt_values = targets,
        label_names = label_names,
        seq_len = 50,
    ).train_loop()

if __name__ == "__main__":
    train_masked_continuous_multi_label()