from accelerate import Accelerator
from src.data.dataloader_diy_data import load_multiple_dataset_continuous_labels
from src.models.mcml_config import (
    MCML_DIFFUSION_KWARGS,
    MCML_LABEL_NAMES,
    MCML_TRAIN_CHECKPOINT_DIR,
    MCML_TRAIN_OUTPUT_DIR,
    MCML_UNET_KWARGS,
    build_mcml_diffusion,
)
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
    label_names = list(MCML_LABEL_NAMES)


    label_emb_mode = MCML_UNET_KWARGS["label_emb_mode"]
    label_wise_mask = MCML_DIFFUSION_KWARGS["label_wise_mask"]
    uncondition_prop = MCML_DIFFUSION_KWARGS["uncondition_prop"]
    LR = 2e-4

    data = load_multiple_dataset_continuous_labels(
        data_paths=datapaths,
        label_cols=label_names,
        split_ratio=0.05,
        seq_len=MCML_UNET_KWARGS["seq_len"],
    )

    targets = joint_target_values_3x3

    diffusion = build_mcml_diffusion()

    accelerator = Accelerator(log_with=["wandb"], mixed_precision='fp16',)
    #model_save_name = f"Ecoli_{len(label_names)}labels_cfg_perlabel_uncond{uncondition_prop}_global_lr{LR}_fixed"
    model_save_name = MCML_TRAIN_OUTPUT_DIR
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
        checkpoint_dir=MCML_TRAIN_CHECKPOINT_DIR,
        batch_size=3000,
        num_workers = 16,
        learning_rate=LR,
        tgt_values = targets,
        label_names = label_names,
        seq_len=MCML_UNET_KWARGS["seq_len"],
    ).train_loop()

if __name__ == "__main__":
    train_masked_continuous_multi_label()
