from accelerate import Accelerator
from src.data.dataloader_diy_data import load_data_with_label_name
from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels as Diffusion_MCML
from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels as UNet_MCML
#from src.utils.train_single_gpu import TrainLoop_single_gpu as TrainLoop
from src.utils.train_multi_gpu import TrainLoop_multi_gpu as TrainLoop
from src.utils.create_targets import get_targets_from_data_with_HandL_values

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
#PYTHONPATH=$(pwd) accelerate launch --config_file environments/accelerator_config.yaml src/scripts/train_cml_ecoli.py

def train_continuous_multi_label():
    datapath = "data/E_Coli/Supplementary_Data_15_Integrated_Phenotypic_Measurements_with_MFE.csv"
    label_names = ['polysome.mean.load', 'clean.lin.prot.mean', 'ss.rna.dna.mean', 'MFE']
    cfg_mode = 'per_label'
    label_wise_mask = False
    uncondition_prop = 0.2
    LR = 2e-4

    data = load_data_with_label_name(
        datapath=datapath,
        max_seq_len=96,
        seq_col='gs.sequence',
        labels=label_names,
        do_normalize=True,
        log2_transform_labels=['ss.rna.dna.mean'],
        split_ratio=0.1)

    targets = get_targets_from_data_with_HandL_values(data, label_names=label_names, H_val=0.95, L_val=0.05)

    unet = UNet_MCML(
        dim=200, # 200
        channels=1,
        dim_mults=(1, 2, 4), # (1,2,4)
        resnet_block_groups=4, # 4
        seq_len = 96,
        dropout = 0.2,
        num_label = len(label_names),
        label_emb_mode = cfg_mode, # 'per_label'
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
    model_save_name = f"Ecoli_{len(label_names)}labels_cfg_perlabel_uncond{uncondition_prop}_global_lr{LR}_fixed"
    label_names_str = "-".join(label_names)
    notes = (
    f"Ecoli_labels={label_names_str}_"
    f"cfg={cfg_mode}_"
    f"uncond={uncondition_prop}_"
    f"global_lr={LR}"
)
    accelerator.init_trackers(
        project_name="UTR-Diffusion_E.Coli",
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
        save_epoch=500,
        save_name=model_save_name,
        batch_size=3000,
        num_workers = 16,
        learning_rate=LR,
        tgt_values = targets,
        label_names = label_names,
        seq_len = 96,
    ).train_loop()

if __name__ == "__main__":
    train_continuous_multi_label()