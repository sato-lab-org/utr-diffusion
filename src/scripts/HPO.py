from accelerate import Accelerator, DataLoaderConfiguration
from sympy.matrices.expressions.kronecker import validate

# from src.data.dataloader import load_data
from src.data.dataloader_diy_data import load_data
from src.models.diffusion import Diffusion
from src.models.unet import UNet
from src.utils.train_single_gpu import TrainLoop
from src.config.HP_setup import load_cfg
import optuna
import json
import wandb

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

def train_with_optuna(trial, params, data):

    params = {'time_steps': trial.suggest_categorical('time_steps', search_space['time_steps']),
              'unet_dim': params.unet_dim,
              'dropout': params.dropout_rate,
              'beta_last': params.beta_last,
              'cond_weight': params.cond_weight,
              'uncond_prop': params.uncond_prop,
              'end_epoch': params.end_epoch,
              }


    unet = UNet(
        dim=params['unet_dim'],  # 200
        channels=1,
        dim_mults=(1, 2, 4),  # (1,2,4)
        resnet_block_groups=4,  # 4
        seq_len=50,
        dropout=params['dropout'],
    )

    diffusion = Diffusion(
        model = unet,
        timesteps=params['time_steps'],
        beta_last = params['beta_last'],
        condition_weight=params['cond_weight'],
        uncondition_prop=params['uncond_prop']
    )

    accelerator = Accelerator(log_with=["wandb"])
    print(f'Acclerator device: {accelerator.device}')
    accelerator.init_trackers(
        project_name="vanilla_diffusion_HPO",
        init_kwargs={
            "wandb": {
                "name": f"ts_{params['time_steps']}",
                "group": f"TS_10k_class_5_ts_50_beta_0.2_cond_1_uncond_0.1_drop_0.2",
                "notes": json.dumps(params)
            }
        },
    )

    valid_loss = TrainLoop(
        data=data,
        model=diffusion,
        accelerator=accelerator,
        end_epoch=1000,
        log_step=4,
        valid_epoch= 5,
        sample_epoch=100,
        save_epoch=999999,
        model_name='HPO_for_Timesteps_10k_class_5_ts_50_beta_0.2_cond_1_uncond_0.2_drop_0.2',
        batch_size=3750,
        num_workers=4,
        learning_rate=1e-4,
        trial_name = f"ts_{params['time_steps']}",
        do_validation= True,
    ).train_loop()

    wandb.finish()

    return valid_loss

if __name__ == "__main__":
    dataset = load_data(saved_data_path="../../data/HEK293/MRL_10k_class_5_[4-8].csv", classes=5, split_ratio=0.05)
    fix_params = load_cfg('../config/HPO_timesteps.yaml')
    search_space = {'time_steps': [500,1000]}
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(sampler = sampler, direction='minimize')
    study.optimize(lambda trial: train_with_optuna(trial, fix_params, dataset), n_trials=len(search_space['time_steps']))

