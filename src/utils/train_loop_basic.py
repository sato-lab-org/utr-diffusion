import os
import copy
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from src.utils.utils import EMA
from src.data.dataloader import SequenceDataset
from torch.utils.data import DataLoader
from src.utils.utils import write_to_fasta


class BasicTrainLoop():
    def __init__(self,
                 model: torch.nn.Module,
                 accelerator: Accelerator,
                 start_epoch: int = 1,
                 end_epoch: int = 10000,
                 log_step: int = 50,
                 valid_epoch: int = 5,
                 sample_epoch: int = 500,
                 save_epoch: int = 500,
                 save_name: str = '',
                 batch_size: int = 960,
                 num_workers: int = 4,
                 learning_rate: float = 1e-3,
                 num_classes: int = 3,
                 seq_len:int = 50,
                 ):
        # Model, Optimizer and Accelerator
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=2e-4, betas=(0.9, 0.95)) # 2e-4
        self.accelerator = accelerator
        print(f'Accelerator device: {accelerator.device}')

        # Parameters and Hyper-parameters
        self.sample_epoch, self.save_epoch, self.valid_epoch = sample_epoch, save_epoch, valid_epoch
        self.start_epoch, self.end_epoch, self.log_step = start_epoch, end_epoch, log_step
        self.seq_similarity, self.global_step, self.train_loss, self.valid_loss, self.recon_loss = 0, 0, 0.0, 0.0, 0.0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.peak_lr = learning_rate
        self.save_name= save_name
        self.best_valid_loss = float('inf')
        self.num_classes = num_classes
        self.is_save_process = False

        self.rec_count = 0

        # multi-gpu setting
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_checkpoint_load = False


    def _prepare_data_loader(self, data):
        if data != {}:  # case "data={}" for sample only
            seq_train = SequenceDataset(seqs=data["Train"], c=data['Train_label'])
            seq_valid = SequenceDataset(seqs=data["Valid"], c=data['Valid_label'])
            train_dl = DataLoader(seq_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
            valid_dl = DataLoader(seq_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)
            return train_dl, valid_dl
        else:
            print('Training data not provided, running in sampling mode!!')
            return None, None


    def log_update(self, mode, epoch: int = 0):
        # always unwrap model for safe access
        model_for_log = self.accelerator.unwrap_model(self.model)

        if mode == 'init':
            param_size = round(sum(p.numel() for p in model_for_log.parameters()) / 1e6, 2)
            self.accelerator.log(
                {
                    'peak learning rate': self.peak_lr,
                    'timestep': getattr(model_for_log, 'timestep', None),
                    'beta': getattr(model_for_log, 'beta_last', None),
                    'conditional weight': getattr(model_for_log, 'cond_weight', None),
                    'unconditional ratio': getattr(model_for_log, 'uncond_prop', None),
                    'dropout': getattr(model_for_log.model, 'dropout_rate', None),
                    'Unet dimension': getattr(model_for_log.model, 'dim', None),
                    'max epoch': self.end_epoch,
                    'data class number': self.num_classes,
                    'data loader workers': self.num_workers,
                    'batch size': self.batch_size,
                    'model_size_M': param_size,
                }
            )

        elif mode == 'train':
            self.accelerator.log(
                {
                    'train loss': self.train_loss,
                    'epoch': epoch,
                    'learning rate': self.optimizer.param_groups[0]['lr'],
                },
                step=self.global_step,
            )

        elif mode == 'valid':
            self.accelerator.log(
                {
                    'valid loss': self.valid_loss,
                    'recon loss': self.recon_loss,
                    'epoch': epoch,
                },
                step=self.global_step,
            )


    def save_checkpoint(self, epoch, multi_gpu_enabled=False):
        checkpoint_dict = {
            "model": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "ema_model": self.accelerator.get_state_dict(self.ema_model) if multi_gpu_enabled else None
        }
        save_dir = f"{self.save_name}/checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/epoch_{epoch}.pt"
        torch.save(checkpoint_dict, save_path)
        print(f'[Succeed] Model Checkpoint saved to {save_path}!')


    def load_checkpoint(self, path):
        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.start_epoch = checkpoint_dict["epoch"]
        if "ema_model" in checkpoint_dict and checkpoint_dict["ema_model"] is not None:
            self.ema_model.load_state_dict(checkpoint_dict["ema_model"])
            self.ema_checkpoint_load = True
        else:
            self.ema_checkpoint_load = False
        print(f'[Succeed] Model Checkpoint loaded from {path}!')


    def _calculate_reconstruction_loss(self, x, label):
        self.rec_count += 1
        if self.rec_count >= 10000:
            with torch.no_grad():
                T = torch.full((x.shape[0],), int(self.model.timestep) - 1, device=self.accelerator.device, dtype=torch.long)
                x_T = self.model.q_sample(x_start=x, t=T).float()
                x_T_to_0 = self.model.reverse_process_guided(x_T=x_T, classes=label)
                x_0 = x_T_to_0[-1]
            self.rec_count = 0
            self.recon_loss = F.mse_loss(x, x_0, reduction='mean').item()

    def _save_fasta(self, sequences, folder_name:str= 'snapshots', epoch:int =None, trial_name:str=None):
            print('Saving fasta file...')
            write_to_fasta(sequences, folder_name=os.path.join(self.save_name, folder_name), epoch=epoch, trial_name=trial_name)