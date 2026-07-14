from typing import Any
import torch
import os
from accelerate import Accelerator
from tqdm import tqdm
from functools import partial
from src.utils.sample_util import inference
from src.utils.utils import get_warmup_flatten_cosine_schedule as lr_schedule
from .train_loop_basic import BasicTrainLoop

class TrainLoop_single_gpu(BasicTrainLoop):
    def __init__(
        self,
        data: dict[str, Any],
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
        label_names = None,
        tgt_values = None, # if None discrete else continueous
        seq_len: int= 50
    ):
        super().__init__(model=model, accelerator=accelerator, start_epoch=start_epoch, end_epoch=end_epoch, log_step=log_step,
                         valid_epoch=valid_epoch, sample_epoch=sample_epoch, save_epoch=save_epoch, save_name=save_name,
                         batch_size=batch_size, num_workers=num_workers, learning_rate=learning_rate,
                         num_classes= data['Classes'] if 'Classes' in data else 0)

        # some setting params
        self.label_names = label_names
        self.tgt_values = tgt_values
        self.seq_len = seq_len

        # Dataloader and Learning schedule
        self.train_dl, self.valid_dl = self._prepare_data_loader(data)
        self.schedule = lr_schedule(
            optimizer=self.optimizer,
            num_training_steps=len(self.train_dl) * self.end_epoch if self.train_dl is not None else 1,
            warmup_rate=0.05,
            flatten_rate=0.7,
        )


    def train_loop(self):
        # Prepare for training
        self.model, self.optimizer, self.train_dl, self.valid_dl, self.schedule = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dl, self.valid_dl, self.schedule)

        self.log_update(mode='init')
        for epoch in tqdm(range(self.start_epoch, self.end_epoch + 1)):
            # training
            self.train(epoch=epoch)

            #validation
            if epoch % self.valid_epoch == 0:
                self.validation(epoch=epoch)

            # Sampling
            if epoch % self.sample_epoch == 0:
                self.sample(epoch=epoch)

            # Saving checkpoint
            if epoch % self.save_epoch == 0:
                self.save_checkpoint(epoch=epoch)

        self.accelerator.end_training()
        return self.valid_loss ## validation modal is not finished yet


    def train(self, epoch):
        self.model.train()  # shift to train mode
        for step, batch in enumerate(self.train_dl):
            x, y = batch
            with self.accelerator.autocast():  # Mixed precision on 混合精度オンにする
                loss = self.model(x, y)

            # update optimizer, scheduler and global step
            self.optimizer.zero_grad(set_to_none=True)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.schedule.step()
            self.global_step += 1

            # logging
            if self.global_step % self.log_step == 0:
                self.train_loss = loss.item()
                self.log_update(mode='train', epoch=epoch)


    def validation(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            with self.accelerator.autocast():
                for batch in self.valid_dl:
                    x, y = batch
                    loss = self.model(x, y)
                    total_loss += loss.item()

        self.valid_loss = total_loss / len(self.valid_dl)
        if self.valid_loss <= self.best_valid_loss:
            self.best_valid_loss = self.valid_loss

        # logging
        self.log_update(mode='valid', epoch=epoch)


    def sample(self, epoch):
        self.model.eval()
        sample_fn = partial(
            inference,
            diffusion_model=self.model,
            seq_len = self.seq_len,
            class_num=self.num_classes,
            cond_weight=self.model.cond_weight,
            label_names=self.label_names,
            target_values=self.tgt_values,
            device= self.accelerator.device
        )
        with torch.no_grad():
            with self.accelerator.autocast():
                print("\nGenerating synthetic sequences...")
                if epoch == self.end_epoch and self.is_save_process:
                    seqs, all_images = sample_fn(output_all_steps=True)
                    torch.save({k: v.cpu().numpy() for k, v in all_images.items()},
                               os.path.join(self.save_name, 'snapshots',"all_images_denoising_process.pt"))
                    print("all_images_denoising_process.pt saved!")
                else:
                    seqs = sample_fn()

            self._save_fasta(sequences=seqs, epoch=epoch)


    def sample_offline(self, sample_bs:int= 1000, trial_name=None):
        device = self.accelerator.device
        model_for_sampling = self.ema_model if self.ema_checkpoint_load else self.model
        print("Sampling with:", "EMA" if self.ema_checkpoint_load else "RAW")
        model_for_sampling.to(device)
        model_for_sampling.eval()
        sample_fn = partial(
            inference,
            diffusion_model=model_for_sampling,
            sample_bs = sample_bs,
            seq_len = self.seq_len,
            class_num=self.num_classes,
            cond_weight=self.model.cond_weight,
            label_names = self.label_names,
            target_values=self.tgt_values,
            device=device,
        )
        with torch.no_grad():
            print("accelerator.mixed_precision =", self.accelerator.mixed_precision)
            with self.accelerator.autocast():
                print("torch.is_autocast_enabled() =", torch.is_autocast_enabled())
                if torch.cuda.is_available():
                    print("torch.get_autocast_gpu_dtype() =", torch.get_autocast_gpu_dtype())
                print("\nGenerating synthetic sequences...")
                seqs = sample_fn()
                self._save_fasta(sequences=seqs, folder_name='samples', trial_name=trial_name)


    
    def load_checkpoint_then_do_sample(self, checkpoint_path, trial_name=None, sample_bs:int=1000):
        self.load_checkpoint(checkpoint_path)
        self.sample_offline(trial_name=trial_name,  sample_bs=sample_bs)
        



