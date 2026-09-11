import torch
from einops import rearrange
from collections import defaultdict
from scipy.stats import wasserstein_distance
from torch.backends.opt_einsum import strategy

from .scheduler import get_schedule_jump, schedule_jump_params, schedule_no_jump_params
from tqdm.auto import tqdm
from .utils import get_amino_images_for_alter_codons, build_gt_from_image_and_pos


# this repaint sampler works for continuous multi-label generation
class RePaint_Codon_Continuous_Multi_Labels:
    def __init__(self,
                 diffusion,
                 tgt_labels: list,
                 sample_bs: int = 100,
                 seq_len: int = 50,
                 cond_weight: float = 2.0,
                 return_all=True,
                 skip_frames: int = 1,
                 ):
        self.model = diffusion
        self.device = diffusion.device
        self.sample_bs = sample_bs
        self.seq_len = seq_len
        self.cond_weight = cond_weight
        self.tgt_labels = tgt_labels
        self.num_joint_class = len(tgt_labels)
        self.label_dim = 1 if isinstance(self.tgt_labels[0], (float, int)) else len(self.tgt_labels[0])
        self.return_all = return_all
        self.shape = [sample_bs * self.num_joint_class] + [1, 4, seq_len]
        self.skip_frames = skip_frames

    @torch.no_grad()
    def p_resample(self, gt, mask, schedule:dict = schedule_jump_params):
        labels = [[mrl, mfe] for mrl, mfe in self.tgt_labels for _ in range(self.sample_bs)]
        labels = torch.tensor(labels, dtype=torch.float, device=self.device)
        gt = gt.expand(self.sample_bs * self.num_joint_class, -1, -1, -1).to(self.device)
        mask = mask.expand(self.sample_bs * self.num_joint_class, -1, -1, -1).to(self.device)

        result = {'samples': [], 'forward_steps': [], 'backward_steps': []}

        if self.return_all:
            for idx, (sample, forward_step, backward_step) in enumerate(self.p_resample_loop(gt, mask, labels, schedule)):
                if idx % self.skip_frames != 0:
                    continue
                result['samples'].append(sample.detach().cpu().to(torch.float16).numpy())
                result['forward_steps'].append(forward_step)
                result['backward_steps'].append(backward_step)
        else:
            for sample, forward_step, backward_step in self.p_resample_loop(gt, mask, labels, schedule):
                result = sample.detach().cpu().to(torch.float16).numpy()
        return result

    def p_resample_loop(self, gt, mask, labels, schedule:dict = schedule_jump_params):
        backward_step_count = 0
        forward_step_count = 0

        times = get_schedule_jump(**schedule)
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = tqdm(time_pairs)

        n_sample = self.shape[0]
        image = torch.randn(self.shape, device=self.device)

        context_mask = torch.concat([torch.ones_like(labels), torch.zeros_like(labels)], dim=0).to(self.device)
        # double the batch and make 0 index unconditional
        labels = labels.repeat(2, 1)


        for t_last, t_cur in time_pairs:
            if t_cur < t_last:  # reverse
                backward_step_count += 1
                gt_noised = self.noise_steps(x_0=gt, t=t_last)
                mixed_image = gt_noised * mask + image * (1 - mask)

                timesteps = torch.full((n_sample,), t_last, dtype=torch.long, device=self.device)
                image = self.model.p_sample_guided(x=mixed_image, t=timesteps, classes=labels,
                                                   cond_weight=self.cond_weight,
                                                   context_mask=context_mask, t_index=t_last)
            else:
                forward_step_count += 1
                image = self.noise_step(x_t_m1=image, t=t_last)

            yield image, forward_step_count, backward_step_count

    def noise_step(self, x_t_m1, t):
        B = x_t_m1.shape[0]
        t = torch.full((B,), t, dtype=torch.long, device=self.device)
        x_t = self.model.q_sample_single_step(x_t_m1, t)
        return x_t

    def noise_steps(self, x_0, t):
        B = x_0.shape[0]
        t = torch.full((B,), t, dtype=torch.long, device=self.device)
        x_t = self.model.q_sample(x_0, t)
        return x_t