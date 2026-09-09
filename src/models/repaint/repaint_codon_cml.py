import math
import numbers

import torch
from tqdm.auto import tqdm

from .scheduler import get_schedule_jump, resolve_schedule
from .utils import get_guidance_mask_parameter, normalize_target_labels


class RePaint_Codon_Continuous_Multi_Labels:
    """RePaint sampler for fixed nucleotide/codon constraints."""

    def __init__(
            self,
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
        self._guidance_mask_parameter = get_guidance_mask_parameter(diffusion)

        if isinstance(sample_bs, bool) or not isinstance(sample_bs, numbers.Integral):
            raise TypeError("sample_bs must be an integer")
        if sample_bs < 1:
            raise ValueError("sample_bs must be positive")
        if isinstance(seq_len, bool) or not isinstance(seq_len, numbers.Integral):
            raise TypeError("seq_len must be an integer")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        if isinstance(skip_frames, bool) or not isinstance(skip_frames, numbers.Integral):
            raise TypeError("skip_frames must be an integer")
        if skip_frames < 1:
            raise ValueError("skip_frames must be positive")

        cond_weight = float(cond_weight)
        if not math.isfinite(cond_weight):
            raise ValueError("cond_weight must be finite")

        self.sample_bs = int(sample_bs)
        self.seq_len = int(seq_len)
        self.cond_weight = cond_weight
        self.tgt_labels = normalize_target_labels(tgt_labels)
        self.num_joint_class = len(self.tgt_labels)
        self.num_tgt_labels = self.num_joint_class  # historical attribute name
        first_target = self.tgt_labels[0]
        self.label_dim = 1 if isinstance(first_target, numbers.Real) else len(first_target)
        self.return_all = bool(return_all)
        self.shape = [self.sample_bs * self.num_joint_class, 1, 4, self.seq_len]
        self.skip_frames = int(skip_frames)

    def _build_batch_labels(self):
        labels = [
            joint_label
            for joint_label in self.tgt_labels
            for _ in range(self.sample_bs)
        ]
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        return labels

    def _prepare_constraint_tensor(self, value, name):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (1, 4, self.seq_len):
            raise ValueError(
                f"{name} must have shape [1, 4, {self.seq_len}] or "
                f"[B, 1, 4, {self.seq_len}]"
            )
        if tensor.shape[0] not in (1, self.shape[0]):
            raise ValueError(
                f"{name} batch dimension must be 1 or {self.shape[0]}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor.expand(self.shape[0], -1, -1, -1)

    @torch.no_grad()
    def p_resample(self, gt, mask, schedule=None):
        gt = self._prepare_constraint_tensor(gt, 'gt')
        mask = self._prepare_constraint_tensor(mask, 'mask')
        if (mask < 0).any() or (mask > 1).any():
            raise ValueError("mask values must be in [0, 1]")
        labels = self._build_batch_labels()

        result = {'samples': [], 'forward_steps': [], 'backward_steps': []}
        iterator = self.p_resample_loop(gt, mask, labels, schedule=schedule)
        if self.return_all:
            for index, (sample, forward_step, backward_step) in enumerate(iterator):
                if index % self.skip_frames != 0:
                    continue
                result['samples'].append(sample.detach().cpu().to(torch.float16).numpy())
                result['forward_steps'].append(forward_step)
                result['backward_steps'].append(backward_step)
            return result

        final_sample = None
        for sample, _, _ in iterator:
            final_sample = sample
        if final_sample is None:
            raise RuntimeError("RePaint schedule produced no sampling steps")
        return final_sample.detach().cpu().to(torch.float16).numpy()

    def p_resample_loop(self, gt, mask, labels, schedule=None):
        backward_step_count = 0
        forward_step_count = 0
        schedule = resolve_schedule(
            schedule,
            max_timestep=getattr(self.model, 'timestep', None),
        )

        times = get_schedule_jump(**schedule)
        time_pairs = tqdm(list(zip(times[:-1], times[1:])), leave=False)

        n_sample = self.shape[0]
        image = torch.randn(self.shape, device=self.device)
        guidance_mask = torch.cat(
            [torch.ones_like(labels), torch.zeros_like(labels)],
            dim=0,
        ).to(self.device)
        guided_labels = labels.repeat(2, 1)

        for t_last, t_cur in time_pairs:
            if t_cur < t_last:
                backward_step_count += 1
                gt_noised = self.noise_steps(gt, t_last)
                mixed_image = gt_noised * mask + image * (1 - mask)
                timesteps = torch.full(
                    (n_sample,),
                    t_last,
                    dtype=torch.long,
                    device=self.device,
                )
                sample_kwargs = {
                    'x': mixed_image,
                    't': timesteps,
                    'classes': guided_labels,
                    'cond_weight': self.cond_weight,
                    't_index': t_last,
                    self._guidance_mask_parameter: guidance_mask,
                }
                image = self.model.p_sample_guided(**sample_kwargs)
            else:
                forward_step_count += 1
                # This transition is x_(t_last) -> x_(t_cur), so beta[t_cur]
                # is the one-step forward-kernel coefficient.
                image = self.noise_step(image, t_cur)

            yield image, forward_step_count, backward_step_count

    def noise_step(self, x_t_m1, t):
        timesteps = torch.full(
            (x_t_m1.shape[0],),
            t,
            dtype=torch.long,
            device=self.device,
        )
        return self.model.q_sample_single_step(x_t_m1, timesteps)

    def noise_steps(self, x_0, t):
        timesteps = torch.full(
            (x_0.shape[0],),
            t,
            dtype=torch.long,
            device=self.device,
        )
        return self.model.q_sample(x_0, timesteps)
