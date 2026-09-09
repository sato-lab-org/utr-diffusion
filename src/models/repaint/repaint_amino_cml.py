import math
import numbers
import random

import torch
from einops import rearrange
from tqdm.auto import tqdm

from .amino_codon_table import AA_TO_CODON_USAGE_HUMAN_RNA, get_codons_for_amino, rna_to_dna
from .scheduler import get_schedule_jump, resolve_schedule
from .utils import (
    aminos_to_amino_images,
    base2vec,
    build_gt_from_image_and_pos,
    build_target_adaptiveness_probability_table,
    get_guidance_mask_parameter,
    inf_base2vec,
    normalize_target_labels,
    validate_amino_constraints,
    validate_target_adaptiveness,
)


SUPPORTED_STRATEGIES = frozenset({
    'euclidean',
    'wasserstein',
    'usage_weighted_distance',
    'specific_adaptiveness',
    'init_random',
    'init_random_fixed',
    'init_usage',
})


class RePaint_Amino_Continuous_Multi_Labels:
    """RePaint sampler with amino-acid and continuous-label constraints."""

    def __init__(
            self,
            diffusion,
            tgt_labels: list,
            sample_bs: int = 100,
            seq_len: int = 50,
            cond_weight: float = 2.0,
            strategy='wasserstein',
            stop_point=1.0,
            skip_frames: int = 1,
            return_all=True,
            gamma_for_usage_frequency: float = 0.04,
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
        if seq_len < 3:
            raise ValueError("seq_len must be at least 3")
        if isinstance(skip_frames, bool) or not isinstance(skip_frames, numbers.Integral):
            raise TypeError("skip_frames must be an integer")
        if skip_frames < 1:
            raise ValueError("skip_frames must be positive")
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")

        cond_weight = float(cond_weight)
        stop_point = float(stop_point)
        gamma_for_usage_frequency = float(gamma_for_usage_frequency)
        if not math.isfinite(cond_weight):
            raise ValueError("cond_weight must be finite")
        if not math.isfinite(stop_point) or not 0.0 <= stop_point <= 1.0:
            raise ValueError("stop_point must be in [0, 1]")
        if not math.isfinite(gamma_for_usage_frequency) or gamma_for_usage_frequency < 0.0:
            raise ValueError("gamma_for_usage_frequency must be finite and non-negative")

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
        self.strategy = strategy
        self.stop_point = stop_point
        self.skip_frames = int(skip_frames)
        self.gamma_for_usage_frequency = gamma_for_usage_frequency

        self.tgt_aminos = None
        self.pos_list = None
        self.target_adaptiveness = None
        self.effective_adaptiveness_by_amino = None
        self.adaptiveness_probability_table = None
        self.CAI_usage_table = None  # compatibility alias; values target alpha, not CAI
        self.gt_image = None
        self.mask = None
        self.batch_labels = None
        self.all_amino_images = None

    def setup(self, amino_list: list[str], pos_list: list[int], adaptiveness: float = None):
        """Configure amino constraints before calling ``p_resample()``.

        For ``specific_adaptiveness``, ``adaptiveness`` is the expected codon
        relative-adaptiveness alpha and must be in ``(0, 1]``.  It is not a
        sequence-level CAI target.
        """

        normalized_aminos, normalized_positions = validate_amino_constraints(
            amino_list,
            pos_list,
            total_length=self.seq_len,
        )

        if self.strategy == 'specific_adaptiveness' and adaptiveness is None:
            raise ValueError(
                "adaptiveness is required when strategy='specific_adaptiveness'"
            )
        target_adaptiveness = (
            validate_target_adaptiveness(adaptiveness)
            if adaptiveness is not None
            else None
        )

        gt_image, mask = self._make_gt_mask_from_aminos(
            normalized_aminos,
            normalized_positions,
        )
        batch_labels = self._build_batch_labels()
        all_amino_images = aminos_to_amino_images(
            normalized_aminos,
            with_padding=True,
        ).to(self.device)

        if self.strategy == 'init_usage':
            codon_usage_images = get_amino_images_by_codon_usage(
                tgt_aminos=normalized_aminos,
                sample_size=self.shape[0],
                device=self.device,
            )
            gt_image = build_gt_from_image_and_pos(
                codon_images=codon_usage_images,
                pos_list=normalized_positions,
                total_length=self.seq_len,
                device=self.device,
            )

        if self.strategy == 'specific_adaptiveness':
            (
                adaptiveness_probability_table,
                effective_adaptiveness_by_amino,
            ) = build_target_adaptiveness_probability_table(
                target_adaptiveness=target_adaptiveness,
                return_effective_targets=True,
            )
        else:
            adaptiveness_probability_table = None
            effective_adaptiveness_by_amino = None

        # Commit the new setup only after every validation and construction
        # step has succeeded, so a failed second setup cannot mix old/new state.
        self.tgt_aminos = normalized_aminos
        self.pos_list = normalized_positions
        self.target_adaptiveness = target_adaptiveness
        self.effective_adaptiveness_by_amino = effective_adaptiveness_by_amino
        self.gt_image = gt_image
        self.mask = mask
        self.batch_labels = batch_labels
        self.all_amino_images = all_amino_images
        self.adaptiveness_probability_table = adaptiveness_probability_table
        self.CAI_usage_table = adaptiveness_probability_table

        return self

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

    def _make_gt_mask_from_aminos(self, amino_list, pos_list):
        batch_size = self.shape[0]
        sequences = [['N'] * self.seq_len for _ in range(batch_size)]
        mask = torch.zeros(self.shape, dtype=torch.float32)

        for amino, pos in zip(amino_list, pos_list):
            selected_codons = random.choices(
                get_codons_for_amino(amino),
                k=batch_size,
            )
            for batch_index, codon in enumerate(selected_codons):
                sequences[batch_index][pos:pos + 3] = list(codon)
            mask[:, :, :, pos:pos + 3] = 1.0

        images = [
            torch.tensor(
                [base2vec[base] for base in sequence],
                dtype=torch.float32,
            ).T
            for sequence in sequences
        ]
        gt_image = torch.stack(images, dim=0).unsqueeze(1).to(self.device)
        return gt_image, mask.to(self.device)

    def _prepare_legacy_tensor(self, value, name):
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

    def _sampling_inputs(self, gt, mask, tgt_aminos, pos_list):
        legacy_values = (gt, mask, tgt_aminos, pos_list)
        if all(value is None for value in legacy_values):
            if any(
                value is None
                for value in (
                    self.gt_image,
                    self.mask,
                    self.batch_labels,
                    self.all_amino_images,
                    self.pos_list,
                )
            ):
                raise RuntimeError("call setup(...) before p_resample()")
            return (
                self.gt_image,
                self.mask,
                self.batch_labels,
                self.all_amino_images,
                self.pos_list,
                self.tgt_aminos,
            )

        if any(value is None for value in legacy_values):
            raise TypeError(
                "legacy p_resample requires gt, mask, tgt_aminos, and pos_list together"
            )

        amino_list, positions = validate_amino_constraints(
            tgt_aminos,
            pos_list,
            total_length=self.seq_len,
        )
        prepared_gt = self._prepare_legacy_tensor(gt, 'gt')
        prepared_mask = self._prepare_legacy_tensor(mask, 'mask')
        if (prepared_mask < 0).any() or (prepared_mask > 1).any():
            raise ValueError("mask values must be in [0, 1]")
        expected_mask = torch.zeros_like(prepared_mask)
        for position in positions:
            expected_mask[:, :, :, position:position + 3] = 1.0
        if not torch.equal(prepared_mask, expected_mask):
            raise ValueError(
                "legacy mask must constrain exactly the three bases at every pos_list entry"
            )

        amino_images = aminos_to_amino_images(
            amino_list,
            with_padding=True,
        ).to(self.device)
        return (
            prepared_gt,
            prepared_mask,
            self._build_batch_labels(),
            amino_images,
            positions,
            amino_list,
        )

    def p_resample(
            self,
            gt=None,
            mask=None,
            tgt_aminos=None,
            pos_list=None,
            schedule=None,
    ):
        """Run RePaint after ``setup``, or through the legacy four-argument API."""

        (
            gt,
            mask,
            labels,
            amino_images,
            positions,
            amino_list,
        ) = self._sampling_inputs(gt, mask, tgt_aminos, pos_list)

        result = {'samples': [], 'forward_steps': [], 'backward_steps': []}
        iterator = self.p_resample_loop(
            gt,
            mask,
            labels,
            amino_images,
            positions,
            amino_list,
            schedule=schedule,
        )
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

    def p_resample_loop(
            self,
            gt,
            mask,
            labels,
            tgt_amino_images,
            amino_pos,
            tgt_aminos=None,
            schedule=None,
    ):
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
        amino_list = self.tgt_aminos if tgt_aminos is None else tgt_aminos

        for t_last, t_cur in time_pairs:
            if t_cur < t_last:
                backward_step_count += 1
                early_stop = (
                    backward_step_count + forward_step_count
                    > self.stop_point * len(times[:-1])
                )
                if not early_stop:
                    gt = self.apply_codon_flexibility(
                        gt,
                        image,
                        mask,
                        tgt_amino_images,
                        amino_pos,
                        amino_list,
                    )

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
                with torch.no_grad():
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

    def apply_codon_flexibility(
            self,
            gt,
            image,
            mask,
            all_amino_images,
            amino_pos,
            tgt_aminos=None,
    ) -> torch.Tensor:
        if self.strategy in {'init_random', 'init_usage', 'init_random_fixed'}:
            return gt

        amino_list = self.tgt_aminos if tgt_aminos is None else tgt_aminos
        mask_1d = mask[0, 0, 0, :]
        constrained_indices = mask_1d.nonzero(as_tuple=False).flatten()
        fix_region = image[:, :, :, constrained_indices]
        new_codon_images = choose_codon_by_strategy(
            query=fix_region,
            candidates=all_amino_images,
            tgt_aminos=amino_list,
            strategy=self.strategy,
            gamma=self.gamma_for_usage_frequency,
            adaptiveness_probability_table=self.adaptiveness_probability_table,
        )
        return build_gt_from_image_and_pos(
            codon_images=new_codon_images,
            pos_list=amino_pos,
            total_length=self.seq_len,
            device=self.device,
        )


def choose_codon_by_strategy(
        query: torch.Tensor,
        candidates: torch.Tensor,
        strategy: str = 'wasserstein',
        tgt_aminos=None,
        gamma: float = 0.04,
        adaptiveness_probability_table=None,
        CAI_usage_table=None,
) -> torch.Tensor:
    """Choose one synonymous codon for each constrained amino acid.

    ``CAI_usage_table`` is accepted as a compatibility keyword.  Its contents
    are probabilities targeting relative adaptiveness alpha, not sequence CAI.
    """

    if adaptiveness_probability_table is None:
        adaptiveness_probability_table = CAI_usage_table

    batch_size = query.shape[0]
    query = rearrange(query, 'b c h (n p) -> b (c n) h p', p=3)
    query = query.unsqueeze(2).expand(-1, -1, 6, -1, -1)
    candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    if strategy == 'euclidean':
        distance = euclidean_distance(query, candidates)
    elif strategy == 'wasserstein':
        distance = wasserstein_1d_distance(query, candidates)
    elif strategy == 'usage_weighted_distance':
        distance = usage_weighted_distance(
            query,
            candidates,
            tgt_aminos,
            gamma=gamma,
        )
    elif strategy == 'specific_adaptiveness':
        distance = target_adaptiveness_weighted_distance(
            dist=euclidean_distance(query, candidates),
            tgt_aminos=tgt_aminos,
            adaptiveness_probability_table=adaptiveness_probability_table,
            gamma=gamma,
        )
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    codon_index = distance.argmin(-1)
    gather_index = codon_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3)
    return torch.gather(candidates, 2, gather_index.unsqueeze(2)).squeeze(2)


def wasserstein_1d_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    query_cdf = torch.cumsum(query, dim=3)
    candidate_cdf = torch.cumsum(candidates, dim=3)
    return (query_cdf - candidate_cdf).abs().sum(dim=[3, 4])


def euclidean_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Return the historical summed per-element distance.

    Existing experiments named this metric ``euclidean`` even though summing
    ``sqrt(diff**2)`` is an L1 distance.  Preserve that calibrated behavior for
    checkpoint/``gamma`` compatibility.
    """

    diff = query - candidates
    return torch.sqrt(diff ** 2).sum(dim=[3, 4])


def usage_weighted_distance(
        query: torch.Tensor,
        candidates: torch.Tensor,
        tgt_aminos,
        gamma: float = 0.04,
) -> torch.Tensor:
    if tgt_aminos is None:
        raise ValueError("tgt_aminos is required for usage_weighted_distance")
    if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    distance = euclidean_distance(query, candidates)
    probability_rows = []
    valid_rows = []
    for amino in tgt_aminos:
        probabilities = list(AA_TO_CODON_USAGE_HUMAN_RNA[amino].values())
        probability_rows.append(probabilities + [0.0] * (6 - len(probabilities)))
        valid_rows.append([True] * len(probabilities) + [False] * (6 - len(probabilities)))

    codon_probability = torch.tensor(
        probability_rows,
        dtype=distance.dtype,
        device=distance.device,
    )
    codon_probability = codon_probability / codon_probability.sum(dim=1, keepdim=True)
    valid_mask = torch.tensor(valid_rows, dtype=torch.bool, device=distance.device)
    score = distance / codon_probability.unsqueeze(0).clamp_min(1e-12).pow(float(gamma))
    return score.masked_fill(~valid_mask.unsqueeze(0), float('inf'))


def target_adaptiveness_weighted_distance(
        dist: torch.Tensor,
        tgt_aminos,
        adaptiveness_probability_table,
        gamma: float = 0.04,
):
    if tgt_aminos is None:
        raise ValueError("tgt_aminos is required for specific_adaptiveness")
    if adaptiveness_probability_table is None:
        raise ValueError(
            "an adaptiveness probability table is required for specific_adaptiveness"
        )
    if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    probability = torch.stack([
        adaptiveness_probability_table[amino].to(
            device=dist.device,
            dtype=dist.dtype,
        )
        for amino in tgt_aminos
    ])
    valid_mask = probability > 0
    weighted_distance = dist / probability.unsqueeze(0).clamp_min(1e-12).pow(float(gamma))
    return weighted_distance.masked_fill(~valid_mask.unsqueeze(0), float('inf'))


def target_cai_weighted_distance(
        dist: torch.Tensor,
        tgt_aminos,
        cai_usage_table,
        gamma: float = 0.04,
):
    """Compatibility alias; the table targets alpha, not sequence CAI."""

    return target_adaptiveness_weighted_distance(
        dist=dist,
        tgt_aminos=tgt_aminos,
        adaptiveness_probability_table=cai_usage_table,
        gamma=gamma,
    )


def get_amino_images_by_codon_usage(
        tgt_aminos,
        sample_size: int,
        device,
        base_encoding=inf_base2vec,
):
    all_amino_images = []
    for amino in tgt_aminos:
        usage = AA_TO_CODON_USAGE_HUMAN_RNA[amino]
        codons = list(usage)
        probabilities = torch.tensor(
            list(usage.values()),
            dtype=torch.float32,
            device=device,
        )
        probabilities = probabilities / probabilities.sum()
        sampled_indices = torch.multinomial(
            probabilities,
            num_samples=sample_size,
            replacement=True,
        )
        sampled_codons = [codons[index] for index in sampled_indices.cpu().tolist()]
        amino_images = [
            torch.tensor(
                [base_encoding[rna_to_dna(base)] for base in codon],
                dtype=torch.float32,
                device=device,
            ).T
            for codon in sampled_codons
        ]
        all_amino_images.append(torch.stack(amino_images))

    return torch.stack(all_amino_images, dim=1)
