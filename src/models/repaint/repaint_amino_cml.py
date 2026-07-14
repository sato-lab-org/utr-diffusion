import torch
import random
from einops import rearrange
from collections import defaultdict

from scipy.constants import degree_Fahrenheit
from scipy.stats import wasserstein_distance
from torch.backends.opt_einsum import strategy
from .scheduler import get_schedule_jump, schedule_jump_params, schedule_no_jump_params
from tqdm.auto import tqdm
from .utils import aminos_to_amino_images, build_gt_from_image_and_pos, inf_base2vec, base2vec, build_codon_usage_table_for_specific_CAI
from .amino_codon_table import AA_TO_CODON_USAGE_HUMAN_RNA, get_codons_for_amino, rna_to_dna

# this repaint sampler works for continuous multi-label generation
class RePaint_Amino_Continuous_Multi_Labels:
    def __init__(self,
                 diffusion,
                 tgt_labels: list,
                 sample_bs: int = 100,
                 seq_len: int = 50,
                 cond_weight: float = 2.0,
                 strategy='wasserstein',
                 stop_point=1.0,
                 skip_frames: int = 1,
                 gamma_for_usage_frequency: float = 0.04,
                 return_all=True):
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
        self.strategy = strategy
        self.stop_point = stop_point
        self.skip_frames = skip_frames
        self.tgt_aminos = None
        self.pos_list = None
        self.gamma_for_usage_frequency = gamma_for_usage_frequency
        self.CAI_usage_table = None # codon usage table to calculate weights for usage_weighted_distance, when user want to generate seq with specific CAI
        self.gt_image = None # Do we need to establish a space for gt_image like torch.zero(self.shape)
        self.mask = None # torch.zero(self.shape)
        self.batch_labels = None
        self.all_amino_images = None

    def setup(self, amino_list: list[str], pos_list: list[int], adaptiveness:int = None):
        self.tgt_aminos = amino_list
        self.pos_list = pos_list
        self._build_gt_mask_from_aminos()

        labels = [joint_label for joint_label in self.tgt_labels for _ in range(self.sample_bs)]
        self.batch_labels = torch.tensor(labels, dtype=torch.float, device=self.device)
        self.all_amino_images = aminos_to_amino_images(self.tgt_aminos, with_padding=True).to(self.device)

        if self.strategy == 'init_usage': #initilization base on codon usage frequency
            codon_usage_images = get_amino_images_by_codon_usage(
                tgt_aminos=self.tgt_aminos,
                sample_size=self.sample_bs * self.num_joint_class,
                device=self.device,
            )
            self.gt_image = build_gt_from_image_and_pos(
                codon_images=codon_usage_images,
                pos_list=self.pos_list,
                device=self.device,
            )
        if self.strategy == 'specific_adaptiveness' and adaptiveness is not None:
            self.CAI_usage_table = build_codon_usage_table_for_specific_CAI(
                amino_to_codons=AA_TO_CODON_USAGE_HUMAN_RNA,
                codon_usage_table=AA_TO_CODON_USAGE_HUMAN_RNA,
                target_cai=adaptiveness,
            )


    def _build_gt_mask_from_aminos(self):
        batch_size = self.sample_bs * self.num_joint_class
        seqs = [['N'] * self.seq_len for _ in range(batch_size)]
        mask = torch.zeros(self.shape, dtype=torch.float)

        prev_pos, amino_length = -3, 3
        for amino, pos in zip(self.tgt_aminos, self.pos_list):
            if prev_pos + amino_length > pos:
                raise ValueError("aminos overlap.")

            alter_codons = get_codons_for_amino(amino)

            # batch-wise random codon initialization
            selected_codons = random.choices(alter_codons, k=batch_size)
            for b, codon in enumerate(selected_codons):
                seqs[b][pos:pos + amino_length] = list(codon)

            mask[:, :, :, pos:pos + amino_length] = 1.0
            prev_pos = pos

        images = []
        for seq in seqs:
            image = torch.tensor([base2vec.get(base) for base in seq], dtype=torch.float).T
            images.append(image)

        gt_image = torch.stack(images, dim=0).unsqueeze(1)

        self.gt_image=gt_image.to(self.device)
        self.mask = mask.to(self.device)


    def p_resample(self):
        result = {'samples': [], 'forward_steps': [], 'backward_steps': []}
        if self.return_all:
            for idx, (sample, forward_step, backward_step) in enumerate(
                    self.p_resample_loop(self.gt_image, self.mask, self.batch_labels, self.all_amino_images, self.pos_list)):
                if idx % self.skip_frames != 0:
                    continue
                result['samples'].append(sample.detach().cpu().to(torch.float16).numpy())
                result['forward_steps'].append(forward_step)
                result['backward_steps'].append(backward_step)

        else:
            for sample, forward_step, backward_step in self.p_resample_loop(self.gt_image, self.mask, self.batch_labels, self.all_amino_images, self.pos_list):
                result = sample.detach().cpu().to(torch.float16).numpy()

        return result

    def p_resample_loop(self, gt, mask, labels, tgt_amino_images, amino_pos):
        backward_step_count, forward_step_count = 0, 0

        times = get_schedule_jump(**schedule_jump_params)
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = tqdm(time_pairs)

        n_sample = self.shape[0]
        image = torch.randn(self.shape, device=self.device)

        context_mask = torch.concat([torch.ones_like(labels), torch.zeros_like(labels)], dim=0).to(self.device)
        # double the batch and make 0 index unconditional
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)  # (B,1)

        labels = labels.repeat(2, 1)  # (2B, D)

        for t_last, t_cur in time_pairs:
            if t_cur < t_last:  # reverse
                backward_step_count += 1
                # early stopping
                early_stop = backward_step_count + forward_step_count > self.stop_point * len(time_pairs)
                if not early_stop:
                    gt = self.apply_codon_flexibility(gt, image, mask, tgt_amino_images, amino_pos)

                gt_noised = self.noise_steps(x_0=gt, t=t_last)
                mixed_image = gt_noised * mask + image * (1 - mask)
                timesteps = torch.full((n_sample,), t_last, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    image = self.model.p_sample_guided(
                        x=mixed_image,
                        t=timesteps,
                        classes=labels,
                        cond_weight=self.cond_weight,
                        context_mask=context_mask,
                        t_index=t_last
                    )
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

    def apply_codon_flexibility(self, gt, image, mask, all_amino_images, amino_pos) -> torch.Tensor:

        if self.strategy in ['init_random', 'init_usage', 'init_random_fixed']:
            return gt
        # mask, image [B, 1, C, L]
        mask_1d = mask[0, 0, 0, :]
        index = mask_1d.nonzero(as_tuple=False).flatten()
        fix_region = image[:, :, :, index]  # [B, 1, 4, 50] -> [B, 1, 4, 3 * N]

        new_codon_images = choose_codon_by_strategy(fix_region, all_amino_images, self.tgt_aminos, self.strategy, self.gamma_for_usage_frequency, self.CAI_usage_table)  # [B, N, 6]
        gt_image = build_gt_from_image_and_pos(codon_images=new_codon_images, pos_list=amino_pos, device=self.device)

        return gt_image


def choose_codon_by_strategy(
        query: torch.Tensor,
        candidates: torch.Tensor,
        tgt_aminos: list[str],
        strategy: str = 'wasserstein',
        gamma: float = None,
        CAI_usage_table:dict = None,
) -> torch.Tensor:
    """
    query:     [B, 1, 4, 3 * N] — N: number of specified amino
    candidates:[N, 6, 4, 3] — all amino images, one amino mapping 6 codons at max
    return:    [B, N, 6] —
    """
    batch_size = query.shape[0]
    query = rearrange(query, 'b c h (n p) -> b (c n) h p', p=3)  # [B, 1, 4, 3 * N] -> [B, N, 4, 3]
    query = query.unsqueeze(2).expand(-1, -1, 6, -1, -1)  # [B, N, 4, 3] -> [B, N, 6, 4, 3]
    candidates = candidates.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [N, 6, 4, 3] -> [B, N, 6, 4, 3]

    if strategy == 'euclidean':
        dist = euclidean_distance(query, candidates)
    elif strategy == 'wasserstein':
        dist = wasserstein_1d_distance(query, candidates)
    elif strategy == 'usage_weighted_distance':
        dist = usage_weighted_distance(query, candidates, tgt_aminos, gamma=gamma)
    elif strategy == 'specific_adaptiveness':
        dist = euclidean_distance(query, candidates)
        dist = target_cai_weighted_distance(
            dist=dist,
            tgt_aminos=tgt_aminos,
            cai_usage_table=CAI_usage_table,
            gamma=gamma,
        )
    else:
        raise ValueError(f'Unknown strategy {strategy}')

    codon_idx = dist.argmin(-1)  # [B, N, 6] -> [B, N]
    codon_idx_exp = codon_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 3)  # [B, N] -> [B, N, 4, 3]
    codon_images = torch.gather(candidates, 2, codon_idx_exp.unsqueeze(2)).squeeze(2)

    return codon_images  # [B, N, 4, 3]


def wasserstein_1d_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    # query, candidates [B, N, 6, 4, 3]
    # output: [B, N, 6]

    # CDF over base dimension
    query_cdf = torch.cumsum(query, dim=3)  # [B, N, 6, 4, 3]
    candidate_cdf = torch.cumsum(candidates, dim=3)

    # abs diff * support
    wasser = (query_cdf - candidate_cdf).abs()  # w = 1
    wasser = wasser.sum(dim=[3, 4])

    return wasser


def euclidean_distance(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    # query, candidates [B, N, 6, 4, 3]
    # output: [B, N, 6]
    diff = (query - candidates).abs()
    dist = torch.sqrt(diff ** 2).sum(dim=[3, 4])

    return dist


def usage_weighted_distance(query: torch.Tensor, candidates: torch.Tensor, tgt_aminos, gamma: float = 1.0, ) -> torch.Tensor:
    """
    query, candidates: [B, N, 6, 4, 3]
    output: [B, N, 6]

    score_i = distance_i / (p_i ** gamma)
    """

    # 1. base distance
    dist = euclidean_distance(query, candidates)  # [B, N, 6]

    # 2. build codon usage prob tensor [N, 6]
    prob_rows, valid_rows = [], []

    for aa in tgt_aminos:
        usage_dict = AA_TO_CODON_USAGE_HUMAN_RNA[aa]
        probs = list(usage_dict.values())

        valid_len = len(probs)
        probs = probs + [0.0] * (6 - valid_len)
        valid = [True] * valid_len + [False] * (6 - valid_len)

        prob_rows.append(probs)
        valid_rows.append(valid)

    codon_prob = torch.tensor(prob_rows, dtype=dist.dtype, device=dist.device, )  # [N, 6]
    valid_mask = torch.tensor(valid_rows, dtype=torch.bool, device=dist.device, )  # [N, 6]

    # 3. score = distance / p^gamma
    eps = 1e-8
    score = dist / ((codon_prob.unsqueeze(0) + eps) ** gamma)

    # 4. avoid padded codons
    score = score.masked_fill(~valid_mask.unsqueeze(0), float("inf"))

    return score


def target_cai_weighted_distance(
        dist: torch.Tensor,
        tgt_aminos: list[str],
        cai_usage_table: dict,
        gamma: float = 0.04,
):
    """
    dist: [B, N, 6]
    tgt_aminos: length N
    cai_usage_table[aa]: [6]

    return:
        weighted_dist: [B, N, 6]
    """

    device = dist.device
    dtype = dist.dtype

    prob = torch.stack([cai_usage_table[aa].to(device=device, dtype=dtype) for aa in tgt_aminos], dim=0)  # [N, 6]
    valid_mask = prob > 0

    weighted_dist = dist * (1.0 / (prob.unsqueeze(0) + 1e-12) ** gamma)

    weighted_dist = weighted_dist.masked_fill(~valid_mask.unsqueeze(0), float("inf"))

    return weighted_dist


def get_amino_images_by_codon_usage(
        tgt_aminos,
        sample_size: int,
        device,
        base2vec=inf_base2vec,
):
    """
    Return:
        codon_images: [B, N, 4, 3]
    """

    all_amino_images = []

    for amino in tgt_aminos:

        usage_dict = AA_TO_CODON_USAGE_HUMAN_RNA[amino]

        codons = list(usage_dict.keys())
        probs = torch.tensor(list(usage_dict.values()), dtype=torch.float, device=device)
        probs = probs / probs.sum()

        sampled_idx = torch.multinomial(probs, num_samples=sample_size, replacement=True)

        sampled_codons = [codons[i] for i in sampled_idx.detach().cpu().tolist()]

        codon_images_for_amino = []

        for codon in sampled_codons:
            codon = codon.upper()

            codon_image = torch.tensor(
                [base2vec[rna_to_dna(base)] for base in codon],
                dtype=torch.float,
                device=device
            ).T  # [3, 4] -> [4, 3]

            codon_images_for_amino.append(codon_image)

        codon_images_for_amino = torch.stack(codon_images_for_amino, dim=0)  # [B, 4, 3]

        all_amino_images.append(codon_images_for_amino)

    codon_images = torch.stack(all_amino_images, dim=1)  # [B, N, 4, 3]

    return codon_images