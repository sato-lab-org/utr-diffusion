import inspect

import pytest
import torch

from src.models.repaint.amino_codon_table import (
    AA_TO_CODON_USAGE_HUMAN_RNA,
    AMINO_TO_CODONS,
)
from src.models.repaint.repaint_amino_cml import (
    RePaint_Amino_Continuous_Multi_Labels,
    choose_codon_by_strategy,
)
from src.models.repaint.repaint_codon_cml import (
    RePaint_Codon_Continuous_Multi_Labels,
)
from src.models.repaint.scheduler import (
    DEFAULT_REPAINT_SCHEDULE,
    get_schedule_jump,
    resolve_schedule,
)
from src.models.repaint.utils import (
    build_codon_usage_table_for_specific_CAI,
    build_gt_mask_from_aminos,
    build_target_adaptiveness_probability_table,
    probabilities_for_target_adaptiveness,
    validate_amino_constraints,
    validate_target_adaptiveness,
)


TINY_SCHEDULE = {
    't_T': 1,
    'n_sample': 1,
    'jump_length': 1,
    'jump_n_sample': 1,
}


def test_repaint_amino_keeps_legacy_positional_prefixes():
    init_parameters = list(
        inspect.signature(RePaint_Amino_Continuous_Multi_Labels.__init__).parameters
    )
    assert init_parameters[:10] == [
        "self",
        "diffusion",
        "tgt_labels",
        "sample_bs",
        "seq_len",
        "cond_weight",
        "strategy",
        "stop_point",
        "skip_frames",
        "return_all",
    ]
    assert init_parameters[10:] == ["gamma_for_usage_frequency"]

    strategy_parameters = list(inspect.signature(choose_codon_by_strategy).parameters)
    assert strategy_parameters[:3] == ["query", "candidates", "strategy"]


class _BaseMockDiffusion:
    device = torch.device('cpu')
    timestep = 1

    def __init__(self):
        self.calls = []
        self.forward_timesteps = []

    def q_sample(self, x_start, t, noise=None):
        return x_start

    def q_sample_single_step(self, x_t_m1, t, noise=None):
        self.forward_timesteps.extend(t.detach().cpu().tolist())
        return x_t_m1


class MockMCMLDiffusion(_BaseMockDiffusion):
    def p_sample_guided(
            self,
            x,
            classes,
            t,
            t_index,
            uncond_mask,
            cond_weight,
    ):
        self.calls.append({
            'mask_name': 'uncond_mask',
            'mask': uncond_mask.clone(),
            'classes': classes.clone(),
        })
        return x


class MockCMLDiffusion(_BaseMockDiffusion):
    def p_sample_guided(
            self,
            x,
            classes,
            t,
            t_index,
            context_mask,
            cond_weight,
    ):
        self.calls.append({
            'mask_name': 'context_mask',
            'mask': context_mask.clone(),
            'classes': classes.clone(),
        })
        return x


class ExplodingMCMLDiffusion(_BaseMockDiffusion):
    def p_sample_guided(
            self,
            x,
            classes,
            t,
            t_index,
            uncond_mask,
            cond_weight,
    ):
        self.calls.append('called')
        raise TypeError('internal sampler failure')


@pytest.mark.parametrize('amino', ['A', 'R', 'L', 'M', 'W'])
@pytest.mark.parametrize('target', [0.5, 0.7, 1.0])
def test_target_adaptiveness_table_is_normalized_and_has_expected_value(amino, target):
    table = build_target_adaptiveness_probability_table(target)
    codons = AMINO_TO_CODONS[amino]
    probability = table[amino]
    valid_probability = probability[:len(codons)].to(torch.float64)
    usage = torch.tensor(
        [AA_TO_CODON_USAGE_HUMAN_RNA[amino][codon] for codon in codons],
        dtype=torch.float64,
    )
    relative_adaptiveness = usage / usage.max()
    effective_target = min(max(target, float(relative_adaptiveness.min())), 1.0)

    assert probability.shape == (6,)
    assert torch.isfinite(probability).all()
    assert (probability >= 0).all()
    assert float(valid_probability.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(torch.dot(valid_probability, relative_adaptiveness)) == pytest.approx(
        effective_target,
        abs=2e-6,
    )
    assert torch.equal(probability[len(codons):], torch.zeros(6 - len(codons)))


def test_probability_table_can_report_effective_targets_after_clipping():
    table, effective = build_target_adaptiveness_probability_table(
        0.5,
        return_effective_targets=True,
    )

    assert table["M"][0] == pytest.approx(1.0)
    assert effective["M"] == pytest.approx(1.0)
    assert effective["W"] == pytest.approx(1.0)
    assert effective["L"] >= 0.5


def test_probability_builder_rejects_zero_base_probability():
    with pytest.raises(ValueError, match='strictly positive'):
        probabilities_for_target_adaptiveness(
            q=torch.tensor([1.0, 0.0]),
            w=torch.tensor([1.0, 0.5]),
            target_adaptiveness=0.7,
        )


def test_legacy_cai_builder_is_an_alias_for_target_adaptiveness():
    current = build_target_adaptiveness_probability_table(0.7)
    legacy = build_codon_usage_table_for_specific_CAI(
        amino_to_codons=AA_TO_CODON_USAGE_HUMAN_RNA,
        codon_usage_table=AA_TO_CODON_USAGE_HUMAN_RNA,
        target_cai=0.7,
    )
    assert current.keys() == legacy.keys()
    for amino in current:
        torch.testing.assert_close(current[amino], legacy[amino])


def test_amino_constraint_validation_normalizes_lowercase():
    assert validate_amino_constraints(['m', 'k'], [0, 3], 6) == (
        ['M', 'K'],
        [0, 3],
    )


@pytest.mark.parametrize(
    ('aminos', 'positions', 'error'),
    [
        ([], [], 'at least one'),
        (['A'], [], 'same length'),
        (['*'], [0], 'stop-codon'),
        (['X'], [0], 'unsupported'),
        (['A'], [-1], 'bounds'),
        (['A'], [48], 'bounds'),
        (['A', 'K'], [3, 0], 'sorted'),
        (['A', 'K'], [0, 2], 'overlap'),
    ],
)
def test_amino_constraint_validation_rejects_invalid_inputs(aminos, positions, error):
    with pytest.raises((TypeError, ValueError), match=error):
        validate_amino_constraints(aminos, positions, 50)


@pytest.mark.parametrize('target', [True, float('nan'), float('inf'), 0.0, -0.1, 1.1])
def test_target_adaptiveness_validation_rejects_invalid_values(target):
    with pytest.raises((TypeError, ValueError)):
        validate_target_adaptiveness(target)


def test_amino_setup_and_no_argument_sampling_use_mcml_mask():
    diffusion = MockMCMLDiffusion()
    sampler = RePaint_Amino_Continuous_Multi_Labels(
        diffusion=diffusion,
        tgt_labels=[(8.0, -2.0)],
        sample_bs=2,
        seq_len=12,
        strategy='specific_adaptiveness',
        return_all=False,
    )
    sampler.setup(['A', 'K'], [6, 9], adaptiveness=0.7)
    result = sampler.p_resample(schedule=TINY_SCHEDULE)

    assert result.shape == (2, 1, 4, 12)
    assert sampler.gamma_for_usage_frequency == pytest.approx(0.04)
    assert len(diffusion.calls) == 1
    call = diffusion.calls[0]
    assert call['mask_name'] == 'uncond_mask'
    assert call['classes'].shape == (4, 2)
    assert call['mask'].shape == (4, 2)
    torch.testing.assert_close(call['mask'][:2], torch.ones(2, 2))
    torch.testing.assert_close(call['mask'][2:], torch.zeros(2, 2))


def test_amino_sampler_preserves_legacy_four_argument_api():
    diffusion = MockCMLDiffusion()
    sampler = RePaint_Amino_Continuous_Multi_Labels(
        diffusion=diffusion,
        tgt_labels=[(8.0, -2.0)],
        sample_bs=1,
        seq_len=12,
        return_all=False,
    )
    gt, mask = build_gt_mask_from_aminos(['A', 'K'], [6, 9], total_length=12)
    result = sampler.p_resample(gt, mask, ['A', 'K'], [6, 9], TINY_SCHEDULE)

    assert result.shape == (1, 1, 4, 12)
    assert diffusion.calls[0]['mask_name'] == 'context_mask'


def test_amino_sampler_requires_complete_setup_or_legacy_arguments():
    sampler = RePaint_Amino_Continuous_Multi_Labels(
        diffusion=MockMCMLDiffusion(),
        tgt_labels=[(8.0, -2.0)],
        sample_bs=1,
        seq_len=12,
    )
    with pytest.raises(RuntimeError, match='setup'):
        sampler.p_resample(schedule=TINY_SCHEDULE)
    with pytest.raises(TypeError, match='requires gt, mask'):
        sampler.p_resample(gt=torch.zeros(1, 4, 12), schedule=TINY_SCHEDULE)


def test_failed_second_setup_does_not_corrupt_previous_state():
    sampler = RePaint_Amino_Continuous_Multi_Labels(
        diffusion=MockMCMLDiffusion(),
        tgt_labels=[(8.0, -2.0)],
        sample_bs=1,
        seq_len=12,
        strategy='specific_adaptiveness',
    )
    sampler.setup(['A'], [9], adaptiveness=0.7)
    old_state = (sampler.tgt_aminos, sampler.pos_list, sampler.gt_image.clone())
    with pytest.raises(ValueError, match='required'):
        sampler.setup(['K'], [6])

    assert sampler.tgt_aminos == old_state[0]
    assert sampler.pos_list == old_state[1]
    torch.testing.assert_close(sampler.gt_image, old_state[2])


@pytest.mark.parametrize(
    ('diffusion_class', 'expected_mask_name'),
    [
        (MockMCMLDiffusion, 'uncond_mask'),
        (MockCMLDiffusion, 'context_mask'),
    ],
)
@pytest.mark.parametrize('targets, label_dim', [([1.0, 2.0], 1), ([(1.0, 2.0, 3.0)], 3)])
def test_codon_sampler_supports_both_mask_names_and_generic_labels(
        diffusion_class,
        expected_mask_name,
        targets,
        label_dim,
):
    diffusion = diffusion_class()
    sampler = RePaint_Codon_Continuous_Multi_Labels(
        diffusion=diffusion,
        tgt_labels=targets,
        sample_bs=1,
        seq_len=6,
        return_all=False,
    )
    result = sampler.p_resample(
        gt=torch.zeros(1, 4, 6),
        mask=torch.zeros(1, 4, 6),
        schedule=TINY_SCHEDULE,
    )

    batch_size = len(targets)
    assert result.shape == (batch_size, 1, 4, 6)
    assert diffusion.calls[0]['mask_name'] == expected_mask_name
    assert diffusion.calls[0]['classes'].shape == (2 * batch_size, label_dim)
    assert diffusion.calls[0]['mask'].shape == (2 * batch_size, label_dim)


def test_internal_type_error_is_not_caught_and_retried():
    diffusion = ExplodingMCMLDiffusion()
    sampler = RePaint_Codon_Continuous_Multi_Labels(
        diffusion=diffusion,
        tgt_labels=[(8.0, -2.0)],
        sample_bs=1,
        seq_len=6,
        return_all=False,
    )
    with pytest.raises(TypeError, match='internal sampler failure'):
        sampler.p_resample(
            gt=torch.zeros(1, 4, 6),
            mask=torch.zeros(1, 4, 6),
            schedule=TINY_SCHEDULE,
        )
    assert diffusion.calls == ['called']


def test_schedule_defaults_are_copied_and_support_partial_overrides():
    first = resolve_schedule()
    second = resolve_schedule({'t_T': 1})
    first['t_T'] = 999

    assert DEFAULT_REPAINT_SCHEDULE['t_T'] == 200
    assert second['t_T'] == 1
    assert second['jump_length'] == DEFAULT_REPAINT_SCHEDULE['jump_length']
    assert get_schedule_jump(**resolve_schedule(TINY_SCHEDULE)) == [0, -1]

    with pytest.raises(ValueError, match='exceeds diffusion timestep'):
        resolve_schedule({'t_T': 2}, max_timestep=1)


def test_no_jump_counts_zero_and_one_produce_same_schedule():
    common = {'t_T': 5, 'n_sample': 1, 'jump_length': 2}
    zero = get_schedule_jump(**common, jump_n_sample=0)
    one = get_schedule_jump(**common, jump_n_sample=1)
    assert zero == one


def test_forward_jump_uses_destination_timestep_beta():
    diffusion = MockMCMLDiffusion()
    diffusion.timestep = 3
    sampler = RePaint_Codon_Continuous_Multi_Labels(
        diffusion=diffusion,
        tgt_labels=[(8.0, -2.0)],
        sample_bs=1,
        seq_len=6,
        return_all=False,
    )
    sampler.p_resample(
        gt=torch.zeros(1, 4, 6),
        mask=torch.zeros(1, 4, 6),
        schedule={
            't_T': 3,
            'n_sample': 1,
            'jump_length': 1,
            'jump_n_sample': 2,
        },
    )

    assert diffusion.forward_timesteps == [2, 1]
