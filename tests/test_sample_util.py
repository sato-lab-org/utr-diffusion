import torch

from src.utils.sample_util import (
    inference,
    inference_continuous_double_label,
    inference_continuous_multi_label,
    inference_continuous_single_label,
    inference_double_label,
)


class MockDiffusion:
    def __init__(self):
        self.classes = []

    def sample(self, classes, shape, cond_weight, output_all_steps=False):
        self.classes.append(classes)
        sample = torch.zeros(shape)
        sample[:, :, 0, :] = 1.0
        return [sample.clone(), sample] if output_all_steps else sample


def test_inference_is_conditioned_by_default_and_returns_single_label_steps():
    diffusion = MockDiffusion()

    sequences, steps = inference(
        diffusion,
        class_num=1,
        sample_bs=1,
        seq_len=3,
        output_all_steps=True,
        device="cpu",
    )

    assert diffusion.classes[0] is not None
    assert torch.equal(diffusion.classes[0], torch.tensor([1.0]))
    assert sequences == [">class_1_seq_0\nAAA\n"]
    assert set(steps) == {1}
    assert steps[1].shape == (2, 1, 4, 3)


def test_legacy_unconditional_switch_passes_no_class_labels():
    diffusion = MockDiffusion()

    sequences = inference(
        diffusion,
        class_num=1,
        sample_bs=1,
        seq_len=3,
        device="cpu",
        with_condition=False,
    )

    assert diffusion.classes == [None]
    assert sequences == [">class_1_seq_0\nAAA\n"]


def test_per_target_helpers_keep_all_step_mappings():
    diffusion = MockDiffusion()

    _, discrete = inference_double_label(
        diffusion, num_labels=2, num_classes=1, sample_bs=1, seq_len=3,
        output_all_steps=True, device="cpu",
    )
    _, continuous_single = inference_continuous_single_label(
        diffusion, target_values=[4.0], sample_bs=1, seq_len=3,
        output_all_steps=True, device="cpu",
    )
    _, continuous_double = inference_continuous_double_label(
        diffusion, target_values=[[4.0, -20.0]], sample_bs=1, seq_len=3,
        output_all_steps=True, device="cpu",
    )
    _, continuous_multi = inference_continuous_multi_label(
        diffusion, target_values=[[4.0, -20.0]], label_names=["MRL", "MFE"],
        sample_bs=1, seq_len=3, output_all_steps=True, device="cpu",
    )

    assert set(discrete) == {"(1, 1)"}
    assert set(continuous_single) == {0}
    assert set(continuous_double) == {0}
    assert set(continuous_multi) == {0}
