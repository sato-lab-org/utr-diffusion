import unittest

try:
    import torch
    from torch import nn

    from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels
except ModuleNotFoundError:
    torch = None
    nn = None
    Diffusion_Masked_Continuous_Multi_Labels = None


if nn is not None:
    class RecordingNoiseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.recorded_classes = None
            self.recorded_presence = None

        def forward(self, x, time=None, classes=None, label_is_presence=None):
            self.recorded_classes = classes.detach().clone()
            self.recorded_presence = label_is_presence.detach().clone()
            return torch.zeros_like(x)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DiffusionMcmlMathTests(unittest.TestCase):
    def make_diffusion(self, timestep=2):
        model = RecordingNoiseModel()
        diffusion = Diffusion_Masked_Continuous_Multi_Labels(
            model=model,
            timestep=timestep,
            beta_last=0.2,
            condition_weight=4,
        )
        return model, diffusion

    def test_single_forward_step_uses_one_minus_beta(self):
        _, diffusion = self.make_diffusion()
        x = torch.ones((1, 1, 4, 3))
        noise = torch.zeros_like(x)
        t = torch.tensor([1], dtype=torch.long)

        actual = diffusion.q_sample_single_step(x, t, noise=noise)
        expected = x * torch.sqrt(1 - diffusion.betas[1])
        torch.testing.assert_close(actual, expected)

    def test_guided_sampling_sanitizes_nan_but_preserves_presence(self):
        model, diffusion = self.make_diffusion(timestep=1)
        x = torch.zeros((2, 1, 4, 3))
        raw_classes = torch.tensor([[1.0, float("nan")], [float("nan"), -2.0]])
        classes = raw_classes.repeat(2, 1)
        uncond_mask = torch.cat([torch.ones_like(raw_classes), torch.zeros_like(raw_classes)])

        result = diffusion.p_sample_guided(
            x=x,
            classes=classes,
            t=torch.zeros(2, dtype=torch.long),
            t_index=0,
            uncond_mask=uncond_mask,
            cond_weight=4,
        )

        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(model.recorded_classes).all())
        torch.testing.assert_close(
            model.recorded_classes,
            torch.tensor([[1.0, 0.0], [0.0, -2.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        torch.testing.assert_close(
            model.recorded_presence,
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        )

    def test_reverse_guided_process_accepts_two_dimensional_labels(self):
        _, diffusion = self.make_diffusion(timestep=1)
        x_t = torch.zeros((2, 1, 4, 3))
        classes = torch.tensor([[1.0, float("nan")], [2.0, -3.0]])

        steps = diffusion.reverse_process_guided(x_t, classes)

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].shape, x_t.shape)
        self.assertTrue(torch.isfinite(steps[0]).all())


if __name__ == "__main__":
    unittest.main()
