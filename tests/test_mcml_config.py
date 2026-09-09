import unittest
from pathlib import Path

from src.models.mcml_config import (
    MCML_CHECKPOINT_PATH,
    MCML_DIFFUSION_KWARGS,
    MCML_LABEL_NAMES,
    MCML_LOCAL_CHECKPOINT_PATH,
    MCML_TRAIN_CHECKPOINT_DIR,
    MCML_TRAIN_OUTPUT_DIR,
    MCML_UNET_KWARGS,
    build_mcml_diffusion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class McmlReleaseConfigTests(unittest.TestCase):
    def test_checkpoint_compatible_defaults(self):
        self.assertEqual(MCML_LABEL_NAMES, ("real_MRL", "pred_MFE"))
        self.assertEqual(
            MCML_UNET_KWARGS,
            {
                "dim": 200,
                "init_dim": 200,
                "channels": 1,
                "dim_mults": (1, 2, 4),
                "resnet_block_groups": 4,
                "learned_sinusoidal_dim": 18,
                "seq_len": 50,
                "output_attention": False,
                "dropout": 0.2,
                "num_label": 2,
                "label_emb_mode": "per_label",
            },
        )
        self.assertEqual(
            MCML_DIFFUSION_KWARGS,
            {
                "timestep": 200,
                "beta_last": 0.01,
                "condition_weight": 4,
                "uncondition_prop": 0.2,
                "label_wise_mask": False,
            },
        )

    def test_release_paths_are_unambiguous(self):
        self.assertEqual(MCML_CHECKPOINT_PATH, "checkpoints/mcml_epoch_2000.pt")
        self.assertEqual(MCML_TRAIN_OUTPUT_DIR, "outputs/real_MRL_pred_MFE_260k_mcml")
        self.assertEqual(
            MCML_TRAIN_CHECKPOINT_DIR,
            "outputs/real_MRL_pred_MFE_260k_mcml/checkpoints",
        )
        self.assertEqual(
            MCML_LOCAL_CHECKPOINT_PATH,
            "outputs/real_MRL_pred_MFE_260k_mcml/checkpoints/epoch_2000.pt",
        )

    def test_train_and_sample_use_the_shared_factory(self):
        for relative_path in ("src/scripts/train_mcml.py", "src/scripts/sample_mcml.py"):
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("from src.models.mcml_config import", source)
            self.assertIn("build_mcml_diffusion()", source)

    def test_mcml_training_opts_into_the_release_checkpoint_directory(self):
        source = (REPO_ROOT / "src/scripts/train_mcml.py").read_text(encoding="utf-8")
        self.assertIn("MCML_TRAIN_CHECKPOINT_DIR", source)
        self.assertIn("checkpoint_dir=MCML_TRAIN_CHECKPOINT_DIR", source)

    def test_design_uses_shared_factory_and_checkpoint_default(self):
        source = (REPO_ROOT / "design_utr.py").read_text(encoding="utf-8")
        self.assertIn(
            "from src.models.mcml_config import MCML_CHECKPOINT_PATH, build_mcml_diffusion",
            source,
        )
        self.assertIn("default=MCML_CHECKPOINT_PATH", source)
        self.assertIn("build_mcml_diffusion(", source)

    def test_training_keeps_all_three_authoritative_datasets(self):
        source = (REPO_ROOT / "src/scripts/train_mcml.py").read_text(encoding="utf-8")
        for filename in (
            "real_MRL_pred_MFE_260k.csv",
            "real_MRL_pred_MFE_260k_missing_MFE.csv",
            "real_MRL_pred_MFE_260k_missing_MRL.csv",
        ):
            self.assertIn(filename, source)

    def test_small_model_can_be_constructed(self):
        try:
            diffusion = build_mcml_diffusion(
                unet_overrides={
                    "dim": 8,
                    "init_dim": 8,
                    "resnet_block_groups": 4,
                    "dropout": 0.0,
                },
                timestep=2,
            )
        except ModuleNotFoundError as exc:
            if exc.name in {"torch", "einops", "memory_efficient_attention_pytorch", "numpy"}:
                self.skipTest(f"optional model dependency is unavailable: {exc.name}")
            raise

        self.assertEqual(diffusion.timestep, 2)
        self.assertEqual(diffusion.model.seq_len, 50)
        self.assertEqual(diffusion.model.num_label, 2)
        self.assertEqual(diffusion.model.label_emb_mode, "per_label")

    def test_unet_has_no_tensorflow_or_stale_typing_imports(self):
        source = (REPO_ROOT / "src/models/unet_mcml.py").read_text(encoding="utf-8")
        self.assertNotIn("tensorflow", source)
        self.assertNotIn("cProfile", source)
        self.assertNotIn("typing import Optional", source)


if __name__ == "__main__":
    unittest.main()
