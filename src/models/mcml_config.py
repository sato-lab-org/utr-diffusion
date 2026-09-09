"""Authoritative architecture settings for the released MCML checkpoint.

Keep training, standalone sampling, and ``design_utr.py`` on these builders so
that a command-line change cannot silently create a checkpoint-incompatible
network.
"""

from pathlib import Path

MCML_LABEL_NAMES = ("real_MRL", "pred_MFE")
MCML_UNET_KWARGS = {
    "dim": 200,
    "init_dim": 200,
    "channels": 1,
    "dim_mults": (1, 2, 4),
    "resnet_block_groups": 4,
    "learned_sinusoidal_dim": 18,
    "seq_len": 50,
    "output_attention": False,
    "dropout": 0.2,
    "num_label": len(MCML_LABEL_NAMES),
    "label_emb_mode": "per_label",
}
MCML_DIFFUSION_KWARGS = {
    "timestep": 200,
    "beta_last": 0.01,
    "condition_weight": 4,
    "uncondition_prop": 0.2,
    "label_wise_mask": False,
}

MCML_CAI_GAMMA = 0.04
MCML_CHECKPOINT_STATE = "model"
MCML_HF_REPO_ID = "chuankai-dai/utr-diffusion-checkpoint"
MCML_HF_FILENAME = "checkpoints/mcml_epoch_2000.pt"
MCML_HF_REVISION = "b9bbe1e45531febd8b1fec6fbf20ff598af6408d"

# Training writes the released checkpoint below this repository-local output.
MCML_TRAIN_OUTPUT_DIR = "outputs/real_MRL_pred_MFE_260k_mcml"
MCML_TRAIN_CHECKPOINT_DIR = f"{MCML_TRAIN_OUTPUT_DIR}/checkpoints"
# Keep the train_mcml.py output filename unchanged from the completed run.
MCML_LOCAL_CHECKPOINT_PATH = f"{MCML_TRAIN_CHECKPOINT_DIR}/epoch_2000.pt"


def resolve_mcml_checkpoint_path(checkpoint: str | Path | None = None) -> Path:
    """Resolve an explicit/local checkpoint or download the pinned Hub release."""

    if checkpoint is not None:
        return Path(checkpoint).expanduser()

    repository_root = Path(__file__).resolve().parents[2]
    repository_checkpoint = repository_root / MCML_HF_FILENAME
    if repository_checkpoint.is_file():
        return repository_checkpoint

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic checkpoint download; "
            "install the release environment or pass --checkpoint PATH"
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=MCML_HF_REPO_ID,
            filename=MCML_HF_FILENAME,
            revision=MCML_HF_REVISION,
        )
    )


def build_mcml_unet(**overrides):
    """Build the checkpoint-compatible MCML U-Net, with explicit test overrides."""
    from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels

    kwargs = {**MCML_UNET_KWARGS, **overrides}
    return UNet_Masked_Continuous_Multi_Labels(**kwargs)


def build_mcml_diffusion(*, unet=None, unet_overrides=None, **diffusion_overrides):
    """Build the released MCML diffusion stack without importing training code."""
    from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels

    if unet is None:
        unet = build_mcml_unet(**(unet_overrides or {}))
    kwargs = {**MCML_DIFFUSION_KWARGS, **diffusion_overrides}
    return Diffusion_Masked_Continuous_Multi_Labels(model=unet, **kwargs)
