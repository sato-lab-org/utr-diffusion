import os
import yaml
from datetime import datetime

def save_experiment_config(
        save_dir,
        checkpoint_path,
        repaint,
        target_labels,
        patterns,
        family_name,
        extra_config=None,
        overwrite=False,
):
    """
    Save sampling / repaint experiment configuration.
    Future evaluation scripts can directly load this config.
    """
    if os.path.exists(os.path.join(save_dir, 'experiment_config.yaml')) and not overwrite:
        print(f'[INFO] Skip existing experiment config for pattern family: {family_name}')
        return

    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # checkpoint
        "checkpoint_path": checkpoint_path,

        # repaint config
        "repaint": {
            "sample_bs": repaint.sample_bs,
            "seq_len": repaint.seq_len,
            "cond_weight": repaint.cond_weight,
            "return_all": repaint.return_all,
            "skip_frames": repaint.skip_frames,
        },

        # target labels
        "target_labels": target_labels,

        # codon patterns
        "patterns": patterns,
    }

    if extra_config is not None:
        config["extra_config"] = extra_config

    yaml_path = os.path.join(save_dir, "experiment_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"[INFO] Experiment config saved to {yaml_path}")
