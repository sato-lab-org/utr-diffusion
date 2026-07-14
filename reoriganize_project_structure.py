from pathlib import Path
import shutil
import re


def reorganize_trials(
    save_root="save",
    ckpt_root="checkpoints",
    output_root="design_outputs",
    mode="copy",          # "copy" or "move"
    dry_run=True,
    overwrite=False,
):
    """
    Old structure:
        save/
            trial_name/
                epoch_100.fasta
                epoch_100.csv
                all_images_denoising_process.pt
                all_sequences_denoising_process.fasta
                all_sequences_denoising_process.csv

            trial_name_at_2000epoch_sample_xxx/
                xxx.fasta
                xxx.csv
                xxx.jpg

        checkpoints/
            trial_name_at_1000epoch.pt
            trial_name_at_2000epoch.pt

    New structure:
        design_outputs/
            trial_name/
                checkpoints/
                    epoch_1000.pt
                    epoch_2000.pt

                samples/
                    epoch_100.fasta
                    all_images_denoising_process.pt
                    all_sequences_denoising_process.fasta

                samples/
                    sample_xxx/
                        xxx.fasta

                evals/
                    samples/
                        epoch_100.csv
                        all_sequences_denoising_process.csv
                        xxx.jpg

                    samples/
                        sample_xxx/
                            xxx.csv
                            xxx.jpg
    """
    save_root = Path(save_root)
    ckpt_root = Path(ckpt_root)
    output_root = Path(output_root)

    if mode not in {"copy", "move"}:
        raise ValueError("mode must be 'copy' or 'move'")

    if not save_root.exists():
        raise FileNotFoundError(f"save_root not found: {save_root}")

    if not ckpt_root.exists():
        raise FileNotFoundError(f"ckpt_root not found: {ckpt_root}")

    # --------------------------------------------------
    # helper functions
    # --------------------------------------------------
    def parse_ckpt_name(pt_name: str):
        """
        Supported checkpoint formats:
            trial_name_at_2000epoch.pt -> trial_name, 2000
            trial_name_epoch_2000.pt   -> trial_name, 2000
        """
        patterns = [
            r"^(.*)_at_(\d+)epoch\.pt$",
            r"^(.*)_epoch_(\d+)\.pt$",
        ]

        for pattern in patterns:
            m = re.match(pattern, pt_name)
            if m is not None:
                return m.group(1), int(m.group(2))

        return None, None

    def parse_sample_trial_name(folder_name: str):
        """
        Supported sample folder formats:

            trial_name_at_2000epoch_sample_xxx
            ->
            trial_name, 2000, epoch_2000_sample_xxx

            trial_name_sample_xxx
            ->
            trial_name, None, sample_xxx
        """

        # format 1: trial_name_at_2000epoch_sample_xxx
        m = re.match(r"^(.*)_at_(\d+)epoch_(sample_.*)$", folder_name)
        if m is not None:
            base_trial_name = m.group(1)
            ckpt_epoch = int(m.group(2))
            sample_name = f"epoch_{ckpt_epoch}_{m.group(3)}"
            return base_trial_name, ckpt_epoch, sample_name

        # format 2: trial_name_sample_xxx
        m = re.match(r"^(.*)_(sample_.*)$", folder_name)
        if m is not None:
            base_trial_name = m.group(1)
            ckpt_epoch = None
            sample_name = m.group(2)
            return base_trial_name, ckpt_epoch, sample_name

        return None, None, None

    def is_fasta(file_path: Path):
        return file_path.suffix.lower() in {".fasta", ".fa"}

    def is_eval_file(file_path: Path):
        return file_path.suffix.lower() in {
            ".csv", ".jpg", ".jpeg", ".png", ".pdf", ".json", ".txt"
        }

    def is_snapshot_related_file(file_path: Path):
        """
        Files belonging to normal training trial folder.
        """
        special_files = {
            "all_images_denoising_process.pt",
            "all_sequences_denoising_process.fasta",
            "all_sequences_denoising_process.csv",
        }

        if file_path.name in special_files:
            return True

        return re.match(
            r"^epoch_\d+.*\.(fasta|fa|csv|jpg|jpeg|png|pdf|json|txt|pt)$",
            file_path.name,
            re.IGNORECASE,
        ) is not None

    def is_snapshot_generation_file(file_path: Path):
        """
        Files that should go to:
            trial_name/samples/
        """
        if file_path.name in {
            "all_images_denoising_process.pt",
            "all_sequences_denoising_process.fasta",
        }:
            return True

        return is_fasta(file_path)

    def is_snapshot_eval_file(file_path: Path):
        """
        Files that should go to:
            trial_name/evals/samples/
        """
        if file_path.name == "all_sequences_denoising_process.csv":
            return True

        return is_eval_file(file_path)

    def transfer(src: Path, dst: Path):
        if dst.exists() and not overwrite:
            print(f"[skip] exists: {dst}")
            return

        print(f"[{mode}] {src} -> {dst}")

        if dry_run:
            return

        dst.parent.mkdir(parents=True, exist_ok=True)

        if mode == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    # --------------------------------------------------
    # collect checkpoints
    # --------------------------------------------------
    ckpt_map = {}

    for pt_file in ckpt_root.glob("*.pt"):
        trial_name, epoch = parse_ckpt_name(pt_file.name)

        if trial_name is None:
            print(f"[warn] unmatched checkpoint filename, skip: {pt_file.name}")
            continue

        ckpt_map.setdefault(trial_name, []).append((epoch, pt_file))

    for trial_name in ckpt_map:
        ckpt_map[trial_name] = sorted(ckpt_map[trial_name], key=lambda x: x[0])

    # --------------------------------------------------
    # process save folders
    # --------------------------------------------------
    processed_base_trials = set()

    for old_dir in sorted(save_root.iterdir()):
        if not old_dir.is_dir():
            continue

        folder_name = old_dir.name

        base_trial_name, ckpt_epoch, sample_name = parse_sample_trial_name(folder_name)

        # --------------------------------------------------
        # Case 1: post-training sampling folder
        # --------------------------------------------------
        if base_trial_name is not None:
            processed_base_trials.add(base_trial_name)

            print(f"\n=== sample trial: {folder_name} ===")
            print(f"[info] base trial: {base_trial_name}")
            print(f"[info] checkpoint epoch: {ckpt_epoch}")
            print(f"[info] sample name: {sample_name}")

            sample_root = output_root / base_trial_name / "samples" / sample_name
            sample_eval_root = output_root / base_trial_name / "evals" / "samples" / sample_name

            for file_path in sorted(old_dir.iterdir()):
                if not file_path.is_file():
                    continue

                if is_fasta(file_path):
                    dst = sample_root / file_path.name
                    transfer(file_path, dst)

                elif is_eval_file(file_path):
                    dst = sample_eval_root / file_path.name
                    transfer(file_path, dst)

                else:
                    print(f"[warn] unmatched sample file, skip: {file_path.name}")

            continue

        # --------------------------------------------------
        # Case 2: normal training trial folder
        # --------------------------------------------------
        trial_name = folder_name
        processed_base_trials.add(trial_name)

        print(f"\n=== training trial: {trial_name} ===")

        trial_root = output_root / trial_name
        ckpt_out_root = trial_root / "checkpoints"
        snapshot_root = trial_root / "samples"
        snapshot_eval_root = trial_root / "evals" / "samples"

        # checkpoints
        matched_ckpts = ckpt_map.get(trial_name, [])

        if not matched_ckpts:
            print(f"[info] no matching checkpoint found for: {trial_name}")
        else:
            for epoch, pt_path in matched_ckpts:
                dst = ckpt_out_root / f"epoch_{epoch}.pt"
                transfer(pt_path, dst)

        # snapshot / eval files
        for file_path in sorted(old_dir.iterdir()):
            if not file_path.is_file():
                continue

            if not is_snapshot_related_file(file_path):
                print(f"[warn] unmatched training file, skip: {file_path.name}")
                continue

            if is_snapshot_generation_file(file_path):
                dst = snapshot_root / file_path.name
                transfer(file_path, dst)

            elif is_snapshot_eval_file(file_path):
                dst = snapshot_eval_root / file_path.name
                transfer(file_path, dst)

            else:
                print(f"[warn] unsupported training file, skip: {file_path.name}")

    # --------------------------------------------------
    # checkpoint-only trials
    # --------------------------------------------------
    extra_trials = set(ckpt_map.keys()) - processed_base_trials

    if extra_trials:
        print("\n=== checkpoints without matching save folder ===")

        for trial_name in sorted(extra_trials):
            print(f"[info] checkpoint-only trial: {trial_name}")

            ckpt_out_root = output_root / trial_name / "checkpoints"

            for epoch, pt_path in ckpt_map[trial_name]:
                dst = ckpt_out_root / f"epoch_{epoch}.pt"
                transfer(pt_path, dst)

    print("\nDone.")


if __name__ == "__main__":
    reorganize_trials(
        save_root="save",
        ckpt_root="checkpoints",
        output_root="outputs",
        mode="copy",
        dry_run=False,
        overwrite=False,
    )