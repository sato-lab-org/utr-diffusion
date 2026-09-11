import argparse
import math
import subprocess
import sys
from functools import partial
from pathlib import Path

import torch

from src.models.diffusion_mcml import Diffusion_Masked_Continuous_Multi_Labels as Diffusion_MCML
from src.models.unet_mcml import UNet_Masked_Continuous_Multi_Labels as UNet_MCML
from src.models.repaint.amino_codon_table import AMINO_TO_CODONS
from src.models.repaint.repaint_amino_cml import RePaint_Amino_Continuous_Multi_Labels as Repaint_Amino_CML
from src.models.repaint.utils import bulid_gt_and_mask_from_codons, write_fasta


def build_parser():
    p = argparse.ArgumentParser(description="Generate 50-nt UTR sequences with optional single-value MRL/MFE conditioning and optional RePaint base or amino-acid constraints; CAI guidance requires a complete CDS-amino suffix.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/mcml_epoch_2000.pt", help="Path to the MCML checkpoint (default: checkpoints/mcml_epoch_2000.pt).")
    p.add_argument("--mrl", type=float, default=None, help="Target MRL; recommended range 2 to 9 (training data: 0 to 12); omit to leave MRL unconditioned. Example: --mrl 6")
    p.add_argument("--mfe", type=float, default=None, help="Target MFE; recommended range -30 to 0 (training data: -31 to 0); omit to leave MFE unconditioned. Example: --mfe -20")
    p.add_argument("--cai", type=float, default=None, help="CAI-guidance target for codon relative adaptiveness in (0, 1]; requires --cds-amino. Example: --cai 0.7")
    p.add_argument("--base", nargs="*", default=None, help="Fixed 3-nt constraints at 0-based nucleotide starts as POS:CODON. Example: --base 2:AGC 8:GTG")
    p.add_argument("--amino", nargs="*", default=None, help="Sparse amino-acid constraints at 0-based nucleotide starts as POS:AA. Example: --amino 26:M 31:D 37:L")
    p.add_argument("--cds-amino", nargs="*", default=None, help="Complete contiguous CDS suffix as 0-based nucleotide POS:AA entries; must start with M and end at nucleotide 49. Example: --cds-amino 32:M 35:A 38:G 41:L 44:K 47:L")
    p.add_argument("--out", type=str, default="demo_output/demo.fasta", help="Output FASTA path; existing output is overwritten (default: demo_output/demo.fasta).")
    p.add_argument("--batch-size", type=int, default=100, help="Number of sequences to generate (default: 100).")
    p.add_argument("--cond-weight", type=float, default=4.0, help="Classifier-free guidance weight (default: 4.0).")
    p.add_argument("--do-eval", action="store_true", help="Evaluate generated sequences and save the CSV and plots next to the FASTA.")
    p.add_argument("--eval-repo", type=str, default="evaluation", help="Path to the bundled evaluation directory (default: evaluation).")
    p.add_argument("--device", type=str, default="cuda:0", help="Sampling device; CUDA sampling uses FP16 (default: cuda:0).")
    return p


def parse_index_value_pairs(items, value_name):
    pos_list, value_list = [], []
    previous_end = 0
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --{value_name} item {item!r}; use pos:value.")
        pos_text, value = item.split(":", 1)
        pos = int(pos_text)
        value = value.strip().upper()
        if value_name == "base":
            value = value.replace("U", "T")
            if len(value) != 3 or set(value).difference("ACGT"):
                raise ValueError(f"Invalid codon {value!r}; each codon must contain exactly three A/C/G/T/U bases.")
        elif value not in AMINO_TO_CODONS:
            raise ValueError(f"Invalid amino acid {value!r}.")
        if pos < 0 or pos + 3 > 50:
            raise ValueError(f"Constraint {item!r} is outside the 50-nt sequence.")
        if pos < previous_end:
            raise ValueError(f"Constraints overlap or are not ordered around position {pos}.")
        pos_list.append(pos)
        value_list.append(value)
        previous_end = pos + 3
    return pos_list, value_list


def validate_cds_amino(pos_list, amino_list):
    if len(amino_list) < 2:
        raise ValueError("--cds-amino requires an initiating M and at least one downstream amino acid.")
    if amino_list[0] != "M":
        raise ValueError("The first --cds-amino constraint must be the initiating amino acid M.")
    if "*" in amino_list:
        raise ValueError("--cds-amino does not accept a stop-codon constraint.")
    expected_start = 50 - 3 * len(amino_list)
    expected_positions = list(range(expected_start, 50, 3))
    if pos_list != expected_positions:
        expected = " ".join(f"{pos}:{amino}" for pos, amino in zip(expected_positions, amino_list))
        raise ValueError(f"--cds-amino must be contiguous and fill the 3' suffix of the 50-nt sequence: {expected}")


def p_sample_guided_mcml(diffusion, x, classes, t, t_index, context_mask=None, cond_weight=0.0, uncond_mask=None):
    mask = uncond_mask if uncond_mask is not None else context_mask
    if mask is None:
        raise ValueError("The MCML guidance mask is missing.")
    return Diffusion_MCML.p_sample_guided(diffusion, x=x, classes=classes, t=t, t_index=t_index, uncond_mask=mask, cond_weight=cond_weight)


def build_diffusion(args, device):
    unet = UNet_MCML(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
        seq_len=50,
        dropout=0.2,
        num_label=2,
        label_emb_mode="per_label",
    )
    diffusion = Diffusion_MCML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        condition_weight=args.cond_weight,
        uncondition_prop=0.2,
        label_wise_mask=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    diffusion.load_state_dict(checkpoint["model"])
    diffusion = diffusion.to(device)
    diffusion.eval()
    diffusion.p_sample_guided = partial(p_sample_guided_mcml, diffusion)
    return diffusion


def run_evaluator(fasta_path, eval_repo, device):
    fasta_path = str(Path(fasta_path).resolve())
    eval_dir = Path(eval_repo).resolve()
    eval_script = eval_dir / "evaluate.py"
    model_path = eval_dir / "Model/model.pt"
    if not eval_script.exists():
        raise FileNotFoundError(f"[Eval] evaluate.py not found: {eval_script}")
    if not model_path.exists():
        raise FileNotFoundError(f"[Eval] model not found: {model_path}")
    cmd = [
        sys.executable,
        "evaluate.py",
        "--fasta", fasta_path,
        "--model", "Model/model.pt",
        "--device", device,
        "--batch-toks", str(4096 * 8),
        "--seed", "1337",
        "--mfe-batch", "100",
    ]
    print("[Eval] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(eval_dir), check=True)
    out_csv = str(Path(fasta_path).with_suffix(".csv"))
    print(f"[Eval] OK -> {out_csv}", flush=True)
    return out_csv


def design_utr(args):
    for name in ("base", "amino", "cds_amino"):
        if getattr(args, name) == []:
            raise ValueError(f"--{name.replace('_', '-')} requires at least one pos:value constraint.")
    constraints = [args.base is not None, args.amino is not None, args.cds_amino is not None]
    if sum(constraints) > 1:
        raise ValueError("Use only one of --base, --amino, or --cds-amino.")
    if args.cai is not None:
        if not math.isfinite(args.cai) or not 0 < args.cai <= 1:
            raise ValueError("--cai accepts one finite value in (0, 1].")
        if not args.cds_amino:
            raise ValueError("--cai requires a complete --cds-amino suffix.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} is unavailable; sampling was not allowed to fall back from FP16 to CPU/FP32.")
    targets = [[args.mrl if args.mrl is not None else float("nan"), args.mfe if args.mfe is not None else float("nan")]]
    diffusion = build_diffusion(args, device)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        if args.base:
            # The restored legacy codon sampler imports this old helper name but
            # does not use it. Keep the compatibility local to this entry point.
            from src.models.repaint import utils as repaint_utils
            if not hasattr(repaint_utils, "get_amino_images_for_alter_codons"):
                repaint_utils.get_amino_images_for_alter_codons = repaint_utils.aminos_to_amino_images
            from src.models.repaint.repaint_codon_cml import RePaint_Codon_Continuous_Multi_Labels as Repaint_Codon_CML
            pos_list, codon_list = parse_index_value_pairs(args.base, "base")
            repaint = Repaint_Codon_CML(diffusion=diffusion, sample_bs=args.batch_size, seq_len=50, cond_weight=args.cond_weight, tgt_labels=targets, return_all=False)
            gt, mask = bulid_gt_and_mask_from_codons(codon_list=codon_list, pos_list=pos_list)
            result = repaint.p_resample(gt=gt.to(device), mask=mask.to(device))
        elif args.amino or args.cds_amino:
            constraint_name = "cds-amino" if args.cds_amino else "amino"
            constraints_text = args.cds_amino if args.cds_amino else args.amino
            pos_list, amino_list = parse_index_value_pairs(constraints_text, constraint_name)
            if args.cds_amino:
                validate_cds_amino(pos_list, amino_list)
            strategy = "specific_adaptiveness" if args.cai is not None else "wasserstein"
            repaint = Repaint_Amino_CML(diffusion=diffusion, sample_bs=args.batch_size, seq_len=50, cond_weight=args.cond_weight, tgt_labels=targets, strategy=strategy, gamma_for_usage_frequency=0.04, return_all=False)
            repaint.setup(amino_list=amino_list, pos_list=pos_list, adaptiveness=args.cai)
            result = repaint.p_resample()
        else:
            labels = torch.tensor([target for target in targets for _ in range(args.batch_size)], dtype=torch.float, device=device)
            result = diffusion.sample(classes=labels, shape=(len(labels), 1, 4, 50), cond_weight=args.cond_weight)
            result = result.detach().cpu().to(torch.float16).numpy()

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".fasta":
        raise ValueError("--out must use the .fasta suffix.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(result, str(out_path), tgt_values=targets, batch_bs=args.batch_size)
    print(f"[OK] Saved: {out_path}")

    if args.do_eval:
        out_csv = run_evaluator(fasta_path=args.out, eval_repo=args.eval_repo, device=str(device))
        from src.plot.visualization import read_csv_and_plot
        read_csv_and_plot(out_csv, args)


if __name__ == "__main__":
    design_utr(build_parser().parse_args())
