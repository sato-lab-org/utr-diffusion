from src.models.diffusion_cml import Diffusion_Continuous_Multi_Labels as Diffusion_CML
from src.models.unet_cml import UNet_Continuous_Multi_Labels as UNet_CML
from src.models.repaint.repaint_amino_cml import RePaint_Amino_Continuous_Multi_Labels as Repaint_Amino_CML
from src.models.repaint.repaint_codon_cml import RePaint_Codon_Continuous_Multi_Labels as Repaint_Codon_CML
from src.models.repaint.utils import build_gt_mask_from_aminos
from src.models.repaint.utils import bulid_gt_and_mask_from_codons, write_fasta
import torch
import argparse
from pathlib import Path
import sys, subprocess

def build_parser():
    p = argparse.ArgumentParser(
        description="Design UTR with target MRL/MFE under codon/amino constraints (RePaint sampling)."
    )
    p.add_argument("--checkpoint", type=str, default='checkpoints/MRL_MFE_967k_ep_2k_ts_200_beta_0.01_cond_1_uncond_0.2_drop_0.2_lr_1e-4_at_epoch_2000.pt', help="Path to checkpoint .pt")
    p.add_argument("--mode", choices=["codon", "amino"], required=True, help="Constraint type")
    p.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="One or more targets in 'MRL,MFE' form. Example: --targets 4,-20",
    )
    p.add_argument(
        "--codon",
        nargs="*",
        default=None,
        help="Codon constraints as 'pos:CODON'. Example: --codon 2:AGC 8:GTG ...",
    )
    p.add_argument(
        "--amino",
        nargs="*",
        default=None,
        help="Amino constraints as 'pos:AA'. Example: --amino 2:M 5:F ...",
    )
    p.add_argument("--out", type=str, default='design_outputs/amino_demo.fasta', help="Output fasta path")
    # Sampling hyperparams
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--cond-weight", type=float, default=2.0)

    p.add_argument("--do-eval", action="store_true", help="Run UTR-Diffusion-eval/evaluate.py on the generated fasta.")
    p.add_argument("--eval-repo", type=str, default='../UTR-Diffusion-eval/', help="Path to UTR-Diffusion-eval repo.")
    p.add_argument('--device', type=str, default='cuda:0', help="cpu | cuda | cuda:0")
    return p

def parse_targets(tokens: list[str]) -> list[list[float]]:
    """
    Parse targets like: ["4,-20", "4,-10"] -> [[4.0, -20.0], [4.0, -10.0]]
    """
    out: list[list[float]] = []
    for t in tokens:
        parts = t.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid --targets item: '{t}'. Use 'MRL,MFE' like '4,-20'.")
        mrl = float(parts[0])
        mfe = float(parts[1])
        out.append([mrl, mfe])
    return out

def parse_index_value_pairs(items: list[str], value_name: str) -> tuple[list[int], list[str]]:
    """
    Parse ["2:AGC","8:GTG"] -> pos_list [2,8], value_list ["AGC","GTG"]
    Positions are interpreted as nucleotide positions (0..49) by your current utils.
    If you intend codon-index instead, you can convert pos*3 here.
    """
    pos_list: list[int] = []
    val_list: list[str] = []
    for it in items:
        if ":" not in it:
            raise ValueError(f"Invalid --{value_name} item: '{it}'. Use 'pos:VAL' like '2:AGC'.")
        p, v = it.split(":", 1)
        p_i = int(p)
        v = v.strip().upper()
        if value_name == 'codon':
            v = v.replace('U', 'T')
        pos_list.append(p_i)
        val_list.append(v)
    return pos_list, val_list



def build_diffusion(args, device):
    unet = UNet_CML(
        dim=200,
        channels=1,
        dim_mults=(1, 2, 4),
        resnet_block_groups=4,
        seq_len=50,
        dropout=0.2,
        num_label=2,  # MRL + MFE
    )

    diffusion = Diffusion_CML(
        model=unet,
        timestep=200,
        beta_last=0.01,
        uncondition_prop=0.2,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    diffusion.load_state_dict(ckpt["model"])
    diffusion = diffusion.to(device)
    diffusion.eval()
    return diffusion

def run_evaluator(
    fasta_path: str,
    eval_dir: str = 'evaluation',
    eval_script: str = 'evaluate.py',
    device: str = "cpu",
    model_path: str = 'Model/model.pt',
    batch_toks: int = 4096 * 8,
    seed: int = 1337,
    mfe_batch: int = 100,
) -> str:
    """
    Run evaluation/evaluate.py via subprocess.
    Assumes RNAfold is available in PATH.
    Returns the output csv path (same dir, .fasta -> .csv).
    """

    fasta_path = str(Path(fasta_path).resolve())
    out_csv = fasta_path.replace(".fasta", ".csv")

    # Default to the bundled evaluator under this repo
    eval_path = Path(eval_dir) / eval_script
    if not eval_path.exists():
        raise FileNotFoundError(f"[Eval] evaluate.py not found: {eval_path}")

    cmd = [
        sys.executable,
        eval_script,
        "--fasta", fasta_path,
        "--model", model_path,
        "--device", device,
        "--batch-toks", str(batch_toks),
        "--seed", str(seed),
        "--mfe-batch", str(mfe_batch),
    ]

    print("[Eval] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(eval_dir), check=True)

    print(f"[Eval] OK -> {out_csv}", flush=True)
    return out_csv

def design_utr(args):
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    targets = parse_targets(args.targets)
    diffusion = build_diffusion(args, device)

    # Build sampler
    if args.mode == "amino":
        if not args.amino:
            raise ValueError("Mode 'amino' requires --amino pos:AA ...")
        pos_list, amino_list = parse_index_value_pairs(args.amino, "amino")

        repaint = Repaint_Amino_CML(
            diffusion=diffusion,
            sample_bs=args.batch_size,
            seq_len=50,
            cond_weight=args.cond_weight,
            tgt_labels=targets,
            return_all=False,
        )
        gt, mask = build_gt_mask_from_aminos(amino_list=amino_list, pos_list=pos_list)
        result = repaint.p_resample(gt=gt.to(device), mask=mask.to(device), tgt_aminos=amino_list, pos_list=pos_list)

    elif args.mode == "codon":
        if not args.codon:
            raise ValueError("Mode 'codon' requires --codon pos:CODON ...")
        pos_list, codon_list = parse_index_value_pairs(args.codon, "codon")

        repaint = Repaint_Codon_CML(
            diffusion=diffusion,
            sample_bs=args.batch_size,
            seq_len=50,
            cond_weight=args.cond_weight,
            tgt_labels=targets,
            return_all=False,
        )

        gt, mask = bulid_gt_and_mask_from_codons(codon_list=codon_list, pos_list=pos_list)
        result = repaint.p_resample(gt=gt.to(device), mask=mask.to(device))
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(result, str(out_path), tgt_values=targets, batch_bs=args.batch_size)
    print(f"[OK] Saved: {out_path}")

    if args.do_eval:
        out_csv = run_evaluator(fasta_path=args.out, device=args.device)
        from src.plot.visualization import read_csv_and_plot
        read_csv_and_plot(out_csv, args)

if __name__ == "__main__":
    args = build_parser().parse_args()
    design_utr(args)