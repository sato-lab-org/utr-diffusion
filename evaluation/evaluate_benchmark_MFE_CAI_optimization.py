"""Evaluate ViennaRNA MFE for experiment-2 model outputs and a random baseline."""
import argparse
from pathlib import Path
import random
import pandas as pd
from Bio import SeqIO
from evaluation_pipeline import predict_mfe
from src.models.repaint.amino_codon_table import AA_TO_CODON_USAGE_HUMAN_RNA

LinearDesign_path = "/gs/bs/tga-satolab-gtex/dai/myscript/LinearDesign/output/exp2_stability_cai/runs/lineardesign_full_v1_20260720T050905Z/candidates.tsv"
DNAChicel_path = "/gs/bs/tga-satolab-gtex/dai/myscript/DnaChisel/output/exp2_stability_cai/runs/dnachisel_full_v1_20260720T061625Z/candidates.tsv"
DERNA_path = "/gs/bs/tga-satolab-gtex/dai/myscript/DERNA/output/exp2_stability_cai/runs/derna_strict100_v1_20260721T070945Z/candidates.tsv"
LinearCDSFold_path = '/gs/bs/tga-satolab-gtex/dai/myscript/LinearCDSfold/output/exp2_stability_cai/runs/linearcdsfold_full_v1_job8245689/candidates.tsv'
# Add LinearCDSFold here when its candidates file is ready.

PEPTIDE_CSV_PATH = (Path(__file__).resolve().parent.parent / "src/experiment/exp2_30_peptide_candidates/exp2_30_human_peptide_candidates.csv")
UTR_Diffusion_path = Path(__file__).resolve().parent.parent / "outputs/real_MRL_pred_MFE_260k_mcml/benchmark_MFE_CAI_optimization"


def evaluate_baseline(name, input_path, output_path, rnafold_path, batch_size):
    """Read a common candidates TSV, calculate MFE, and save essential columns."""
    source = pd.read_csv(input_path, sep="\t")
    required = {"peptide_id", "candidate_id", "raw_sequence"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")

    result = source[["peptide_id", "candidate_id", "raw_sequence"]].copy()
    result.insert(0, "model", name)
    result.rename(columns={"raw_sequence": "Sequence"}, inplace=True)
    result["Sequence"] = result["Sequence"].str.upper().str.replace("T", "U", regex=False)
    result = predict_mfe(result, rnafold_path=rnafold_path, batch_size=batch_size)
    result.to_csv(output_path, index=False)
    print(f"[SAVED] {len(result)} sequences -> {output_path}")


def evaluate_utr_diffusion(input_root, output_dir, rnafold_path, batch_size):
    """Combine all peptide FASTAs within each adaptiveness folder and calculate MFE."""
    for folder in sorted(Path(input_root).glob("*adaptiveness_*")):
        rows = []
        for fasta_path in sorted(folder.glob("*.fasta")):
            # additional function
            if "_shift_" in fasta_path.stem:
                peptide_id, shift_text = fasta_path.stem.rsplit("_shift_", 1)
                shift = int(shift_text)
            else:
                peptide_id = fasta_path.stem
                shift = 0
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                sequence = str(record.seq).upper().replace("T", "U")
                if shift == 0:
                    sequence = "A" + sequence
                rows.append(("UTR-Diffusion", folder.name, fasta_path.stem, peptide_id, shift, record.id, sequence))

        result = pd.DataFrame(rows, columns=["model", "setting", "ID", "peptide_id", "shift_aa", "candidate_id", "Sequence"])
        result.sort_values(by=["peptide_id", "shift_aa", "candidate_id"], inplace=True)
        result.reset_index(drop=True, inplace=True)

        result = predict_mfe(result, rnafold_path=rnafold_path, batch_size=batch_size)
        output_path = output_dir / f"utr_diffusion_{folder.name}_mfe.csv"
        result.to_csv(output_path, index=False)
        print(f"[SAVED] {len(result)} sequences -> {output_path}")


def evaluate_random(output_path, rnafold_path, batch_size, count=3000, seed=1337):
    """Generate reproducible 51-nt RNA sequences beginning with AUG and calculate MFE."""
    rng = random.Random(seed)
    rows = [("random", f"random_{i + 1:04d}", "AUG" + "".join(rng.choices("ACGU", k=48))) for i in range(count)]
    result = pd.DataFrame(rows, columns=["model", "candidate_id", "Sequence"])
    result = predict_mfe(result, rnafold_path=rnafold_path, batch_size=batch_size)
    result.to_csv(output_path, index=False)
    print(f"[SAVED] {len(result)} sequences -> {output_path}")


def evaluate_max_CAI_sequences(input_path, output_path, rnafold_path, batch_size, column_amino="downstream_peptide_16aa", column_id="candidate_id"):
    """
    Generate one maximum-CAI sequence for each 16-aa peptide.

    For each amino acid, select the codon with the highest human
    codon-usage frequency. The resulting sequence has the format:
        UG + 16 codons = 50 nt
    """
    source = pd.read_csv(input_path)
    rows = []

    for _, row in source.iterrows():
        peptide_id = row[column_id]
        peptide = row[column_amino]

        max_cai_cds = "".join(max(AA_TO_CODON_USAGE_HUMAN_RNA[amino], key=AA_TO_CODON_USAGE_HUMAN_RNA[amino].get,) for amino in peptide)
        sequence = "UG" + max_cai_cds
        rows.append(("Max-CAI",peptide_id, peptide, sequence,))

    result = pd.DataFrame(rows, columns=["model", "peptide_id", "peptide", "Sequence", ],)
    result = predict_mfe(result, rnafold_path=rnafold_path, batch_size=batch_size,)
    result.to_csv(output_path, index=False)

    print(f"[SAVED] {len(result)} maximum-CAI sequences -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch ViennaRNA MFE evaluation for experiment 2.")
    parser.add_argument("--output-dir", type=Path, default=UTR_Diffusion_path / "benchmark_evaluation")
    parser.add_argument("--rnafold-path", default="RNAfold")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=1337)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baselines = {
        "LinearDesign": LinearDesign_path,
        "DNAChisel": DNAChicel_path,
        "DERNA": DERNA_path,
        'LinearCDSFold': LinearCDSFold_path
    }
    for name, input_path in baselines.items():
        evaluate_baseline(name, input_path, args.output_dir / f"{name.lower()}_mfe.csv", args.rnafold_path, args.batch_size)

    evaluate_max_CAI_sequences(input_path=PEPTIDE_CSV_PATH, output_path=args.output_dir / "max_cai_mfe.csv", rnafold_path=args.rnafold_path, batch_size=args.batch_size,)
    evaluate_utr_diffusion(UTR_Diffusion_path, args.output_dir, args.rnafold_path, args.batch_size)
    evaluate_random(args.output_dir / "random_mfe.csv", args.rnafold_path, args.batch_size, seed=args.random_seed)


if __name__ == "__main__":
    main()
