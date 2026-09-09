# UTR-Diffusion

Official implementation accompanying the manuscript:

> **UTR-Diffusion: Conditional Diffusion Modeling for Multi-objective and Constrained UTR Design**

UTR-Diffusion generates 50-nt 5′ UTR and 5′ UTR–CDS junction sequences with:

- continuous Mean Ribosome Load (MRL) and Minimum Free Energy (MFE) control;
- joint, partially observed multi-label conditioning;
- exact nucleotide and position-specific amino-acid constraints; and
- codon-relative-adaptiveness control with achieved sequence CAI reporting.

## Installation

```bash
git clone https://github.com/sato-lab-org/utr-diffusion.git
cd utr-diffusion
conda env create -f environment.yaml
conda activate utr-diffusion
```

The environment selects the CUDA 12.1 PyTorch build. Use an NVIDIA driver
compatible with CUDA 12.1; adjust `pytorch-cuda` only if your cluster requires a
different toolkit build.

The optional MFE evaluation pipeline also requires `RNAfold`, which is included
through the ViennaRNA dependency in `environment.yaml`.

## Final MCML checkpoint

The current release uses the masked continuous multi-label (MCML) model trained
by `src/scripts/train_mcml.py`. Its primary complete-label dataset is
`data/HEK293/real_MRL_pred_MFE_260k.csv`; training also consumes the corresponding
missing-MFE and missing-MRL subsets through the MCML data loader.

The training CSV files are not distributed in this repository. To rerun
training, obtain or prepare the study data separately and place these three
files under `data/HEK293/`:

- `real_MRL_pred_MFE_260k.csv`;
- `real_MRL_pred_MFE_260k_missing_MFE.csv`; and
- `real_MRL_pred_MFE_260k_missing_MRL.csv`.

Run the training and diagnostic sampling entry points as modules from the
repository root so the `src` package resolves consistently:

```bash
python -m src.scripts.train_mcml
python -m src.scripts.sample_mcml
```

The second command is a post-training diagnostic: it reads
`outputs/real_MRL_pred_MFE_260k_mcml/checkpoints/epoch_2000.pt`. The public
checkpoint is named `checkpoints/mcml_epoch_2000.pt` on Hugging Face so the MCML
architecture is identifiable without renaming the completed training run.

The released architecture is defined once in `src/models/mcml_config.py`:

- U-Net: 50-nt sequence, base and initial dimensions 200, multipliers
  `(1, 2, 4)`, learned sinusoidal dimension 18, and two per-label embeddings
  for `real_MRL` and `pred_MFE`;
- diffusion: 200 timesteps, final beta `0.01`, condition weight `4.0`, and
  unconditional proportion `0.2`; and
- checkpoint: epoch 2000, including both training and EMA model states.

The released RePaint sampler uses the standard one-step forward kernel
`sqrt(1-beta[t]) * x + sqrt(beta[t]) * noise` and applies the beta associated
with the destination timestep of each forward jump. This corrects two legacy
sampling-code errors without changing the checkpoint tensors or architecture.
For experiment reproducibility, the strategy historically named `euclidean`
continues to use summed element-wise absolute distance (L1); it is not a
squared Frobenius distance.

By default, `design_utr.py` first reuses a repository-local
`checkpoints/mcml_epoch_2000.pt` when one exists. Otherwise it downloads the
pinned release from Hugging Face into the standard user cache (normally
`~/.cache/huggingface/hub`). The Git repository never contains the 1.38 GB
checkpoint.

For an explicitly repository-local copy, run:

```bash
hf download \
  chuankai-dai/utr-diffusion-checkpoint \
  checkpoints/mcml_epoch_2000.pt \
  --local-dir .
```

Expected SHA-256:

```text
7125d9aa94ac67a71801c26364ce9df517a150a2e19b282caf317c835ae73ba0
```

`design_utr.py` loads the raw `model` state used by the manuscript Evaluation 3
and Benchmark 2 scripts. The checkpoint also retains `ema_model` for research
use, but the public design CLI does not expose an alternative weight selector.

## Design examples

MRL, MFE, and CAI are requested independently. Providing both MRL and MFE
performs joint multi-label conditioning, while providing only one uses the
checkpoint's masked single-label conditioning. Omitting both labels selects
the model's unconditional path. This can be used alone, with a sequence
constraint, or with CAI plus its required CDS constraint.

Unconditional generation:

```bash
python design_utr.py \
  --out design_outputs/unconditional_demo.fasta \
  --device cuda:0
```

Joint MRL/MFE conditioning:

```bash
python design_utr.py \
  --mrl 8.0 \
  --mfe -2.0 \
  --out design_outputs/mrl_mfe_demo.fasta \
  --device cuda:0
```

Single-label conditioning:

```bash
python design_utr.py \
  --mrl 8.0 \
  --out design_outputs/mrl_only_demo.fasta \
  --device cuda:0

python design_utr.py \
  --mfe -20.0 \
  --out design_outputs/mfe_only_demo.fasta \
  --device cuda:0
```

The three sequence-constraint options are mutually exclusive:

- `--nucleotide POSITION:SEQUENCE` fixes one or more nucleotide segments;
- `--amino POSITION:AA` fixes one or more amino acids at sparse, independent
  zero-based nucleotide starts; and
- `--cds-amino PEPTIDE` constrains a contiguous coding suffix that reaches the
  end of the 50-nt sequence.

### Nucleotide constraint

`--nucleotide` accepts arbitrary-length DNA or RNA strings, not only complete
codons. RNA `U` is normalized to DNA `T`.

```bash
python design_utr.py \
  --mrl 4.0 \
  --mfe -20.0 \
  --nucleotide 8:CGCTCA 32:UCA \
  --out design_outputs/nucleotide_demo.fasta \
  --device cuda:0
```

### Sparse amino-acid constraint

Positions given to `--amino` are zero-based nucleotide starts. They need not be
contiguous or share a reading frame.

```bash
python design_utr.py \
  --mrl 8.0 \
  --mfe -2.0 \
  --amino 26:M 31:D 37:L \
  --out design_outputs/amino_demo.fasta \
  --device cuda:0
```

### Contiguous CDS suffix and CAI control

`--cds-amino` takes the complete amino-acid sequence for a coding suffix,
including its initial methionine. For a peptide of length `N`, the nucleotide
start is determined automatically as `50 - 3N`, so the encoded peptide ends at
position 49. The peptide must begin with `M`; its first codon is fixed to `AUG`.
It must fit within the 50-nt sequence (`3N <= 50`), and CAI control requires at
least one downstream residue in addition to the initial methionine (`N >= 2`).
For example, an 8-aa peptide starts at position 26 and occupies codon starts
26, 29, 32, 35, 38, 41, 44, and 47.

Add `--cai` to request a codon relative-adaptiveness value for the synonymous
codons encoding this suffix:

```bash
python design_utr.py \
  --mrl 8.0 \
  --mfe -2.0 \
  --cai 0.90 \
  --cds-amino MGKVKVGV \
  --out design_outputs/cds_cai_090.fasta \
  --device cuda:0
```

`--cai` is the sampler's requested codon relative-adaptiveness target, not a
guarantee that every generated sequence will have exactly that CAI. The CLI
calculates and reports each sequence's achieved CAI after generation. The fixed
initiating `AUG` is excluded from that calculation, while every downstream
codon—including downstream methionine or tryptophan—contributes.

The manuscript Evaluation 3 layout is the 8-aa case above: 26 nt of UTR,
followed by `AUG` and seven downstream codons. Benchmark 2 uses a 10-aa peptide,
which automatically starts at position 20 and contains nine downstream CAI
codons. These are documented experimental configurations, not special CLI
modes or presets.

Each amino acid has its own feasible synonymous-codon adaptiveness range. If α
falls outside that range (methionine and tryptophan, for example, can only have
adaptiveness 1), the sampler clips that position to the nearest feasible value.
The CLI prints a warning and records every position's effective value in both
CAI CSV files instead of silently presenting the requested α as attainable.

Each CAI run writes:

- the requested FASTA file;
- `*_cai.csv`, with observed sequence-level CAI, effective per-position α, and
  peptide-preservation checks;
- `*_cai_summary.csv`, with peptide-valid CAI statistics and invalid counts;
  and
- `*_cai.jpg`, comparing achieved CAI with the specified α value.

The codon-usage weighting strength is fixed at `0.04` in
`src/models/mcml_config.py` for reproducibility.

## Optional MRL/MFE evaluation

Add `--do-eval` to any design command to run the bundled evaluator:

```bash
python design_utr.py \
  --mrl 8.0 \
  --mfe -2.0 \
  --cai 0.90 \
  --cds-amino MGKVKVGV \
  --out design_outputs/cds_cai_090.fasta \
  --do-eval \
  --device cuda:0
```

This additionally writes a CSV with predicted MRL/MFE values, an MRL–MFE
distribution plot, and a wrapped codon-diversity/entropy plot for the complete
peptide constraint.

## Main CLI options

| Option | Meaning |
|---|---|
| `--mrl` | Requested MRL label; usable alone or with MFE/CAI |
| `--mfe` | Requested MFE label; usable alone or with MRL/CAI |
| `--cai` | Requested codon relative-adaptiveness target; requires `--cds-amino` |
| `--checkpoint` | Optional local checkpoint override; otherwise use local release file or Hugging Face cache |
| `--nucleotide` | One or more arbitrary-length `position:SEQUENCE` constraints |
| `--amino` | One or more sparse `position:AA` constraints |
| `--cds-amino` | Complete contiguous suffix peptide, beginning with M |
| `--batch-size` | Number of sequences generated for the requested condition |
| `--do-eval` | Run bundled MRL/MFE evaluation and plots |
| `--device` | PyTorch device such as `cuda:0` or `cpu` |
| `--force` | Explicitly allow replacement of existing output files |

Run `python design_utr.py --help` for the complete interface.

Existing outputs are never replaced silently; pass `--force` only when
replacement is intentional. All default demo outputs are kept under
`design_outputs/`, separate from training and benchmark results under
`outputs/`.

## Licensing note

This repository currently does not declare a repository-wide license.
The RePaint-derived `src/models/repaint/scheduler.py` and
`src/models/repaint/utils.py` retain their upstream Huawei attribution and CC
BY-NC-SA 4.0 notices. Review those terms before redistribution or commercial
use; this README does not grant additional rights.
