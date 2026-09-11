# EXP2 Generation Benchmark Protocol

**Protocol ID:** `exp2-cds50-generation`
**Version:** `0.2-draft`
**Role:** Authoritative protocol for candidate generation in all five independent method projects
**Companion document:** `README_EXP2_EVALUATION.md`
**Target-panel directory:** `exp2_30_peptide_candidates/`

> This document defines the common scientific task, candidate budget, and cross-project output contract.
> It does **not** define the canonical CAI or ViennaRNA implementation. Those are centralized in
> `README_EXP2_EVALUATION.md`.

---

## 1. Scientific question

EXP2 asks:

> Given exactly the same short peptide and the same synonymous-codon design space, which method produces candidate sequences with the best trade-off between human codon adaptation and predicted RNA folding stability while preserving the encoded peptide?

This is a **fair CDS-only, full-window proxy-CDS benchmark**.

No method is allowed to gain additional mutable 5′-UTR positions.

---

## 2. Methods

Primary methods:

- `utr_diffusion`
- `lineardesign`
- `dnachisel_cai_gc_proxy`
- `derna`
- `linearcdsfold`

Controls to support later:

- `random_synonymous`
- `original_cds`
- `best_codon_control`

### Method interpretation

- **UTR-Diffusion:** stochastic, target-conditioned candidate generator under amino-acid constraints.
- **LinearDesign:** direct MFE–CAI weighted optimizer controlled by `lambda`; it is not a target-CAI interface.
- **DNA Chisel:** CAI plus GC-composition proxy optimizer in this protocol. It does not directly optimize ViennaRNA MFE.
- **DERNA:** direct MFE–CAI weighted/Pareto optimizer.
- **LinearCDSfold:** direct MFE–CAI exact, beam, or Pareto optimizer.

Do not rename DNA Chisel as a direct CAI–MFE optimizer.

---

## 3. Benchmark target panel

The target panel contains 30 human peptide targets.

Required source files:

```text
exp2_30_human_peptide_candidates.csv
exp2_30_human_peptide_candidates.tsv
exp2_30_human_peptide_targets_17aa.fasta
exp2_30_human_downstream_peptides_16aa.fasta
```

Each record contains:

- `peptide_id`
- `gene`
- `uniprot_accession`
- `downstream_peptide_16aa`
- `optimizer_input_17aa`
- synonymous-space metadata

Definitions:

```text
downstream_peptide_16aa = 16 amino acids whose synonymous codons may vary
optimizer_input_17aa    = M + downstream_peptide_16aa
species                 = Homo sapiens
taxid                   = 9606
genetic code            = Standard nuclear genetic code
```

Do not change target IDs, peptide strings, ordering, or metadata.

No terminal stop codon is included in the benchmark fragment.

---

## 4. Common sequence layout

### 4.1 UTR-Diffusion model-space sequence

UTR-Diffusion generates exactly 50 RNA nucleotides:

```text
positions 1–2:   fixed UG
positions 3–50:  16 amino-acid-constrained codons = 48 nt
```

Therefore:

```text
UG + 16 codons = 50 nt
```

### 4.2 Reconstructed evaluation sequence

The central evaluator prepends one fixed `A`:

```text
A + generated_50nt
= AUG + 16 downstream codons
= 51 nt
```

The reconstructed sequence must translate to:

```text
M + downstream_peptide_16aa
```

### 4.3 Classical optimizer sequence

LinearDesign, DERNA, and LinearCDSfold receive:

```text
optimizer_input_17aa = M + downstream_peptide_16aa
```

Their candidate sequence is expected to be:

```text
AUG + 16 downstream codons = 51-nt RNA
```

DNA Chisel operates on the corresponding DNA fragment:

```text
ATG + 16 downstream codons = 51-nt DNA
```

### 4.4 Fairness rule

The only biological sequence freedom counted in the primary benchmark is the synonymous codon choice at the 16 downstream amino-acid positions.

Frozen requirements:

- initiator `AUG` is fixed
- no 5′-UTR bases are optimized
- no insertion or deletion
- no frame shift
- no amino-acid change
- no stop codon
- final RNA length is 51 nt
- downstream mutable region is 48 nt

---

## 5. Synonymous-space stratification

For peptide:

```text
p = a1 a2 ... a16
```

let `d(ai)` be the number of Standard-code synonymous codons for amino acid `ai`.

The theoretical synonymous sequence-space size is:

```text
|S(p)| = product_i d(ai)
```

The panel reports:

```text
log10_synonymous_space = log10(|S(p)|)
```

This is a compact computational descriptor of search-space size, not a standardized biological index.

The `low`, `medium`, and `high` strata are panel-balancing bins. They are not three biological classes and are not optimization targets.

---

## 6. Candidate-generation budget

For each peptide and each method:

```text
maximum raw candidate budget = 100
```

For 30 peptides:

```text
maximum = 3,000 raw candidates per method
```

### Counting rules

The benchmark does not require 100 unique sequences.

Every run must report:

- `attempted_count`
- `returned_count`
- `generation_success_count`
- `generation_failure_count`

The central evaluator later reports:

- `valid_count`
- `unique_count`
- `pareto_nondominated_count`

Never copy or repeat a candidate simply to reach 100.

If a method naturally returns fewer than 100 candidates, retain all returned candidates and report the smaller count.

If a method returns more than 100 candidates:

1. preserve the full raw output;
2. export at most 100 candidates for the count-matched primary analysis;
3. use a deterministic, pre-frozen down-selection rule;
4. record both the full and selected counts.

The exact method parameter grids are frozen only after the pilot.

---

## 7. Project separation

The five methods may remain in five independent project directories.

Each project is responsible only for:

1. reading the common target panel;
2. executing its own generation/optimization method;
3. preserving raw tool output and logs;
4. exporting candidates in the common handoff format;
5. recording complete provenance.

Each method project must **not** perform the final benchmark CAI, MFE, Pareto, statistical, or plotting analysis.

Those steps belong to the central evaluation project described in `README_EXP2_EVALUATION.md`.

---

## 8. Required export bundle

Each independent method project must create one self-contained export bundle:

```text
exp2_exports/
└── <protocol_version>/
    └── <run_id>/
        ├── candidates.tsv
        ├── attempts.tsv
        ├── run_manifest.json
        ├── parameter_sets.tsv
        ├── raw/
        ├── logs/
        └── checksums.sha256
```

Recommended final archive:

```text
<method>__<protocol_version>__<run_id>.tar.gz
```

Do not use absolute paths inside the TSV/JSON handoff files. Paths must be relative to the export-bundle root so the bundle remains portable.

### Required checks

Before export:

- every listed file exists;
- `checksums.sha256` verifies successfully;
- candidate IDs are unique within the bundle;
- attempt IDs are unique within `method × peptide_id`;
- all raw outputs referenced by TSV rows are retained;
- no final CAI/MFE ranking is embedded into generation logic.

---

## 9. `attempts.tsv` schema

One row per requested generation attempt or optimizer parameter evaluation.

Required columns:

```text
benchmark_protocol_id
benchmark_protocol_version
method
run_id
peptide_id
attempt_id
parameter_set_id
seed
method_mode
command_id
start_time_utc
end_time_utc
runtime_seconds
exit_code
attempt_status
failure_stage
failure_reason
stdout_path
stderr_path
raw_output_path
hostname
scheduler_job_id
```

Allowed `attempt_status` values:

```text
success
no_candidate
tool_error
timeout
parse_error
invalid_input
not_run
```

A failed attempt must still have a row.

---

## 10. `candidates.tsv` schema

One row per candidate returned by the method parser.

Required columns:

```text
benchmark_protocol_id
benchmark_protocol_version
method
method_version
method_commit
run_id
peptide_id
gene
uniprot_accession
downstream_peptide_16aa
optimizer_input_17aa
attempt_id
candidate_id
candidate_rank_within_attempt
parameter_set_id
seed
method_mode
method_parameters_json
raw_sequence
raw_alphabet
raw_layout
raw_output_path
generation_status
parser_status
parser_message
tool_reported_cai
tool_reported_mfe
tool_reported_structure
runtime_seconds
created_at_utc
```

### Allowed `raw_alphabet`

```text
RNA
DNA
```

### Allowed `raw_layout`

```text
rna50_ug_plus_16codons
rna51_aug_plus_16codons
dna51_atg_plus_16codons
```

Do not force all projects to reconstruct both 50- and 51-nt forms. The central evaluator performs canonical reconstruction.

### Diagnostic tool metrics

`tool_reported_cai`, `tool_reported_mfe`, and `tool_reported_structure` are optional diagnostics.

They must never be treated as the primary benchmark values.

Use an empty field when the tool does not report the metric.

---

## 11. `parameter_sets.tsv` schema

One row per parameter combination.

Required columns:

```text
parameter_set_id
method
method_mode
parameters_json
selection_role
pilot_or_full
notes
```

Examples of method-specific parameters:

- UTR-Diffusion: target MFE, target CAI proxy/conditioning setting, guidance scale, repaint parameters, seed.
- LinearDesign: lambda, beam/default mode, codon table path/hash.
- DNA Chisel: initial-sequence policy, CAI boost, GC target, GC boost, seed.
- DERNA: energy model, mode, lambda, sweep increment, Pareto thresholds.
- LinearCDSfold: exact/beam/Pareto mode, objective type, lambda, beam size, Pareto thresholds.

Do not hard-code scientific grids only inside Python or shell scripts. Store them in versioned configuration and export them here.

---

## 12. `run_manifest.json` requirements

The manifest must record:

```text
benchmark protocol ID/version
method
method version
method commit
repository URL
adapter version/commit
full command templates
target-panel file paths and SHA256
number of peptides
requested candidate budget
parameter-grid file and SHA256
base seed
seed-derivation rule
hostname
scheduler information
environment/modules/container
executable paths and SHA256
codon-table path and SHA256 used internally by the method
start/end UTC timestamps
attempt/return/failure counts
known method limitations
export-bundle relative paths
```

The seed derivation must use SHA256 and must not use Python's built-in `hash()`.

---

## 13. Raw-output preservation

Every method adapter must preserve:

- original tool stdout;
- original tool stderr;
- original tool output file;
- exact input file supplied to the tool;
- exact command line;
- parser log;
- scheduler log;
- method-generated auxiliary files required to understand the result.

A parser failure must not delete the raw output.

Temporary compute-scratch files may be deleted only after the export bundle is complete and checksummed.

---

## 14. Method-specific generation constraints

### 14.1 UTR-Diffusion

Required:

- raw output layout: `rna50_ug_plus_16codons`
- raw sequence begins with `UG`
- 16 amino-acid constraints applied to positions 3–50
- explicit seed per attempt
- explicit generation targets and guidance parameters
- preserve all generated samples, including duplicates

UTR-Diffusion must not prepend `A` inside the generation adapter unless the original 50-nt raw sequence is also retained.

### 14.2 LinearDesign

Required:

- input peptide: `optimizer_input_17aa`
- raw output layout: `rna51_aug_plus_16codons`
- explicit lambda per attempt
- explicit codon table
- preserve tool-reported MFE/CAI only as diagnostics
- do not represent lambda as target CAI
- retain duplicate outputs from different lambda values as separate attempts

### 14.3 DNA Chisel

Required:

- output layout: `dna51_atg_plus_16codons` or equivalent RNA conversion with original DNA retained
- `EnforceTranslation` over the complete 17-aa fragment
- fixed start codon
- no stop codon
- `MaximizeCAI`/`use_best_codon` plus a frozen GC objective
- explicit initial-sequence policy
- explicit GC target, boosts, and seed
- label method as `dnachisel_cai_gc_proxy`
- do not claim direct MFE optimization

### 14.4 DERNA

Required:

- input peptide: `optimizer_input_17aa`
- raw output layout: `rna51_aug_plus_16codons`
- explicit energy model and mode
- explicit lambda or Pareto/sweep settings
- preserve full sweep/Pareto raw output
- do not discard repeated sequences before export

### 14.5 LinearCDSfold

Required:

- input peptide: `optimizer_input_17aa`
- raw output layout: `rna51_aug_plus_16codons`
- explicit mode: exact, beam, or Pareto
- explicit objective: LD or DN
- explicit lambda/beam/Pareto settings
- explicit codon table
- preserve full result text/CSV

---

## 15. Pilot phase

Do not run the complete 30-peptide benchmark before pilot review.

Pilot panel:

```text
2 low-space peptides
2 medium-space peptides
2 high-space peptides
```

Pilot peptide IDs must be frozen in a common configuration file.

The pilot must establish:

- tool input compatibility;
- candidate parser correctness;
- practical runtime;
- useful parameter ranges;
- duplicate behavior;
- expected number of candidates;
- failure behavior;
- complete export-bundle compliance.

After pilot review, freeze:

- exact method parameter grids;
- base seed and seed derivation;
- maximum candidates and down-selection;
- method-specific time limits;
- full-run scheduler layout;
- adapter versions.

---

## 16. Adapter acceptance tests

Each method adapter must pass:

1. one low-space peptide;
2. one high-space peptide;
3. correct target-panel parsing;
4. deterministic rerun for deterministic mode;
5. reproducible seed behavior for stochastic mode;
6. raw-output preservation;
7. generation bundle schema validation;
8. checksum validation;
9. clean failure on malformed input;
10. successful execution outside the source-code directory;
11. no deletion of failed raw outputs;
12. no invocation of the central evaluator;
13. no complete benchmark launch during adapter development.

---

## 17. Scientific definitions frozen in this version

Frozen:

- 30 human peptide panel;
- 16 variable downstream amino acids;
- model-space layout `UG + 16 codons = 50 nt`;
- reconstructed layout `AUG + 16 codons = 51 nt`;
- fixed initiator AUG;
- Standard genetic code;
- no stop codon;
- CDS-only synonymous search space;
- maximum 100 raw candidates per peptide per method;
- no requirement for 100 unique candidates;
- five method labels;
- DNA Chisel interpreted as a CAI+GC proxy;
- independent method-project generation;
- central, method-agnostic evaluation;
- peptide as the primary paired unit.

Not yet frozen:

- exact generation parameter grids;
- initial DNA Chisel sequence policy;
- method-specific time limits;
- common codon-table path/hash;
- ViennaRNA version/settings;
- final Pareto and analysis thresholds.

The unfrozen evaluation items are governed by `README_EXP2_EVALUATION.md`.

---

## 18. Reporting language

Preferred description:

> Candidate generation was performed independently in method-specific projects under a shared 50-nt proxy-CDS protocol. UTR-Diffusion generated `UG` followed by 16 amino-acid-constrained codons, whereas classical optimizers received the corresponding 17-aa peptide beginning with methionine. Each method exported raw candidates and provenance without performing the final cross-method evaluation.

Required caveat:

> The benchmark is a short, full-window proxy-CDS transfer experiment and is not evidence that the current UTR-Diffusion model performs full-length native CDS design.

---

## 19. Protocol versioning

Any scientific change requires:

- incrementing this protocol version;
- documenting the change;
- updating checksums;
- regenerating export manifests;
- preventing result mixing across protocol versions.

Recommended immutable candidate key:

```text
benchmark_protocol_version
+ method
+ run_id
+ peptide_id
+ attempt_id
+ candidate_id
```

End of generation protocol.
