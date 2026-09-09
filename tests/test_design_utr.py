import csv
import inspect
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import torch

    import design_utr
except ImportError as error:  # Keep static-only environments useful.
    np = None
    torch = None
    design_utr = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(design_utr is None, f"runtime dependencies unavailable: {IMPORT_ERROR}")
class DesignUtrTests(unittest.TestCase):
    def parse_and_validate(self, *tokens: str):
        args = design_utr.build_parser().parse_args(list(tokens))
        design_utr.validate_arguments(args)
        return args

    def test_public_help_uses_orthogonal_targets_and_constraints(self):
        help_text = design_utr.build_parser().format_help()

        for option in (
            "--mrl",
            "--mfe",
            "--cai",
            "--nucleotide",
            "--amino",
            "--cds-amino",
        ):
            self.assertIn(option, help_text)
        for removed_option in ("--mode", "--layout", "--preset", "--peptide"):
            self.assertNotIn(removed_option, help_text)

    def test_evaluator_helper_keeps_legacy_positional_signature(self):
        parameters = inspect.signature(design_utr.run_evaluator).parameters
        self.assertEqual(
            list(parameters),
            [
                "fasta_path",
                "eval_dir",
                "eval_script",
                "device",
                "model_path",
                "batch_toks",
                "seed",
                "mfe_batch",
            ],
        )
        self.assertEqual(parameters["eval_script"].default, "evaluate.py")
        self.assertEqual(parameters["device"].default, "cpu")
        self.assertEqual(parameters["model_path"].default, "Model/model.pt")

    def test_raw_model_weights_are_the_paper_reproduction_default(self):
        args = design_utr.build_parser().parse_args(
            ["--mrl", "4", "--nucleotide", "2:AGC"]
        )

        self.assertEqual(args.checkpoint_weights, "model")

    def test_auto_checkpoint_weights_fall_back_when_ema_is_empty(self):
        class FakeDiffusion:
            def __init__(self):
                self.loaded = None

            def load_state_dict(self, state, strict):
                self.loaded = state

            def to(self, device):
                return self

            def eval(self):
                return self

        diffusion = FakeDiffusion()
        args = Namespace(
            checkpoint="unused.pt",
            checkpoint_weights="auto",
            cond_weight=4.0,
        )
        raw_state = {"weight": object()}
        with (
            patch.object(design_utr, "build_mcml_diffusion", return_value=diffusion),
            patch.object(
                design_utr,
                "_load_checkpoint",
                return_value={"model": raw_state, "ema_model": None, "epoch": 1},
            ),
        ):
            result = design_utr.build_diffusion(args, "cpu")

        self.assertIs(result, diffusion)
        self.assertIs(diffusion.loaded, raw_state)

    def test_mrl_only_resolves_to_fixed_width_masked_target(self):
        args = self.parse_and_validate("--mrl", "8")

        targets = design_utr.resolve_targets(args)

        self.assertEqual(targets[0][0], 8.0)
        self.assertTrue(math.isnan(targets[0][1]))

    def test_mfe_only_resolves_to_fixed_width_masked_target(self):
        args = self.parse_and_validate("--mfe", "-20")

        targets = design_utr.resolve_targets(args)

        self.assertTrue(math.isnan(targets[0][0]))
        self.assertEqual(targets[0][1], -20.0)

    def test_joint_mrl_mfe_target_preserves_model_label_order(self):
        args = self.parse_and_validate("--mrl", "8", "--mfe", "-2")

        self.assertEqual(design_utr.resolve_targets(args), [[8.0, -2.0]])

    def test_cai_only_uses_both_labels_as_missing(self):
        args = self.parse_and_validate("--cai", "0.9", "--cds-amino", "MG")

        targets = design_utr.resolve_targets(args)

        self.assertTrue(math.isnan(targets[0][0]))
        self.assertTrue(math.isnan(targets[0][1]))

    def test_at_least_one_generation_target_is_required(self):
        args = design_utr.build_parser().parse_args([])

        with self.assertRaises(ValueError):
            design_utr.validate_arguments(args)

    def test_nucleotide_constraints_accept_arbitrary_length_sequences(self):
        positions, sequences = design_utr.parse_nucleotide_constraints(
            ["2:agcu", "10:TT"],
            require_codon=False,
        )

        self.assertEqual(positions, [2, 10])
        self.assertEqual(sequences, ["AGCT", "TT"])

    def test_nucleotide_constraints_reject_overlap_and_terminal_overflow(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            design_utr.parse_nucleotide_constraints(
                ["2:ACGT", "5:TT"],
                require_codon=False,
            )
        with self.assertRaisesRegex(ValueError, "0-49|within|50|bounds|fit"):
            design_utr.parse_nucleotide_constraints(
                ["48:AAA"],
                require_codon=False,
            )

    def test_nucleotide_constraints_reject_non_nucleotides(self):
        with self.assertRaisesRegex(ValueError, "nucleotide|ACGT"):
            design_utr.parse_nucleotide_constraints(
                ["2:ANR"],
                require_codon=False,
            )

    def test_cds_amino_fills_the_suffix_and_excludes_initial_aug_from_cai(self):
        constraint = design_utr.cds_amino_constraints("mGkV", require_cai=True)

        self.assertEqual(constraint.peptide, "MGKV")
        self.assertEqual(constraint.start, 38)
        self.assertEqual(constraint.codon_positions, (38, 41, 44, 47))
        self.assertEqual(constraint.cai_positions, (41, 44, 47))

    def test_cds_amino_requires_initial_methionine(self):
        with self.assertRaisesRegex(ValueError, "begin with M|start with M"):
            design_utr.cds_amino_constraints("AG", require_cai=False)

    def test_cai_requires_at_least_one_downstream_codon(self):
        with self.assertRaisesRegex(ValueError, "at least 2|downstream"):
            design_utr.cds_amino_constraints("M", require_cai=True)

    def test_cds_amino_rejects_a_peptide_that_cannot_fit_in_50_nt(self):
        with self.assertRaisesRegex(ValueError, "50|too long|fit"):
            design_utr.cds_amino_constraints("M" + "A" * 16, require_cai=False)

    def test_cai_requires_cds_amino_instead_of_sparse_amino(self):
        missing_cds = design_utr.build_parser().parse_args(["--mrl", "8", "--cai", "0.9"])
        sparse_amino = design_utr.build_parser().parse_args(
            ["--mrl", "8", "--cai", "0.9", "--amino", "26:M", "29:G"]
        )

        with self.assertRaisesRegex(ValueError, "--cds-amino"):
            design_utr.validate_arguments(missing_cds)
        with self.assertRaisesRegex(ValueError, "--cds-amino|mutually exclusive"):
            design_utr.validate_arguments(sparse_amino)

    def test_sequence_constraint_families_are_mutually_exclusive(self):
        combinations = (
            ("--nucleotide", "2:AGC", "--amino", "8:M"),
            ("--nucleotide", "2:AGC", "--cds-amino", "MG"),
            ("--amino", "8:M", "--cds-amino", "MG"),
        )
        for constraint_tokens in combinations:
            with self.subTest(constraint_tokens=constraint_tokens):
                with self.assertRaises(SystemExit):
                    design_utr.build_parser().parse_args(
                        ["--mrl", "8", *constraint_tokens]
                    )

    def test_no_sequence_constraint_calls_diffusion_directly(self):
        class FakeDiffusion:
            device = torch.device("cpu")

            def __init__(self):
                self.calls = []

            def sample(self, classes, shape, cond_weight, output_all_steps=False):
                self.calls.append(
                    {
                        "classes": classes.detach().cpu(),
                        "shape": shape,
                        "cond_weight": cond_weight,
                        "output_all_steps": output_all_steps,
                    }
                )
                return torch.zeros(shape)

        args = self.parse_and_validate("--mrl", "8", "--batch-size", "2")
        targets = design_utr.resolve_targets(args)
        diffusion = FakeDiffusion()

        with (
            patch.object(
                design_utr,
                "Repaint_Codon_MCML",
                side_effect=AssertionError("nucleotide RePaint should not be constructed"),
            ),
            patch.object(
                design_utr,
                "Repaint_Amino_MCML",
                side_effect=AssertionError("amino RePaint should not be constructed"),
            ),
        ):
            samples = design_utr.generate_samples(args, diffusion, targets)

        self.assertEqual(tuple(samples.shape), (2, 1, 4, 50))
        self.assertEqual(len(diffusion.calls), 1)
        call = diffusion.calls[0]
        self.assertEqual(tuple(call["classes"].shape), (2, 2))
        self.assertTrue(torch.equal(call["classes"][:, 0], torch.full((2,), 8.0)))
        self.assertTrue(torch.isnan(call["classes"][:, 1]).all())
        self.assertEqual(call["shape"], (2, 1, 4, 50))

    def test_output_must_have_a_fasta_suffix(self):
        args = design_utr.build_parser().parse_args(
            ["--mrl", "4", "--nucleotide", "2:AGC", "--out", "outputs/would_overwrite.csv"]
        )

        with self.assertRaisesRegex(ValueError, "must end"):
            design_utr.validate_arguments(args)

        args.out = "outputs/valid.FASTA"
        design_utr.validate_arguments(args)

    def test_duplicate_batch_target_pairs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            design_utr.parse_targets(["4,-20", "4.0,-20.0"])

    def test_cai_metadata_reports_position_specific_feasible_alpha(self):
        records = [
            {
                "ID": "sequence_0",
                "Sequence": "A" * 26 + "ATG" * 8,
                "target_MRL": 8.0,
                "target_MFE": -2.0,
                "sample_index": 0,
            }
        ]
        args = Namespace(cds_amino="MMMMMMMM", cai=0.5)

        measured, constraint = design_utr.add_cai_measurements(records, args)

        self.assertEqual(constraint.start, 26)
        self.assertEqual(measured[0]["CAI_codon_positions_0based"], "29,32,35,38,41,44,47")
        self.assertEqual(measured[0]["n_effective_adaptiveness_clipped"], 7)
        self.assertAlmostEqual(
            measured[0]["effective_adaptiveness_geomean_reference"], 1.0
        )

    def test_summary_cai_statistics_exclude_invalid_peptide_constraints(self):
        records = [
            {
                "CAI": 0.8,
                "peptide_valid": True,
                "requested_adaptiveness": 0.9,
                "effective_adaptiveness_by_codon": "29:G=0.900000",
                "effective_adaptiveness_geomean_reference": 0.9,
                "n_effective_adaptiveness_clipped": 0,
            },
            {
                "CAI": 0.1,
                "peptide_valid": False,
                "requested_adaptiveness": 0.9,
                "effective_adaptiveness_by_codon": "29:G=0.900000",
                "effective_adaptiveness_geomean_reference": 0.9,
                "n_effective_adaptiveness_clipped": 0,
            },
        ]

        summary = design_utr._summary_row(records, "overall", "all", "all")

        self.assertEqual(summary["n_with_observed_CAI"], 2)
        self.assertEqual(summary["n_with_CAI"], 1)
        self.assertAlmostEqual(summary["CAI_mean"], 0.8)

    def test_sparse_amino_constraints_reject_overlaps_and_terminal_overflow(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            design_utr.parse_index_value_pairs(["2:M", "4:F"], "amino")
        with self.assertRaisesRegex(ValueError, "between 0 and 47|bounds"):
            design_utr.parse_index_value_pairs(["48:M"], "amino")

    def test_decode_samples_preserves_masked_targets_and_target_major_order(self):
        samples = np.full((3, 1, 4, 50), -1.0, dtype=np.float32)
        samples[0, 0, 0, :] = 1.0
        samples[1, 0, 1, :] = 1.0
        samples[2, 0, 3, :] = 1.0
        targets = [[8.0, float("nan")], [float("nan"), -2.0], [float("nan"), float("nan")]]

        records = design_utr.decode_samples(samples, targets, 1)

        self.assertEqual(records[0]["Sequence"], "A" * 50)
        self.assertEqual(records[1]["Sequence"], "C" * 50)
        self.assertEqual(records[2]["Sequence"], "T" * 50)
        self.assertEqual(records[0]["target_MRL"], 8.0)
        self.assertTrue(math.isnan(records[0]["target_MFE"]))
        self.assertTrue(math.isnan(records[1]["target_MRL"]))
        self.assertEqual(records[1]["target_MFE"], -2.0)

    def test_cai_only_fasta_id_describes_the_requested_cai(self):
        samples = np.zeros((1, 1, 4, 50), dtype=np.float32)

        records = design_utr.decode_samples(
            samples,
            [[float("nan"), float("nan")]],
            1,
            cai=0.9,
        )

        self.assertIn("cai_0.9", records[0]["ID"])
        self.assertNotIn("unconditioned", records[0]["ID"])

    def test_existing_outputs_require_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "design.fasta"
            output.write_text("existing", encoding="utf-8")
            args = self.parse_and_validate("--mrl", "8", "--out", str(output))

            with self.assertRaisesRegex(FileExistsError, "--force"):
                design_utr.ensure_outputs_available(args)

            args.force = True
            design_utr.ensure_outputs_available(args)

    def test_decoded_nucleotide_constraints_are_verified(self):
        args = self.parse_and_validate(
            "--mrl", "8", "--nucleotide", "2:AGCU", "10:TT"
        )
        valid = [{"ID": "valid", "Sequence": "AAAGCTAAAATT" + "A" * 38}]
        invalid = [{"ID": "invalid", "Sequence": "A" * 50}]

        design_utr.verify_generated_constraints(valid, args)
        with self.assertRaisesRegex(RuntimeError, "not preserved"):
            design_utr.verify_generated_constraints(invalid, args)

    def test_cai_summary_groups_targets_even_when_one_or_both_labels_are_missing(self):
        cases = (
            ("mrl_only", ("--mrl", "8"), 8.0, float("nan")),
            ("mfe_only", ("--mfe", "-2"), float("nan"), -2.0),
            ("cai_only", (), float("nan"), float("nan")),
        )
        constraint = design_utr.cds_amino_constraints("MG", require_cai=True)
        for name, target_tokens, target_mrl, target_mfe in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                args = self.parse_and_validate(
                    *target_tokens,
                    "--cai",
                    "0.9",
                    "--cds-amino",
                    "MG",
                )
                record = {
                    "ID": "sequence_0",
                    "Sequence": "A" * 44 + "ATGGGT",
                    "target_MRL": target_mrl,
                    "target_MFE": target_mfe,
                    "sample_index": 0,
                    "requested_adaptiveness": 0.9,
                    "effective_adaptiveness_by_codon": "47:G=0.900000",
                    "effective_adaptiveness_geomean_reference": 0.9,
                    "n_effective_adaptiveness_clipped": 0,
                    "CAI": 0.8,
                    "peptide_valid": True,
                }
                fasta_path = Path(temp_dir) / "design.fasta"
                with (
                    patch.object(
                        design_utr,
                        "add_cai_measurements",
                        return_value=([record], constraint),
                    ),
                    patch("src.plot.visualization.plot_cai_response"),
                ):
                    _, summary_path, _ = design_utr.write_cai_outputs(
                        [record], args, fasta_path
                    )

                with summary_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[1]["n_sequences"], "1")


if __name__ == "__main__":
    unittest.main()
