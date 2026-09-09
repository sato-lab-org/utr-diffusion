import math
import unittest

from src.metrics.cai import calculate_cai, translate_codons


class CaiMetricTests(unittest.TestCase):
    def test_dna_and_rna_give_the_same_cai(self):
        dna = calculate_cai("TTACTG", [0, 3], ["L", "L"])
        rna = calculate_cai("UUACUG", [0, 3], ["L", "L"])

        self.assertAlmostEqual(dna, math.sqrt(0.07 / 0.40))
        self.assertAlmostEqual(dna, rna)

    def test_downstream_methionine_and_tryptophan_are_included(self):
        self.assertEqual(calculate_cai("ATGTGG", [0, 3], ["M", "W"]), 1.0)

    def test_selected_codons_translate_in_position_order(self):
        self.assertEqual(translate_codons("NNN".replace("N", "A") + "AAGGTT", [3, 6]), ["K", "V"])

    def test_rejects_wrong_peptide(self):
        with self.assertRaisesRegex(ValueError, "expected L"):
            calculate_cai("GTT", [0], ["L"])

    def test_rejects_stop_codons(self):
        with self.assertRaisesRegex(ValueError, "stop codon"):
            calculate_cai("TAA", [0])

    def test_rejects_overlapping_positions(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            calculate_cai("AAACCC", [0, 2])


if __name__ == "__main__":
    unittest.main()
