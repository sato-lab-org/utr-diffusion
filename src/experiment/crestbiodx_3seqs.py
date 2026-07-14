from pathlib import Path
from src.models.repaint.amino_codon_table import CODON_TO_AMINO ,dna_to_rna


TARGET_SEQUENCES=[
    {
        'name': 'Ref_1_LinearDesign',
        'seq': 'acatttgcttctgacacaactgtgttcactagcaacctcaaacagacacc'
               'ATGGTCTTCACCCTGGAGGACTTCGTGGGGGATTGGCGGCAGACCGCCGGCTACAACCTCGACCAGGTTCTGGAGCAGGGAGGAGTGTCCTCCCTGTTCCAGAACCTGGGCGTGAGCGTCACCCCTATCCAGCGGATCGTGCTGTCTGGGGAGAACGGCCTG'
               'AAGATCGATATTCATGTTATCATTCCATACGAGGGCCTGTCCGGTGATCAGATGGGCCAGATCGAGAAGATCTTCAAGGTGGTCTACCCCGTCGATGACCACCACTTCAAGGTGATCCTGCACTACGGGACACTGGTCATCGACGGGGTGACCCCTAACATG'
               'ATCGACTACTTTGGCCGGCCCTACGAGGGCATTGCCGTGTTCGACGGCAAGAAGATCACCGTGACAGGCACTCTGTGGAATGGTAACAAGATTATCGATGAGAGGCTGATCAACCCAGACGGCAGTCTGCTGTTTAGGGTGACCATCAACGGGGTGACCGGC'
               'TGGCGGCTCTGCGAGCGGATCCTCGCATGA'
               'gctcgctttcttgctgtccaatttctattaaaggttcctttgttccctaagtccaactactaaactgggggatattatgaagggccttgagcatctggattctgcctaataaaaaacatttattttcattgcaaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    }
    ,
    {
        'name': 'Ref_2_CAI',
        'seq': 'acatttgcttctgacacaactgtgttcactagcaacctcaaacagacacc'
               'ATGGTATTCACTTTAGAAGACTTTGTAGGTGATTGGCGCCAAACGGCGGGTTATAATCTAGATCAAGTATTGGAACAAGGTGGTGTGAGCAGCCTGTTCCAGAACCTGGGCG'
               'TGAGCGTGACCCCCATCCAGAGAATCGTGCTGAGCGGCGAGAACGGCCTGAAGATCGACATCCACGTGATCATCCCCTACGAGGGCCTGAGCGGCGACCAGATGGGCCAGATCGAGAAGATCTTCAAGGTGGTGTACCCCGTGGACGACCACCACTTCAAGG'
               'TGATCCTGCACTACGGCACCCTGGTGATCGACGGCGTGACCCCCAACATGATCGACTACTTCGGCAGACCCTACGAGGGCATCGCCGTGTTCGACGGCAAGAAGATCACCGTGACCGGCACCCTGTGGAACGGCAACAAGATCATCGACGAGAGACTGATCA'
               'ATCCCGACGGCAGCCTGCTGTTCCGGGTGACCATCAATGGCGTGACCGGCTGGAGGCTGTGTGAGCGCATTCTGGCATAAgctcgctttcttgctgtccaatttctattaaaggttcctttgttccctaagtccaactactaaactgggggatattatgaag'
               'ggccttgagcatctggattctgcctaataaaaaacatttattttcattgcaaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
    }
    ,
    {
        'name': 'Ref_3_GEMORNA',
        'seq': 'agggttcgaagttacttcttctcctgctgtaaaagagccacc'
               'ATGGTCTTCACGCTGGAAGACTTCGTGGGCGACTGGAGGCAGACAGCCGGCTACAATCTGGATCAGGTGCTGGAGCAGGGCGGGGTCAGCTCCCTGTTCCAGAACCTGGGGGTGAGCGTC'
               'ACCCCCATCCAGCGCATCGTGCTGAGTGGGGAGAACGGCCTCAAGATCGACATCCACGTGATCATCCCGTACGAGGGCCTGTCCGGCGACCAGATGGGCCAGATCGAGAAGATCTTCAAGGTGGTCTACCCGGTGGACGACCACCACTTCAAGGTCATCCTG'
               'CACTACGGCACCCTGGTGATCGACGGCGTCACCCCGAACATGATCGACTACTTCGGCCGCCCGTACGAGGGCATCGCCGTGTTCGACGGCAAGAAGATCACCGTGACCGGCACCCTGTGGAACGGCAACAAGATCATCGACGAGCGCCTGATCAACCCGGAC'
               'GGCAGCCTCCTCTTCCGGGTGACCATCAACGGCGTCACCGGCTGGCGCCTCTGCGAGCGCATCCTGGCGTAAtaactaactcaccaggcccaggtgcacatacctgactcacccgccagAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    }
]
from src.models.repaint.amino_codon_table import CODON_TO_AMINO, dna_to_rna


def cut_to_50nt_atg_pos(records, atg_pos=26, window_size=50):
    """
    Cut each reference sequence into a 50nt sequence,
    where CDS-start ATG is placed at atg_pos.

    Default:
        ATG starts at position 26
        ATG occupies positions [26, 27, 28]

    Input:
        [
            {'name': 'Ref_1_LinearDesign', 'seq': '...'},
            ...
        ]

    Return:
        [
            {'name': 'Ref_1_LinearDesign', 'seq': '50nt_seq'},
            ...
        ]
    """
    output = []

    for record in records:
        name = record["name"]
        seq = record["seq"]

        # Terai-san's sequence uses uppercase for CDS.
        # Therefore the first uppercase "ATG" should be the CDS-start ATG.
        atg_idx = seq.find("ATG")

        if atg_idx == -1:
            raise ValueError(f"{name}: uppercase CDS-start ATG was not found.")

        start = atg_idx - atg_pos
        end = start + window_size

        if start < 0:
            raise ValueError(
                f"{name}: not enough upstream sequence. "
                f"ATG index={atg_idx}, requested atg_pos={atg_pos}."
            )

        if end > len(seq):
            raise ValueError(
                f"{name}: not enough downstream sequence. "
                f"end={end}, sequence length={len(seq)}."
            )

        seq_50nt = seq[start:end].upper()

        if len(seq_50nt) != window_size:
            raise ValueError(
                f"{name}: cut length is {len(seq_50nt)}, expected {window_size}."
            )

        if seq_50nt[atg_pos:atg_pos + 3] != "ATG":
            raise ValueError(
                f"{name}: ATG is not at positions "
                f"[{atg_pos}, {atg_pos + 1}, {atg_pos + 2}]. "
                f"Found {seq_50nt[atg_pos:atg_pos + 3]}."
            )

        output.append({
            "name": name,
            "seq": seq_50nt,
        })

    return output


def make_amino_patterns(records_50nt, atg_pos=26):
    """
    Make amino-acid patterns from 50nt sequences.

    We fix:
        ATG itself + 21nt after ATG

    That means:
        8 codons = M + 7 amino acids

    Positions:
        [26, 29, 32, 35, 38, 41, 44, 47]

    Return:
        [
            {
                'name': 'Ref_1_LinearDesign_amino',
                'amino': ['M', 'V', 'F', 'T', 'L', 'E', 'D', 'F'],
                'pos':   [26, 29, 32, 35, 38, 41, 44, 47]
            },
            ...
        ]
    """
    patterns = []

    for record in records_50nt:
        name = record["name"]
        seq = record["seq"].upper()

        if len(seq) != 50:
            raise ValueError(f"{name}: sequence length is {len(seq)}, expected 50.")

        if seq[atg_pos:atg_pos + 3] != "ATG":
            raise ValueError(
                f"{name}: ATG is not at positions "
                f"[{atg_pos}, {atg_pos + 1}, {atg_pos + 2}]. "
                f"Found {seq[atg_pos:atg_pos + 3]}."
            )

        # ATG + downstream 21nt = 24nt = 8 codons
        coding_part = seq[atg_pos:atg_pos + 24]

        amino_list = []
        pos_list = []

        for i in range(0, len(coding_part), 3):
            codon_dna = coding_part[i:i + 3]
            codon_rna = dna_to_rna(codon_dna)
            amino = CODON_TO_AMINO[codon_rna]

            amino_list.append(amino)
            pos_list.append(atg_pos + i)

        patterns.append({
            "name": f"{name}_amino",
            "amino": amino_list,
            "pos": pos_list,
        })

    return patterns


def get_CREST_amino_patterns():
    records_50nt = cut_to_50nt_atg_pos(TARGET_SEQUENCES, atg_pos=26, window_size=50)
    amino_patterns = make_amino_patterns(records_50nt, atg_pos=26)
    for pattern in amino_patterns:
        print(f"Name: {pattern['name']}")
        print(f"Amino acids: {pattern['amino']}")
        print(f"Positions: {pattern['pos']}")
        print()
    return amino_patterns
