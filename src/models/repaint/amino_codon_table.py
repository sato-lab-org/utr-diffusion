AMINO_TO_CODONS = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],                  # Alanine           アラニン
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],    # Arginine          アルギニン
    'N': ['AAU', 'AAC'],                                # Asparagine        アスパラギン
    'D': ['GAU', 'GAC'],                                # Aspartic acid     アスパラギン酸
    'C': ['UGU', 'UGC'],                                # Cysteine          システイン
    'Q': ['CAA', 'CAG'],                                # Glutamine         グルタミン
    'E': ['GAA', 'GAG'],                                # Glutamic acid     グルタミン酸
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],                  # Glycine           グリシン
    'H': ['CAU', 'CAC'],                                # Histidine         ヒスチジン
    'I': ['AUU', 'AUC', 'AUA'],                         # Isoleucine        イソロイシン
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],    # Leucine           ロイシン
    'K': ['AAA', 'AAG'],                                # Lysine            リシン
    'M': ['AUG'],                                       # Methionine (START)メチオニン
    'F': ['UUU', 'UUC'],                                # Phenylalanine     フェニルアラニン
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],                  # Proline           プロリン
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],    # Serine
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],                  # Threonine         スレオニン
    'W': ['UGG'],                                       # Tryptophan        トリプトファン
    'Y': ['UAU', 'UAC'],                                # Tyrosine          チロシン
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],                  # Valine            バリン
    '*': ['UAA', 'UAG', 'UGA'],                         # Stop codons       終止コドン
}


CODON_TO_AMINO = {
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'AAU': 'N', 'AAC': 'N',
    'GAU': 'D', 'GAC': 'D',
    'UGU': 'C', 'UGC': 'C',
    'CAA': 'Q', 'CAG': 'Q',
    'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'CAU': 'H', 'CAC': 'H',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'AAA': 'K', 'AAG': 'K',
    'AUG': 'M',
    'UUU': 'F', 'UUC': 'F',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'UGG': 'W',
    'UAU': 'Y', 'UAC': 'Y',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'UAA': '*', 'UAG': '*', 'UGA': '*'
}


def rna_to_dna(seq: str) -> str:
    return seq.replace('U', 'T')


def dna_to_rna(seq: str) -> str:
    return seq.replace('T', 'U')


def get_codons_for_amino(amino: str) -> list:
    return [rna_to_dna(codon) for codon in AMINO_TO_CODONS[amino]]


def get_amino_for_codon(codon: str) -> str:
    return CODON_TO_AMINO[dna_to_rna(codon)]


AA_TO_CODON_USAGE_HUMAN_RNA = {
    'A': {'GCU': 0.27, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.10},
    'R': {'CGU': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.20, 'AGG': 0.21},
    'N': {'AAU': 0.47, 'AAC': 0.53},
    'D': {'GAU': 0.46, 'GAC': 0.54},
    'C': {'UGU': 0.46, 'UGC': 0.54},
    'Q': {'CAA': 0.27, 'CAG': 0.73},
    'E': {'GAA': 0.42, 'GAG': 0.58},
    'G': {'GGU': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25},
    'H': {'CAU': 0.42, 'CAC': 0.58},
    'I': {'AUU': 0.36, 'AUC': 0.47, 'AUA': 0.17},
    'L': {'UUA': 0.07, 'UUG': 0.13, 'CUU': 0.13, 'CUC': 0.20, 'CUA': 0.07, 'CUG': 0.40},
    'K': {'AAA': 0.43, 'AAG': 0.57},
    'M': {'AUG': 1.00},
    'F': {'UUU': 0.46, 'UUC': 0.54},
    'P': {'CCU': 0.29, 'CCC': 0.32, 'CCA': 0.28, 'CCG': 0.11},
    'S': {'UCU': 0.18, 'UCC': 0.22, 'UCA': 0.15, 'UCG': 0.06, 'AGU': 0.15, 'AGC': 0.24},
    'T': {'ACU': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12},
    'W': {'UGG': 1.00},
    'Y': {'UAU': 0.43, 'UAC': 0.57},
    'V': {'GUU': 0.18, 'GUC': 0.24, 'GUA': 0.12, 'GUG': 0.46},
    '*': {'UAA': 0.30, 'UAG': 0.24, 'UGA': 0.46},
}

def get_natural_CAI_for_amino(amino: str) -> float:
    codon_usage = AA_TO_CODON_USAGE_HUMAN_RNA[amino]
    max_usage = max(codon_usage.values())
    cai = sum(codon_usage[codon]/max_usage for codon in codon_usage)
    return cai