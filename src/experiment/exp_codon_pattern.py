import numpy as np
from itertools import product
from src.models.repaint.amino_codon_table import AMINO_TO_CODONS
import random

pattern_single_codon = {
    'name': 'single_codon',
    'codon': ['AGC'],
    'pos': [11],
}


pattern_double_codon = {
    'name': 'double_codon',
    'codon': ['TTG', 'ATG'],
    'pos': [5, 41],
}

pattern_triple_codon = {
    'name': 'triple_codon',
    'codon': ['AGG', 'GCG', 'TTA'],
    'pos': [14, 23, 35],
}

pattern_quadra_codon = {
    'name': 'quadra_codon',
    'codon': ['TGC', 'AAC', 'GGA', 'CTT'],
    'pos': [8, 17, 29, 38],
}

pattern_penta_codon = {
    'name': 'penta_codon',
    'codon': ['ATG', 'GCC', 'TAC', 'CAG', 'TTT'],
    'pos': [2, 20, 26, 32, 44],
}

pattern_half_sequence = {
    'name': 'half_sequence',
    'codon': ['CCAGCTTGGTGAACTCGGTGTTGGA'],
    'pos': [0]
}

pattern_codon_stripes = {
    'name': 'codon_stripes',
    'codon': ['AGC', 'GTG', 'TCG', 'TTG', 'TGT', 'GCC', 'CGT', 'CGA'],
    'pos': [2, 8, 14, 20, 26, 32, 38, 44],
}



Codon_Patterns = [
    pattern_single_codon,
    pattern_double_codon,
    pattern_triple_codon,
    pattern_quadra_codon,
    pattern_penta_codon
]
# Experiment: Refined codon experiment
pattern_1_codon = {
    'name': '1_codon',
    'codon': ['ATG', 'CGC'],
    'pos': [26, 38]
}

pattern_2_codon = {
    'name': '2_codon',
    'codon': ['ATG', 'TTG', 'CGT'],
    'pos': [26, 35, 41]
}

pattern_3_codon = {
    'name': '3_codon',
    'codon': ['ATG', 'GTG', 'CGA', 'CTT'],
    'pos': [26, 32, 38, 44]
}

pattern_4_codon = {
    'name': '4_codon',
    'codon': ['ATG', 'GGC', 'CTA', 'AAC', 'TTA'],
    'pos': [26, 29, 35, 41, 47]
}

Codon_Patterns_Refined = [
    pattern_1_codon,
    pattern_2_codon,
    pattern_3_codon,
    pattern_4_codon
]

def codon_pattern_generator(UTR_len:int=26, codon_num: int=1, pos:list[int] =[38]) -> list[dict]:
    codon_patterns = []
    codon_combineations = iter('ACGT', 3)
    for codon in codon_combineations:
        codon_pattern = {
            'name': f'{codon_num}_codon',
            'codon': codon,
            'pos': [UTR_len] + pos,
        }
        codon_patterns.append(codon_pattern)
    return codon_patterns

# Experiment: Fixed amino-acid with alternative codons

pattern_single_amino = {
    'name': 'single_amino',
    'amino': ['P'],
    'pos': [5]
}


pattern_double_amino = {
    'name': 'double_amino',
    'amino': ['R', 'L'],
    'pos': [8, 23]
}

pattern_triple_amino = {
    'name': 'triple_amino',
    'amino': ['K', 'S', 'V'],
    'pos': [14, 26, 41]
}

pattern_quadra_amino ={
    'name': 'quadra_amino',
    'amino': ['A', 'F', 'G', 'Y'],
    'pos': [10, 22, 35, 47]
}

pattern_penta_amino = {
    'name': 'penta_amino',
    'amino': ['R', 'T', 'L', 'H', 'V'],
    'pos':   [5, 14, 25, 36, 45]
}

pattern_dozen_amino = {
    'name': 'dozen_amino',
    'amino': ['M', 'F', 'D', 'A', 'W', 'E', 'Q', 'Y', 'G', 'N', 'H', 'T'],
    'pos':   [2,   5,   9,  12,  17,  22,  26,  29,  33,  37,  42,  47]
}



Amino_Patterns = [
    pattern_single_amino,
    pattern_double_amino,
    pattern_triple_amino,
    pattern_quadra_amino,
    pattern_penta_amino
]



Kozak = [
    {
        'name': 'Kozak_pattern_1',
        'codon': ['GCCACCAUGG'],
        'pos': [20],
    },
    {
        'name': 'Kozak_pattern_2',
        'codon': ['GCCGCCAUGG'],
        'pos': [20],
    }
]


def build_DRACH_at_pos(pos_list=[12]):
    """
    Build DRACH motif constraint patterns.

    DRACH definition:
        D = A/G/U
        R = A/G
        A = fixed
        C = fixed
        H = A/C/U

    Total combinations:
        3 * 2 * 3 = 18

    Parameters
    ----------
    pos_list : list[int]
        DRACH motif start positions.
        Example:
            [10, 15, 20]

    Returns
    -------
    list[dict]
        Example:
        {
            'name': 'DRACH_pattern_1',
            'codon': ['AAACA', 'AUG'],
            'pos': [20, 26],
        }
    """

    D_choices = ['A', 'G', 'U']
    R_choices = ['A', 'G']
    H_choices = ['A', 'C', 'U']

    patterns = []

    pattern_idx = 1

    for pos in pos_list:
        for D, R, H in product(D_choices, R_choices, H_choices):
            drach = f"{D}{R}AC{H}"
            pattern = {
                'name': f'DRACH_pattern_{pattern_idx}',
                'codon': [drach, 'AUG'],
                'pos': [pos, 26],
            }

            patterns.append(pattern)
            pattern_idx += 1

    return patterns


from itertools import product


def build_DRACH_at_pos_with_Kozak(drach_pos=12):
    """
    Build compositional motif patterns:
        Kozak + DRACH

    Kozak:
        GCCACCAUGG
        GCCGCCAUGG

    DRACH:
        D = A/G/U
        R = A/G
        H = A/C/U

    Total:
        2 * 18 = 36 patterns

    Parameters
    ----------
    drach_pos : int
        Start position of DRACH motif.
        Recommended:
            15

    Returns
    -------
    list[dict]
    """

    kozak_variants = [
        'GCCACCAUGG',
        'GCCGCCAUGG'
    ]

    D_choices = ['A', 'G', 'U']
    R_choices = ['A', 'G']
    H_choices = ['A', 'C', 'U']

    patterns = []
    pattern_idx = 1

    for kozak in kozak_variants:
        for D, R, H in product(D_choices, R_choices, H_choices):
            drach = f"{D}{R}AC{H}"
            pattern = {
                'name': f'Kozak_DRACH_pattern_{pattern_idx}',
                'codon': [drach, kozak],
                'pos': [drach_pos, 20],
            }

            patterns.append(pattern)
            pattern_idx += 1

    return patterns

def build_Kozak_DRACH_with_triplet():
    """
    Pattern:
        Kozak + DRACH + additional balanced 3nt fragment

    Total patterns:
        3 motif variants
        ×
        12 triplets
        =
        36 patterns

    Positions:
        triplet : 5
        DRACH  : 15
        Kozak  : 20
    """

    # 3 representative Kozak + DRACH variants
    motif_variants = [
        # AU-heavy
        {
            'variant_name': 'AU_heavy',
            'kozak': 'GCCACCAUGG',
            'drach': 'UAACU',
        },
        # GC-heavy
        {
            'variant_name': 'GC_heavy',
            'kozak': 'GCCGCCAUGG',
            'drach': 'GGACC',
        },
        # Balanced
        {
            'variant_name': 'balanced',
            'kozak': 'GCCACCAUGG',
            'drach': 'UGACC',
        },
    ]

    # 12 balanced/random-like triplets
    triplets = ['UCA', 'AGU', 'CAG', 'GUA', 'ACU', 'UGA', 'CAU', 'GUC', 'AUC', 'CUA', 'GAC', 'UGC',]

    patterns = []
    pattern_idx = 1

    for motif in motif_variants:
        for triplet in triplets:
            pattern = {
                'name': f"Kozak_DRACH_triplet_pattern_{pattern_idx}",
                'codon': [triplet, motif['drach'], motif['kozak']],
                'pos': [6, 12, 20,],
            }
            patterns.append(pattern)
            pattern_idx += 1

    return patterns





def build_amino_constraint_patterns(
        pos: list[int],
        num_amino_choices: list[int],
        amino_to_codons: dict = AMINO_TO_CODONS,
        exclude_amino=('M', 'W', '*'),
        start_codon_pos: int = None,
        repeat_pick: bool = False,
        seed: int = 42,
):
    """
    Build amino-acid constraint patterns.

    Example
    -------
    pos = [29, 35, 41, 47]
    num_amino_choices = [4, 4, 2, 2]

    -> randomly select 12 unique amino acids
    -> split into [4,4,2,2]
    -> generate all combinations
    """
    pattern_names = ['Single_amino', 'Double_amino', 'Triple_amino', 'Quadra_amino', 'Penta_amino', 'Hexa_amino', 'Hepta_amino', 'Octa_amino', 'Nona_amino', 'Deca_amino']
    pattern_name = pattern_names[len(num_amino_choices) - 1]
    random.seed(seed)

    amino_pool = [
        aa for aa in amino_to_codons.keys()
        if aa not in exclude_amino
    ]

    total_needed = sum(num_amino_choices)
    if repeat_pick:
        selected_amino = random.choices(amino_pool, k=total_needed)
    else:
        selected_amino = random.sample(amino_pool, total_needed)

    amino_groups = []

    start = 0

    for i, n in enumerate(num_amino_choices):
        selected_amino_per_pos = selected_amino[start:start+n]
        print(f'Amino choices for position {pos[i]}: {selected_amino_per_pos}')
        amino_groups.append(selected_amino_per_pos)
        start += n

    patterns = []

    pattern_idx = 1

    def recursive_build(current_amino, depth):

        nonlocal pattern_idx

        if depth == len(amino_groups):
            amino_with_start = current_amino.copy()
            pos_with_start = pos.copy()
            if start_codon_pos is not None:
                amino_with_start = ['M'] + current_amino
                pos_with_start = [start_codon_pos] + pos

            patterns.append({
                'name': f'{pattern_name}_pattern_{pattern_idx}',
                'amino': amino_with_start.copy(),
                'pos': pos_with_start.copy()
            })

            pattern_idx += 1
            return

        for aa in amino_groups[depth]:
            recursive_build(current_amino + [aa], depth + 1)

    recursive_build([], 0)

    return patterns


def build_random_quadra_amino_patterns(
    N,
    pos:list = [29, 35, 41, 47],
    amino_to_codons: dict = AMINO_TO_CODONS,
    exclude_amino=('M', 'W', '*'),
    seed=None,
    name_prefix="quadra_amino_pattern",
):
    """
    Randomly build N quadra-amino constraint patterns.

    Each pattern contains 4 fixed amino acids.
    Across all N patterns, amino acids are sampled without replacement,
    so the total N * 4 amino acids are non-repeated.

    Returns:
        patterns: list[dict]
        [
            {
                "name": "quadra_amino_pattern_1",
                "amino": ["A", "L", "K", "V"],
                "pos": [29, 35, 41, 47]
            },
            ...
        ]
    """

    amino_pool = [aa for aa in amino_to_codons.keys() if aa not in exclude_amino]

    if len(pos) != 4:
        raise ValueError(f"pos must contain 4 positions, but got {len(pos)}")

    required_num = N * 4

    if required_num > len(amino_pool):
        raise ValueError(
            f"N * 4 = {required_num} amino acids are required, "
            f"but amino_pool only contains {len(amino_pool)} amino acids. "
            f"Please use a larger amino_pool or smaller N."
        )

    rng = random.Random(seed)

    selected_aminos = rng.sample(amino_pool, required_num)

    patterns = []

    for i in range(N):
        pattern_aminos = selected_aminos[i * 4:(i + 1) * 4]

        patterns.append({
            "name": f"{name_prefix}_{i + 1}",
            "amino": pattern_aminos,
            "pos": pos.copy()
        })

    return patterns

def build_random_triple_amino_patterns(
    N=6,
    pos=[32, 38, 44],
    amino_to_codons=AMINO_TO_CODONS,
    exclude_amino=('M', 'W', '*'),
    start_codon_pos: int = None,
    seed=None,
    name_prefix="triple_amino_pattern",
):
    """
    Randomly build N triple-amino constraint patterns.

    Each pattern contains 3 fixed amino acids.

    Across all patterns, amino acids are sampled without replacement,
    so every multi-codon amino acid appears exactly once.

    Returns:
        [
            {
                "name": "triple_amino_pattern_1",
                "amino": ["L", "C", "G"],
                "pos": [32, 38, 44]
            },
            ...
        ]
    """

    amino_pool = [aa for aa in amino_to_codons.keys() if aa not in exclude_amino]

    if len(pos) != 3:
        raise ValueError(f"pos must contain 3 positions, but got {len(pos)}")

    required_num = N * 3

    rng = random.Random(seed)

    selected_aminos = rng.sample(amino_pool, required_num,)

    patterns = []

    for i in range(N):
        pattern_aminos = selected_aminos[i * 3:(i + 1) * 3]
        if start_codon_pos is not None:
            pattern_aminos = ['M'] + pattern_aminos
            pattern_pos = [start_codon_pos] + pos.copy()
        else:
            pattern_pos = pos.copy()
        patterns.append({
            "name": f"{name_prefix}_{i + 1}",
            "amino": pattern_aminos,
            "pos": pattern_pos,
        })

    return patterns