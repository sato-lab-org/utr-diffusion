import torch
import pandas as pd
from itertools import product

def get_targets_from_data_with_HandL_values(data, label_names, H_val=0.95, L_val=0.05):
    keys = ['Train_label', 'Valid_label'] if 'Test_label' not in data.keys() else ['Train_label', 'Valid_label', 'Test_label']
    all_labels = []
    for key in keys:
        all_labels.append(data[key])
    all_labels = torch.cat(all_labels, dim=0)
    label_df = pd.DataFrame(all_labels, columns=label_names)

    label_ranges = {
        label_col: {
            'L': label_df[label_col].quantile(L_val),
            'H': label_df[label_col].quantile(H_val)
        }
        for label_col in label_names
    }

    all_modes = list(product(['H', 'L'], repeat=len(label_names)))
    targets = []
    for mode in all_modes:
        target = [label_ranges[col][hl] for col, hl in zip(label_names, mode)]
        targets.append(target)

    print("\nAll target combinations:")
    for mode, target in zip(all_modes, targets):
        mode_str = ", ".join([f"{col}={hl}" for col, hl in zip(label_names, mode)])
        target_str = ", ".join([f"{col}={val:.4f}" for col, val in zip(label_names, target)])
        print(f"{mode_str}  -->  [{target_str}]")

    return targets


def get_targets_from_data_with_sigma_grid(
    data,
    label_names,
    sigma_levels=(-2, -1, 0, 1, 2),
    coupled_value_groups=None,
):
    """
    Generate target combinations from mean +/- sigma values.

    Args:
        data: dict with Train_label, Valid_label, optionally Test_label
        label_names: list[str]
        sigma_levels: sigma multipliers, e.g. (-2, -1, 0, 1, 2)
        coupled_value_groups:
            list of label-name groups that share the same sigma level.
            Example:
                [
                    ["polysome.mean.load", "clean.lin.prot.mean"]
                ]

    Returns:
        targets: list[list[float]]
    """

    if coupled_value_groups is None:
        coupled_value_groups = []

    keys = ['Train_label', 'Valid_label'] if 'Test_label' not in data.keys() else [
        'Train_label', 'Valid_label', 'Test_label'
    ]

    all_labels = torch.cat([data[key] for key in keys], dim=0)

    if isinstance(all_labels, torch.Tensor):
        all_labels = all_labels.detach().cpu().numpy()

    label_df = pd.DataFrame(all_labels, columns=label_names)

    label_stats = {}
    for label_col in label_names:
        mean_val = label_df[label_col].mean()
        std_val = label_df[label_col].std()

        label_stats[label_col] = {
            sigma: mean_val + sigma * std_val
            for sigma in sigma_levels
        }

    grouped_labels = set()
    condition_units = []

    for group in coupled_value_groups:
        for label in group:
            if label not in label_names:
                raise ValueError(f"Label {label} in coupled_value_groups is not in label_names.")
        condition_units.append(tuple(group))
        grouped_labels.update(group)

    for label in label_names:
        if label not in grouped_labels:
            condition_units.append((label,))

    all_modes = list(product(sigma_levels, repeat=len(condition_units)))

    targets = []

    print("\nAll sigma-based target combinations:")
    print(f"Labels: {label_names}")
    print(f"Shared sigma groups: {coupled_value_groups}")
    print(f"Sigma levels: {sigma_levels}")
    print(f"Effective condition dimensions: {len(condition_units)}")
    print(f"Total target combinations: {len(all_modes)}")

    for mode in all_modes:
        sigma_map = {}

        for unit, sigma in zip(condition_units, mode):
            for label in unit:
                sigma_map[label] = sigma

        target = [
            label_stats[label][sigma_map[label]]
            for label in label_names
        ]
        target = [0.0 if abs(round(val, 4))< 1e-8 else round(val, 4) for val in target]
        targets.append(target)

        mode_str = ", ".join([f"{label}={sigma_map[label]:+g}σ" for label in label_names])
        target_str = ", ".join([f"{label}={val:.4f}" for label, val in zip(label_names, target)])
        print(f"{mode_str}  -->  [{target_str}]")

    return targets


from itertools import combinations
import torch
import pandas as pd


def get_targets_extrapolation(
    data,
    label_names,
    controlled_labels=(1, 2, 3),
    other_label=0.0,
    target_levels=(-4.5, 4.5),
    coupled_value_groups=None,
    deduplicate_coupled_groups=True,
    verbose=True,
):
    """
    Generate extrapolation target combinations.

    Args:
        data:
            Kept for API consistency. Not used in this function.
        label_names: list[str]
            Label names in target order.
        controlled_labels: tuple[int]
            Number of controlled condition units to enumerate.
            Example: (1, 2, 3)
        other_label: float
            Value assigned to uncontrolled labels.
        target_levels: tuple[float]
            Extrapolation target values, e.g. (-4.5, 4.5).
        coupled_value_groups: list[list[str]]
            Labels that should be controlled together.
            Example:
                [["polysome.mean.load", "clean.lin.prot.mean"]]
        deduplicate_coupled_groups: bool
            If True, labels in the same coupled group are treated as one
            condition unit and selected only once.
        round_digits: int
            Number of digits for rounding.
        verbose: bool
            Whether to print generated targets.

    Returns:
        targets: list[list[float]]
    """

    if coupled_value_groups is None:
        coupled_value_groups = []

    label_names = list(label_names)

    for group in coupled_value_groups:
        for label in group:
            if label not in label_names:
                raise ValueError(
                    f"Label {label} in coupled_value_groups is not in label_names."
                )

    # Build condition units
    # Example:
    # label_names = [A, B, C, D]
    # coupled_value_groups = [[A, B]]
    # condition_units = [(A, B), (C,), (D,)]
    grouped_labels = set()
    condition_units = []

    if deduplicate_coupled_groups:
        for group in coupled_value_groups:
            unit = tuple(group)
            condition_units.append(unit)
            grouped_labels.update(group)

        for label in label_names:
            if label not in grouped_labels:
                condition_units.append((label,))
    else:
        condition_units = [(label,) for label in label_names]

    max_units = len(condition_units)

    targets = []
    target_records = []

    if verbose:
        print("\nAll extrapolation target combinations:")
        print(f"Labels: {label_names}")
        print(f"Condition units: {condition_units}")
        print(f"Controlled label counts: {controlled_labels}")
        print(f"Target levels: {target_levels}")
        print(f"Other label value: {other_label}")

    for n_controlled in controlled_labels:
        if n_controlled < 1:
            raise ValueError("controlled_labels must contain positive integers.")

        if n_controlled > max_units:
            if verbose:
                print(
                    f"[WARN] Skip n_controlled={n_controlled}, "
                    f"because only {max_units} condition units exist."
                )
            continue

        for selected_units in combinations(condition_units, n_controlled):
            selected_labels = set()
            for unit in selected_units:
                selected_labels.update(unit)

            for level in target_levels:
                target_map = {
                    label: float(other_label)
                    for label in label_names
                }

                for label in selected_labels:
                    target_map[label] = float(level)

                target = [round(target_map[label], 1) for label in label_names]
                # Avoid -0.0
                target = [0.0 if abs(v) < 1e-8 else v for v in target]

                targets.append(target)

                target_records.append({
                    "n_controlled_units": n_controlled,
                    "selected_units": selected_units,
                    "level": level,
                    "target": target,
                })

                if verbose:
                    selected_str = " + ".join(
                        ["(" + ", ".join(unit) + ")" for unit in selected_units]
                    )
                    target_str = ", ".join(
                        [f"{label}={val:.4f}" for label, val in zip(label_names, target)]
                    )
                    print(
                        f"n={n_controlled}, level={level:+g}, "
                        f"controlled={selected_str}  -->  [{target_str}]"
                    )

    if verbose:
        print(f"\nTotal extrapolation targets: {len(targets)}")

    return targets