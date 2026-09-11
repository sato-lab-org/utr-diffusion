import numpy as np
# discrete single label
mrl_target_values_3 = [4.0, 6.0, 8.0]
mrl_target_values_5 = [4.0, 5.0, 6.0, 7.0, 8.0]
mrl_target_values_7 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
mfe_target_values_3 = [-15, -10, -5]
mfe_target_values_5 = [-15, -12.5, -10, -7.5, -5]
mfe_target_values_7 = [-30, -15, -12.5, -10, -7.5, -5, 0]
mrl_target_extreme = [2.0] #[9.0]


# continuous single label experiment 2->9
mrl_target_values_sweep = np.arange(2.0, 9.0 + 0.01, 0.1).round(1).tolist()
mfe_target_values_sweep = np.arange(-35, 0.0 + 0.01, 0.5).round(1).tolist()

# continuous multi-labels experiment
joint_target_values_3x3 = [[4.0, -20.0],[4.0, -10.0], [4.0, -2.0],
                        [6.0, -20.0],[6.0, -10.0], [6.0, -2.0],
                        [8.0, -20.0],[8.0, -10.0], [8.0, -2.0]]

joint_target_values_2x2 = [[4.0, -20.0], [4.0, -2.0], [8.0, -20.0], [8.0, -2.0]]

# Crest BioDX Request ASAI group
# MRL-low_MFE-low & MRL-high_MFE-low, MRL-high_MFE-high
CML_Crest_request = [
    [2.75, -0.1], [2.75, -1.0], [2.75, -2.0], [2.75, -3.0], [2.75, -4.0], # MRL-low_MFE-low
    [8.25, -0.1], [8.25, -1.0], [8.25, -2.0], [8.25, -3.0], [8.25, -4.0], # MRL-high_MFE-low
    [8.25, -15.0],[8.25, -17.5],[8.25, -20.0],[8.25, -22.5],[8.25, -25.0], # MRL-high_MFE-high
    [2.75, -15.0],[2.75, -17.5],[2.75, -20.0],[2.75, -22.5],[2.75, -25.0], # MRL-low_MFE-high
]

#exp_mrl_range = np.linspace(3, 9, 12)
#exp_mfe_range = np.linspace(-30, 0, 12)

sweep_range_mrl = np.linspace(4, 8, 11)
sweep_range_mfe = np.linspace(-20, 0, 11)
joint_target_values_sweep = [[float(mrl), float(mfe)] for mrl in sweep_range_mrl for mfe in sweep_range_mfe]


