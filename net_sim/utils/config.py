# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {

    "param": {
        "rtt": [20],  # rtt
        "project_folder": r"C:\Users\adina\Technion\Research\MH_Project\Code",  # Code folder (And data)
        "results_folder": r"C:\Users\adina\Technion\Research\MH_Project\Results",  # results folder
        "results_filename_base": "debug12345678",  # results filename - must change each run
        "results_filename": "",
        "T": 1500,  # number of timesteps in each rep
        "debug": False,  # debug flag
        "rep": 1,  # number of repetitions
        "er_rates": [0.1, 0.4, 0.1, 0.1],  # erasure rate of each channel: 0=perfect, 1=all erasure
        "er_var_ind": [2],  # Indices to vary - can't be empty. To have all channels at fixed rate - choose one index and set er_var_values to be one value.
        "er_var_values": [0.2, 0.6, 5],  # Values to vary - same to all indices - star, end, steps number
        "er_load": "from_csv",  # erasure type: erasure, from_mat, from_csv
        "er_type": "BEC",  # erasure type: BEC, GE_p_0.1_q_0.3_g_0.01_b_0.8, For the GE only one channel is going to be GE.
        "ge_channel": 2,  # GE channel index
        "er_series_path": "",  # erasure series path
        "er_estimate_type": "stat",  # genie, stat, stat_max, oracle
        "sigma": 1,  # noise variance
        "print_flag": False,  # print flag, False, True
        "prot_type": "",  # "MIXALL", "BS", "AC"
    },

    "run_index": {
        "rep_index": 0,  # rep index
        "rtt_index": 0,  # rtt index
        "er_var_index": 0,  # er var index
    },

}

