# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {

    "param": {
        "rtt": [4],  # rtt
        "project_folder": r"C:\Users\adina\Technion\Research\MH_Project\Code",  # Code folder (And data)
        "results_folder": r"C:\Users\adina\Technion\Re5search\MH_Project\Results",  # results folder
        "results_filename": "85555555555555555555555555555555555555555555555555",  # results filename - must change each run
        "timesteps": 50,  # number of timesteps in each rep
        "debug": False,  # debug flag
        "rep": 1,  # number of repetitions
        "er_rates": [0] * 2,  # erasure rate of each channel: 0=perfect, 1=all erasure
        "er_var_ind": [0],  # Indices to vary - can't be empty. To have all channels at fixed rate - choose one index and set er_var_values to be one value.
        "er_var_values": [0.1, 0.1, 1],  # Values to vary - same to all indices - star, end, steps number
        "er_load": "from_csv",  # erasure type: erasure, from_mat, from_csv
        "er_type": "BEC",  # erasure type: BEC
        "er_series_path": "",  # erasure series path
        "er_estimate_type": "genie",  # genie, stat, stat_max
        "sigma": 1,  # noise variance
        "print_flag": True,  # print flag
    },
}
