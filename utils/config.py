# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {

    "param": {
        "rtt": [20],  # rtt

        "project_folder": r"/data/adina/MH_Project",  # Code folder (And data)
        "results_folder": r"/data/adina/MH_Project/Results",  # results folder
        "results_filename_base": "Demo",  # results filename - must change each run
        "results_filename": "",
        "T": 200,  # number of timesteps in each rep
        "debug": False,  # debug flag
        "rep": 1,  # number of repetitions
        "er_rates": [0.2],  # erasure rate of each channel: 0=perfect, 1=all erasure
        "er_var_ind": [0],  # Indices to vary - can't be empty. To have all channels at fixed rate - choose one index and set er_var_values to be one value.
        "er_var_values": [0.2, 0.2, 1],  # Values to vary - same to all indices - star, end, steps number - overrites er_rates
        "er_load": "from_csv",  # erasure type: erasure, from_csv
        "er_type": "SINR", # erasure type: BEC, GE_p_0.01_g_0.1_b_1, GE_p_0.1_q_0.3_g_0.01_b_0.8, For the GE only one channel is going to be GE. GE_p_0.01_q_0.02_g_0.1_b_1
        "ge_channel": [1],  # GE channel index
        "er_series_path": "",  # erasure series path
        "er_estimate_type": "stat",  # genie, stat, stat_max, oracle
        "sigma": 1,  # noise variance
        "print_flag": True,  # print flag, False, True
        "prot_type": "",  # "MIXALL", "BS", "AC" 
        "in_type": "ber",  # input type: ber, all
        "data_storage": None,  # data storage

    },

    "run_index": {
        "rep_index": 0,  # rep index
        "rtt_index": 0,  # rtt index
        "er_var_index": 0,  # er var index
        "ber_process": None,  # possion series        
    },

}

