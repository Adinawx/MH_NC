# -*- coding: utf-8 -*-
"""Model config in json format"""
import datetime
now = datetime.datetime.now()
tod_string = now.strftime("%Y%m%d_%H%M%S")


CFG = {

    "param": {
        "rtt": [4],  # rtt
        "project_folder": r"C:\Users\shaigi\Desktop\technical\research\network_coding\Re_encoding\MH_NC\Code",  # Code folder (And data)
        "results_folder": r"C:\Users\shaigi\Desktop\technical\research\network_coding\Re_encoding\MH_NC\Results",  # results folder
        "results_filename": tod_string,  # results filename - must change each run
        "timesteps": 30,  # number of timesteps in each rep
        "debug": False,  # debug flag
        "rep": 1,  # number of repetitions
        "er_rates": [0.1, 0.1],  # erasure rate of each channel: 0=perfect, 1=all erasure
        "er_var_ind": [0],  # Indices to vary - can't be empty. To have all channels at fixed rate - choose one index and set er_var_values to be one value.
        "er_var_values": [0.1, 0.1, 1],  # Values to vary - same to all indices - star, end, steps number
        "er_load": "erasure",  # erasure type: erasure, from_mat, from_csv
        "er_type": "BEC",  # erasure type: BEC
        "er_series_path": "",  # erasure series path
        "er_estimate_type": "stat",  # genie, stat, stat_max
        "sigma": 1,  # noise variance
        "COST": 3,  # Transmission cost [time slots]
        "print_flag": True,  # print flag
    },
}
