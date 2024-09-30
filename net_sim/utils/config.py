# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {

    "param": {
        "rtt": 4.0,  # rtt
        "project_folder": r"C:\Users\adina\Technion\Research\MH_Project\Code",  # Code folder
        "results_folder": r"C:\Users\adina\Technion\Research\MH_Project\Results",  # results folder
        "results_filename": "run1",  # results filename - must change each run
        "num_of_nodes": 3,  # number of nodes
        "timesteps": 2000,  # number of timesteps in each rep
        "debug": False,  # debug flag
        "rep": 5,  # number of repetitions
        "er_rate": [0.3, 0.1],  # erasure rate of each channel: 0=perfect, 1=all erasure
        "er_load": "from_csv",  # erasure type: erasure, from_mat, from_csv
        "er_type": "BEC",  # erasure type: BEC
        "er_series_path": "",  # erasure series path
        "er_estimate_type": "stat",  # genie, stat
        "print_flag": False,  # print flag
    },
}
