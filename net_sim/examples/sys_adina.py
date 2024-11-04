"""
A basic example that connects two packet generators to a network wire with
a propagation dec_timea distribution, and then to a packet sink.
"""
import time
from functools import partial
import random
from random import expovariate
import os
import sys
import simpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")


from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.port.airinterface import AirInterface
from ns.port.port import Port
from ns.port.buffer_manager import BufferManeger
from ns.port.nc_enc import NCEncoder
from ns.port.nc_node import NC_node
from ns.port.termination_node import Termination_node
from utils.config import CFG
from utils.config_setup import Config


def constArrival():
    return 1  # time interval


def constPacketSize():
    return 0.0  # bytes, Proportional to the processing time of packets in a node


def constDelay():
    return 2.0  # Delay [steps] # rtt/2


def noise_param(noise_type, *args):
    if noise_type == 'delay_only':
        noise_dict = {
            'type': noise_type,
            'debug': False
        }
    elif noise_type == 'Gaussian':  # Example for args: args = [0,1]
        noise_dict = {
            'type': noise_type,
            'mean': args[0],
            'variance': args[1],
            'debug': False
        }
    elif noise_type == 'erasure':  # Example for args: args = [0.1]
        noise_dict = {
            'type': noise_type,
            'p_e': args[0],
            'debug': False
        }
    elif noise_type == 'from_mat':  # Example for args: args = 'C:\\Users\\tmp.mat'
        noise_dict = {
            'type': noise_type,
            'path': 'C:\\Users\\shaigi\\Desktop\\deepNP\\SINR\\SINR_Mats\\scenario_fast\\sinr_mats_test\\SINR(111).mat',
            'debug': False
        }
    elif noise_type == 'from_csv':  # Example for args: args = 'C:\\Users\\tmp.mat'
        noise_dict = {
            'type': noise_type,
            'eps_hist': args[0],
            'path': args[1],
            'debug': False
        }
    else:
        print(["Wrong input to noise_param. Num of input params:" + str(len(args))])
        return None

    return noise_dict


def curr_loc() -> str:
    file_name = os.path.basename(__file__)
    return f"File: {file_name}, Line: {sys._getframe().f_lineno}"


def get_node_default_params():
    ff_buffer = {'capacity': float('inf'), 'memory_size': float('inf'), 'debug': False}
    ff_pct = {'arrival_dist': constArrival, 'size_dist': constPacketSize, 'debug': False}
    en_enc = {'enc_default_len': float('inf'), 'channel_default_len': 5, 'debug': False}
    fb_buffer = {'capacity': 5, 'memory_size': 5, 'debug': False}
    fb_pct = {'arrival_dist': constArrival, 'size_dist': constPacketSize, 'debug': False}

    node_default_params = {
        'ff_buffer': ff_buffer,
        'ff_pct': ff_pct,
        'en_enc': en_enc,
        'fb_buffer': fb_buffer,
        'fb_pct': fb_pct,
    }

    return node_default_params


def run_1(cfg):
    ### New - 18/8  ######################
    # Redirect stdout and stderr to a file
    # stdout_file = open("output.log", "w")
    # stderr_file = open("error.log", "w")
    # sys.stdout = stdout_file
    # sys.stderr = stderr_file

    # print("Script Parameters:")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Index {i}: Node {arg}")
    ######################################

    env = simpy.Environment()
    timesteps = cfg.param.timesteps
    rtt = cfg.param.rtt
    er_load = cfg.param.er_load

    # Get node default params (note: all debug flags are set to False)
    default_params = get_node_default_params()

    # choose what to display from debug:
    default_params['en_enc']['debug'] = cfg.param.debug

    # Topology
    num_of_nodes = len(cfg.param.er_rates)+1

    # -------------
    # Components: |
    # -------------
    # Nodes
    nc_nodes = []
    for curr_node in range(num_of_nodes):
        nc_nodes.append(NC_node(env, cfg, ind=curr_node, **default_params))

    # Channels
    ff_channels = []
    fb_channels = []
    for curr_ch in range(num_of_nodes - 1):
        eps = cfg.param.er_rates[curr_ch]

        if er_load == 'erasure':
            ff_channels.append(
                AirInterface(env, delay_dist=rtt / 2, noise_dict=noise_param('erasure', [eps]), wire_id=curr_ch,
                             debug=False))

        elif er_load == 'from_csv':
            path = cfg.param.er_series_path
            path = path.replace('AAA', f"ch_{curr_ch}")
            path = path.replace('BBB', f'{eps:.2f}')
            ff_channels.append(
                AirInterface(env, delay_dist=rtt / 2, noise_dict=noise_param(er_load, eps, path), wire_id=curr_ch,
                             debug=False))

        fb_channels.append(
            AirInterface(env, delay_dist=rtt / 2, noise_dict=noise_param('delay_only'), wire_id=curr_ch, debug=False))

    # Terminations
    source_term = Termination_node(env, node_type='source', arrival=constArrival, pct_size=constPacketSize,
                                   pct_debug=False, sink_debug=False)
    dest_term = Termination_node(env, node_type='destination', arrival=constArrival, pct_size=constPacketSize,
                                 pct_debug=False, sink_debug=False)

    source_term.pct_gen.out = nc_nodes[0].ff_in
    for i in range(num_of_nodes - 1):
        nc_nodes[i].ff_out.out = ff_channels[i]
        ff_channels[i].out = nc_nodes[i + 1].ff_in
    nc_nodes[num_of_nodes - 1].ff_out.out = dest_term.sink

    dest_term.pct_gen.out = nc_nodes[num_of_nodes - 1].fb_in
    for i in range(num_of_nodes - 1, 0, -1):
        nc_nodes[i].fb_out.out = fb_channels[i - 1]
        fb_channels[i - 1].out = nc_nodes[i - 1].fb_in
    nc_nodes[0].fb_out.out = source_term.sink

    env.run(until=timesteps)

    ### New - 18/8  ####################################################################################################
    # Logging
    log_df = pd.DataFrame(columns=[f'node{i}'.ljust(90) for i in range(num_of_nodes)])

    # Add rows to the DataFrame for each timestep
    for timestep in range(timesteps):
        log_df.loc[timestep] = [''] * num_of_nodes  # Initialize values to 0 for each node

    for curr_node in nc_nodes:
        time_ind = 0
        for curr_time in range(len(curr_node.en_enc.store_ff_hist.fifo_items())):
            curr_pct = curr_node.en_enc.store_ff_hist.fifo_items()[curr_time]
            arr_time = int(curr_node.en_enc.hist_store_ff_time[time_ind])
            time_ind += 1
            str_log = f"id: {curr_pct.packet_id}, nc id: {curr_pct.nc_serial}, src: {curr_pct.src}, FEC type: {curr_pct.fec_type}, header: {curr_pct.nc_header}, type: {curr_pct.msg_type}"
            log_df.at[arr_time, f'node{curr_node.ind}'.ljust(90)] = str_log.ljust(90)

    log_df.to_csv('ff_log', sep='\t')

    # stdout_file.close()
    # stderr_file.close()
    # Reset stdout and stderr to the default values
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    ####################################################################################################################


def run():
    cfg = Config.from_json(CFG)

    new_folder = r"{}\{}".format(cfg.param.results_folder, cfg.param.results_filename)
    isExist = os.path.exists(new_folder)
    if isExist:
        print("ERROR: NEW FOLDER NAME ALREADY EXISTS. CHANGE DIRECTORY TO AVOID OVERWRITE TRAINED MODEL")
        exit()
    else:
        os.makedirs(new_folder)

    # Config params:
    rep = cfg.param.rep
    channels_num = len(cfg.param.er_rates)
    timesteps = cfg.param.timesteps
    rtt_list = cfg.param.rtt
    num_of_rtt = len(rtt_list)

    # Allocate arrays for results
    mean_delay = np.zeros([num_of_rtt, rep])
    max_delay = np.zeros([num_of_rtt, rep])
    eta = np.zeros([num_of_rtt, rep])

    for rtt_idx, rtt in enumerate(rtt_list):

        run_1.counter = 0
        for r in range(rep):
            print(f"---Repetition {r + 1}---")

            # When reading data from a file, determine here the relevant path, AAA=Channel number and BBB=eps_hist.
            cfg.param.er_series_path = f"{cfg.param.project_folder}" \
                   f"\\Data\\{cfg.param.er_type}\\AAA\\" \
                   f"erasure_series_eps_BBB_series_{r}.csv"

            time1 = time.time()
            run_1(cfg)
            time2 = time.time()
            print(f"time: {time2 - time1}")

            send_times = np.load(f"{new_folder}\\tran_times.npy")
            dec_times = np.load(f"{new_folder}\\dec_times.npy")
            last_dec = dec_times.shape[0]

            erasures_hist = np.zeros([rep, channels_num, timesteps])
            for n in range(channels_num):
                in_delay = int(n*(rtt/2+1))  # Each node receives the first packet after n*(rtt/2+1) timesteps.
                erasures_hist[r, n, in_delay:] = np.load(f"{new_folder}\\erasures_ch{n}.npy")
            erasures_num = [int(np.sum(1-erasures_hist[r, n, int(n*(rtt/2+1)):])) for n in range(channels_num)] # TODO: A bug in the erasures log.

            if last_dec == 0:
                print("No packets were decoded")
                mean_delay[r] = np.nan
                max_delay[r] = np.nan
                eta[r] = np.nan
                continue

            # delay_i = dec_times[:last_dec] - np.arange(last_dec) - channels_num
            delay_i = dec_times[:last_dec] - send_times[:last_dec] - channels_num

            eta[rtt_idx, r] = last_dec / (timesteps - (channels_num + 1) * rtt / 2)  # +1: due to the inherent 1 delay of each node.
            mean_delay[rtt_idx, r] = np.mean(delay_i, axis=0)
            max_delay[rtt_idx, r] = np.max(delay_i)

            print(f"\n")
            print(f"last dec packet: {last_dec}")
            # print(f"erasures num: {erasures_num}")
            print(f"eta: {eta[rtt_idx, r]:.2f}")
            print(f"mean delay: {mean_delay[rtt_idx, r]:.2f}")
            print(f"max delay: {max_delay[rtt_idx, r]:.2f}")

            run_1.counter += 1

        # Print final mean:
        er_rates = cfg.param.er_rates
        print(f"\n")
        print("----Final Results----")
        print(f"RTT={rtt}")
        print(f"Channel rates: {1 - np.array(er_rates)}")
        print(f"Mean eta: {np.mean(eta[rtt_idx, :]):.2f} +- {np.std(eta[rtt_idx, :]):.2f}")
        print(f"Mean delay: {np.mean(mean_delay[rtt_idx, :]):.2f} +- {np.std(mean_delay[rtt_idx, :]):.2f}")
        print(f"Max delay: {np.mean(max_delay[rtt_idx, :]):.2f} +- {np.std(max_delay[rtt_idx, :]):.2f}")

    # Plot erasure series:
    n_plt = 1
    plt.figure()
    for r_plt in range(rep):
        plt.plot(range(int(n_plt*(rtt/2+1)), timesteps), erasures_hist[r_plt, n_plt, int(n_plt*(rtt/2+1)):], label=f"rep {r_plt}")
    plt.legend()
    plt.grid()
    plt.title(f"Erasure series for node {n_plt}")
    plt.show()

    # Plot mean delay:


    a = 5


# TODO: save config file

