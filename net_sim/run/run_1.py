import time
import os
import simpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")

from run import set_sim_params
from ns.port.airinterface import AirInterface
from ns.port.nc_node import NC_node
from ns.port.termination_node import Termination_node


def create_network_and_run(cfg, rtt, er_rates):

    ### 1. Read Params:
    env = simpy.Environment()
    timesteps = cfg.param.timesteps
    er_load = cfg.param.er_load
    num_of_nodes = len(cfg.param.er_rates)+1
    default_params = set_sim_params.get_node_default_params()
    default_params['en_enc']['debug'] = cfg.param.debug  # choose what to display from debug

    ### 2. Create Network Topology:

    # 2.1 Nodes
    nc_nodes = []
    for curr_node in range(num_of_nodes):
        nc_nodes.append(NC_node(env, cfg, ind=curr_node, **default_params))

    # 2.2 Channels:
    ff_channels = []
    fb_channels = []
    for curr_ch in range(num_of_nodes - 1):
        eps = er_rates[curr_ch]

        if er_load == 'erasure':
            ff_channels.append(
                AirInterface(env, delay_dist=rtt / 2, noise_dict=set_sim_params.noise_param('erasure', [eps]), wire_id=curr_ch,
                             debug=False))

        elif er_load == 'from_csv':
            path = cfg.param.er_series_path
            path = path.replace('AAA', f"ch_{curr_ch}")
            path = path.replace('BBB', f'{eps:.2f}')
            ff_channels.append(
                AirInterface(env, delay_dist=rtt / 2, noise_dict=set_sim_params.noise_param(er_load, eps, path), wire_id=curr_ch,
                             debug=False))

        fb_channels.append(
            AirInterface(env, delay_dist=rtt / 2, noise_dict=set_sim_params.noise_param('delay_only'), wire_id=curr_ch, debug=False))

    # 2.3 Terminations
    source_term = Termination_node(env, node_type='source', arrival=set_sim_params.constArrival, pct_size=set_sim_params.constPacketSize,
                                   pct_debug=False, sink_debug=False)
    dest_term = Termination_node(env, node_type='destination', arrival=set_sim_params.constArrival, pct_size=set_sim_params.constPacketSize,
                                 pct_debug=False, sink_debug=False)

    # 2.4 Connect nodes and channels:
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

    ### 3. Run:
    env.run(until=timesteps)

    ### 4. Log:
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


def run_1(cfg, rtt, er_rates, new_folder):

    channels_num = len(er_rates)
    timesteps = cfg.param.timesteps

    time1 = time.time()
    create_network_and_run(cfg, rtt, er_rates)
    time2 = time.time()
    print(f"time: {time2 - time1}")

    send_times = np.load(f"{new_folder}\\tran_times.npy")
    dec_times = np.load(f"{new_folder}\\dec_times.npy")
    last_dec = dec_times.shape[0]

    erasures_hist = np.zeros([channels_num, timesteps])
    for n in range(channels_num):
        in_delay = int(
            n * (rtt / 2 + 1))  # Each node receives the first packet after n*(rtt/2+1) timesteps.
        erasures_hist[n, in_delay:] = np.load(f"{new_folder}\\erasures_ch{n}.npy")
    erasures_num = [int(np.sum(1 - erasures_hist[n, int(n * (rtt / 2 + 1)):])) for n in
                    range(channels_num)]  # TODO: A bug in the erasures log.

    if last_dec == 0:
        print("No packets were decoded")
        mean_delay = np.nan
        max_delay = np.nan
        eta = np.nan

    else:
        # delay_i = dec_times[:last_dec] - np.arange(last_dec) - channels_num
        delay_i = dec_times[:last_dec] - send_times[:last_dec] - channels_num

        eta = last_dec / (
                timesteps - (channels_num + 1) * rtt / 2)  # +1: due to the inherent 1 delay of each node.
        mean_delay = np.mean(delay_i, axis=0)
        max_delay = np.max(delay_i)

        print(f"last dec packet: {last_dec}")
        # print(f"erasures num: {erasures_num}")
        print(f"eta: {eta:.2f}")
        print(f"mean delay: {mean_delay:.2f}")
        print(f"max delay: {max_delay:.2f}")

    return eta, mean_delay, max_delay

# TODO: save config file

