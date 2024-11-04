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
            curr_erasure = curr_node.en_enc.store_ff_ch_hist.fifo_items()[curr_time]
            curr_pct = curr_node.en_enc.store_ff_hist.fifo_items()[curr_time]
            arr_time = int(curr_node.en_enc.hist_store_ff_time[time_ind])
            time_ind += 1
            str_log = f"{curr_erasure} || id: {curr_pct.packet_id}, nc id: {curr_pct.nc_serial}, src: {curr_pct.src}, FEC type: {curr_pct.fec_type}, header: {curr_pct.nc_header}, type: {curr_pct.msg_type}"
            log_df.at[arr_time, f'node{curr_node.ind}'.ljust(90)] = str_log.ljust(90)

    log_df.to_csv('ff_log', sep='\t')


def run_1(cfg, rtt, er_rates, new_folder):

    channels_num = len(er_rates)
    timesteps = cfg.param.timesteps

    time1 = time.time()
    create_network_and_run(cfg, rtt, er_rates)
    time2 = time.time()
    print(f"time: {time2 - time1}\n")

    send_times = np.load(f"{new_folder}\\tran_times.npy")
    dec_times = np.load(f"{new_folder}\\dec_times.npy")
    last_dec = dec_times.shape[0]

    erasures_hist = np.ones([channels_num, timesteps])
    erasures_num = np.zeros(channels_num)
    new_num = np.zeros(channels_num)
    fbf_num = np.zeros(channels_num)
    fec_num = np.zeros(channels_num)
    eow_num = np.zeros(channels_num)
    all_fec = np.zeros(channels_num)
    empty_num = np.zeros(channels_num)
    bls_num = np.zeros(channels_num)
    bls_fec_num = np.zeros(channels_num)
    all_empty = np.zeros(channels_num)

    for n in range(channels_num):
        # 1. Ct Type history:
        ct_type_hist = np.load(f"{new_folder}\\ct_type_ch={n}.npy")

        new_num[n] = sum(ct_type_hist == 'NEW')
        fbf_num[n] = sum(ct_type_hist == 'FB-FEC')
        fec_num[n] = sum(ct_type_hist == 'FEC')
        eow_num[n] = sum(ct_type_hist == 'EOW')
        bls_fec_num[n] = sum(ct_type_hist == 'BLS-FEC')
        all_fec[n] = fbf_num[n] + fec_num[n] + eow_num[n] + bls_fec_num[n]
        empty_num[n] = sum(ct_type_hist == 'EMPTY_FEC')
        bls_num[n] = sum(ct_type_hist == 'EMPTY-BLS')
        all_empty[n] = empty_num[n] + bls_num[n]

        if cfg.param.print_flag:
            print(f"Node {n}: NEW: {new_num[n]}, All FEC:{all_fec[n]}, No Tran: {all_empty[n]}")
            print(f"        FB_FEC: {fbf_num[n]}, FEC: {fec_num[n]}, EOW:{eow_num[n]}, FEC-SPACE: {bls_fec_num[n]}\n",
                  f"        EMPTY-BUFFER: {empty_num[n]}, EMPTY-SPACE: {bls_num[n]}\n")

        # 2. Erasures history: from receving in the next node, thus: n+1
        temp = np.load(f"{new_folder}\\erasures_ch={n+1}.npy")
        erasures_hist[n, :len(temp)] = temp

        erasure_means = [
            (1 - erasures_hist[n, :i + 1]).mean() for i in range((rtt-1))
        ]
        erasure_means += [
            (1 - erasures_hist[n, i:i + rtt]).mean()
            for i in range(len(temp) - (rtt-1))
        ]

        erasures_num[n] = np.sum(1 - erasures_hist[n, :len(temp)])
        if cfg.param.print_flag:
            print(f"Channel {n}: erasures num: {erasures_num[n]}\n")

        # 3. Epsilon history: from estimating eps in the PREV node, thus: n
        # eps_hist = np.load(f"{new_folder}\\eps_mean_ch={n}.npy")[:len(temp)]
        # plot epsilon history:
        # Each NODE receives the first packet after n*(rtt/2+1) timesteps.
        # #in_delay = int((n+1) * (rtt / 2 + 1))  # n+1: starting from second node (ignoring the transmitter)
        # t = np.arange(0, len(temp))
        # plt.figure(figsize=(10, 5))
        # plt.plot(t, eps_hist, label='Estimated Epsilon', marker='o', markersize=3)
        # plt.plot(t, erasure_means, label='Erasure rate', marker='o', markersize=3)
        # plt.xlabel('Timesteps')
        # plt.ylabel('Epsilon')
        # plt.title(f'Epsilon history for channel {n}')
        # plt.grid()
        # plt.legend()
        # plt.show()

    if last_dec == 0:
        print("No packets were decoded")
        mean_delay_ones = np.nan
        max_delay_ones = np.nan
        eta = np.nan

    else:
        delay_i_ones = dec_times[:last_dec] - np.arange(last_dec) - channels_num
        delay_i_NC = dec_times[:last_dec] - send_times[:last_dec] - channels_num

        eta = last_dec / (
                timesteps - (channels_num + 1) * rtt / 2)  # +1: due to the inherent 1 delay of each node.
        mean_delay_ones = np.mean(delay_i_ones, axis=0)
        max_delay_ones = np.max(delay_i_ones)

        mean_delay_NC = np.mean(delay_i_NC, axis=0)
        max_delay_NC = np.max(delay_i_NC)

        print(f"last dec packet: {last_dec}")
        all_erasures = np.sum(erasures_num)
        print(f"all erasures num: {all_erasures}")

        all_empty_fec = np.sum(empty_num)
        print(f"empty buffer num: {all_empty_fec}")
        all_bls = np.sum(bls_num)
        print(f"empty spaces num: {all_bls}")

        print(f"eta: {eta:.2f}")
        print(f"mean delay: {mean_delay_ones:.2f}")
        print(f"max delay: {max_delay_ones:.2f}")
        print(f"mean delay NC: {mean_delay_NC:.2f}")
        print(f"max delay NC: {max_delay_NC:.2f}")

    return eta, mean_delay_ones, max_delay_ones

# TODO: save config file

