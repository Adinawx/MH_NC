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

class RunLog:

    def __init__(self):
        self.N = None # number of nodes
        self.n = None  # node's inedx

        self.arrival_times = None  # arrival times - the time packet i (nc_id from last node!) arrives at the current node.
        self.trans_times = None  # transmission times - first time packet i (nc_id from last node!) is transmitted in a new c_t.
        self.trans_types = None  # transmission types - FEC/NEW etc, as sent from the current node.
        self.erasures = None  # erasures of packet i as detected at the next node
        self.epsilons = None  # epsilons estimations
        self.semi_dec_times = None  # semi-decoded times
        self.last_semi_dec = None  # last semi-decoded packet
        self.dec_times = None  # info-packets decoded times, relevant only at the reciever
        self.last_dec = None  # last decoded info-packet, relevant only at the reciever

        self.new_num = None
        self.fbf_num = None
        self.fec_num = None
        self.eow_num = None
        self.empty_buffer_num = None
        self.empty_bls_num = None

    def load_data(self, N, n, new_folder):
        self.N = N
        self.n = n

        self.arrival_times = np.load(f"{new_folder}\\arrival_times_ch={n}.npy")
        self.semi_dec_times = np.load(f"{new_folder}\\semi_dec_times_ch={n}.npy")
        self.last_semi_dec = self.semi_dec_times.shape[0]
        self.epsilons = np.load(f"{new_folder}\\eps_mean_ch={n}.npy")

        if n < N-1:  # if not the last node
            self.erasures = np.load(f"{new_folder}\\erasures_ch={n+1}.npy")  # from the next node.
            self.trans_times = np.load(f"{new_folder}\\trans_times_ch={n}.npy")
            self.trans_types = np.load(f"{new_folder}\\trans_types_ch={n}.npy")

        if n == N-1:  # if the last node
            self.dec_times = np.load(f"{new_folder}\\dec_times.npy")
            self.last_dec = self.dec_times.shape[0]

    def set_types(self):
        ct_type_hist = self.trans_types

        self.new_num = sum(ct_type_hist == 'NEW')
        self.fbf_num = sum(ct_type_hist == 'FB-FEC')
        self.fec_num = sum(ct_type_hist == 'FEC')
        self.eow_num = sum(ct_type_hist == 'EOW')
        self.empty_buffer_num = sum(ct_type_hist == 'EMPTY_BUFFER')
        self.empty_bls_num = sum(ct_type_hist == 'EMPTY-BLS')

    def print_trans_log(self, T, rtt):
        all_fec = self.fbf_num + self.fec_num + self.eow_num
        all_empty = self.empty_buffer_num + self.empty_bls_num

        print(f"Node {self.n}: NEW: {self.new_num}, All FEC:{all_fec}, No Tran: {all_empty}")
        print(f"        FB_FEC: {self.fbf_num}, FEC: {self.fec_num}, EOW:{self.eow_num}\n",
              f"        EMPTY-BUFFER: {self.empty_buffer_num}, EMPTY-SPACE: {self.empty_bls_num}\n")

        ch_util_rate = all_empty / (T - self.n * (rtt / 2 + 1))
        print(f"Channel Utilization Rate: {ch_util_rate:.2f}\n")

        # Fix erasures log delay:
        erasures_hist = self.erasures

        erasure_means = [
            (1 - erasures_hist[:i + 1]).mean() for i in range((rtt - 1))
        ]
        erasure_means += [
            (1 - erasures_hist[i:i + rtt]).mean()
            for i in range(len(erasures_hist) - (rtt - 1))
        ]
        erasures_num = np.sum(1 - erasures_hist)
        print(f"Channel {self.n}: erasures num: {erasures_num}\n")

def create_network_and_run(cfg, rtt, er_rates):

    ### 1. Read Params:
    env = simpy.Environment()
    T = cfg.param.T
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

            # Read Data From A File #################################################################
            er_type = 'BEC'
            channel_name = "erasure"
            if "GE" in cfg.param.er_type and curr_ch == cfg.param.ge_channel:
                er_type = cfg.param.er_type
                channel_name = "GE"

            cfg.param.er_series_path = f"{cfg.param.project_folder}" \
                                       f"\\Data\\{er_type}\\ch_{curr_ch}\\" \
                                       f"{channel_name}_series_eps_{eps:.2f}_series_{cfg.run_index.rep_index}.csv"
            path = cfg.param.er_series_path
            ########################################################################################

            # path = cfg.param.er_series_path
            # path = path.replace('AAA', f"ch_{curr_ch}")
            # path = path.replace('BBB', f'{eps:.2f}')

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
    env.run(until=T)

    ### 4. Log:
    col_space = 120
    # log_df = pd.DataFrame(columns=[f'node{i} -> node{i+1}'.ljust(col_space) for i in range(num_of_nodes)])

    # Generate column names
    columns = ["SRC"] + [f'node{i} -> node{i + 1}' for i in range(num_of_nodes-1)]
    # Create the DataFrame
    columns = [col.ljust(col_space) for col in columns]
    log_df = pd.DataFrame(columns=columns)

    # Add rows to the DataFrame for each timestep
    for timestep in range(T):
        log_df.loc[timestep] = [''] * num_of_nodes  # Initialize values to 0 for each node

    for curr_node in nc_nodes:
        time_ind = 0
        for curr_time in range(len(curr_node.en_enc.store_ff_hist.fifo_items())):
            curr_erasure = curr_node.en_enc.store_ff_ch_hist.fifo_items()[curr_time]
            curr_pct = curr_node.en_enc.store_ff_hist.fifo_items()[curr_time]
            arr_time = int(curr_node.en_enc.hist_store_ff_time[time_ind])
            time_ind += 1
            # str_log = f"{curr_erasure} || id: {curr_pct.packet_id}, nc id: {curr_pct.nc_serial}, src: {curr_pct.src}, FEC type: {curr_pct.fec_type}, header: {curr_pct.nc_header}, type: {curr_pct.msg_type}"
            # log_df.at[arr_time, f'node{curr_node.ind}'.ljust(col_space)] = str_log.ljust(col_space)

            str_log = f"{curr_erasure} || id: {curr_pct.packet_id}, nc id: {curr_pct.nc_serial}, src: {curr_pct.src}, FEC type: {curr_pct.fec_type}, header: {curr_pct.nc_header}, type: {curr_pct.msg_type}"
            log_df.at[arr_time, columns[curr_node.ind].ljust(col_space)] = str_log.ljust(col_space)

    # Save locally:
    log_df.to_csv('ff_log.txt', sep='\t')


def run_1(cfg, rtt, er_rates, new_folder):
    # Run Simulation #########################################
    time1 = time.time()
    create_network_and_run(cfg, rtt, er_rates)
    time2 = time.time()
    print(f"time: {time2 - time1}\n")
    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
        f.write(f"time: {time2 - time1}\n")

    if cfg.param.debug:
        max_delay_ones = np.nan
        mean_delay_ones = np.nan
        eta = np.nan
        return max_delay_ones, mean_delay_ones, eta
    ##########################################################

    # Each node log: #########################################
    channels_num = len(er_rates)
    N = channels_num + 1
    T = cfg.param.T

    run_log = [RunLog() for _ in range(N)]
    run_log_results = []
    for n in range(N):

        run_log[n].load_data(N, n, new_folder)

        if n < N - 1:
            run_log[n].set_types()
            if cfg.param.print_flag:
                run_log[n].print_trans_log(T, rtt)

        if 0 < n:
            # From current node:
            semi_dec_times = run_log[n].semi_dec_times
            last_semi_dec = run_log[n].last_semi_dec
            # From previous node:
            arrival_times = run_log[n-1].arrival_times
            all_empty = run_log[n-1].empty_buffer_num + run_log[n-1].empty_bls_num

            # Calculate metrics:
            norm_goodput = last_semi_dec / (T - int(rtt / 2) * n - all_empty)
            period_len = 100
            num_periods = int(T / period_len)
            decoded_counts, _ = np.histogram(semi_dec_times, bins=np.linspace(0, T, num_periods + 1))
            bw = np.mean(decoded_counts) / period_len
            ch_util_rate = all_empty / (T - int(rtt / 2) * n)
            # print(f"last semi dec:{last_semi_dec}, arrival times:{len(arrival_times)}")
            delay = semi_dec_times - arrival_times[:last_semi_dec]
            mean_delay = np.mean(delay)
            max_delay = np.max(delay)

            trans_times = run_log[n-1].trans_times
            # print(f"last semi dec:{last_semi_dec}, arrival times:{len(trans_times)}")
            # if last_semi_dec < len(trans_times):
            #     print("Good")
            # else:
            #     print("Bad")
            nc_delay_semi = semi_dec_times - trans_times[:last_semi_dec]
            mean_nc_delay_semi = np.mean(nc_delay_semi)
            max_nc_delay_semi = np.max(nc_delay_semi)

            if cfg.param.print_flag:
                print(f"Node {n}:")
                print(f"Normalized Goodput: {norm_goodput:.2f}")
                print(f"Delivery Rate: {bw:.2f}")
                print(f"Channel Utilization Rate: {ch_util_rate:.2f}")
                print(f"Mean Delay: {mean_delay:.2f}")
                print(f"Max Delay: {max_delay:.2f}")
                print(f"Mean NC Delay: {mean_nc_delay_semi:.2f}")
                print(f"Max NC Delay: {max_nc_delay_semi:.2f}\n")

            with open(f"{new_folder}\\metrics_nodes.txt", "a") as f:
                f.write(f"Node {n}:\n")
                f.write(f"Normalized Goodput: {norm_goodput:.2f}\n")
                f.write(f"Delivery Rate: {bw:.2f}\n")
                f.write(f"Channel Utilization Rate: {ch_util_rate:.2f}\n")
                f.write(f"Mean Delay: {mean_delay:.2f}\n")
                f.write(f"Max Delay: {max_delay:.2f}\n")
                f.write(f"Mean NC Delay: {mean_nc_delay_semi:.2f}\n")
                f.write(f"Max NC Delay: {max_nc_delay_semi:.2f}\n\n")
                f.write("-----------------------------------\n\n")

            # Save results:
            dict = {
                "Node": n,
                "Normalized Goodput": norm_goodput,
                "Delivery Rate": bw,
                "Channel Utilization Rate": ch_util_rate,
                "Mean Delay": mean_delay,
                "Max Delay": max_delay,
                "Mean NC Delay": mean_nc_delay_semi,
                "Max NC Delay": max_nc_delay_semi,
            }
            df = pd.DataFrame(dict, index=[0])
            run_log_results.append(df)
    ###########################################################

    # Full System Performance: ################################

    # Real Decoding At The Receiver:
    dec_times = run_log[N - 1].dec_times
    last_dec = run_log[N - 1].last_dec

    if last_dec == 0:
        print("No packets were decoded")
        return None

    _, wins_len = np.unique(dec_times, return_counts=True)
    print(f"max window length: {np.max(wins_len)}")
    print(f"mean window length: {np.mean(wins_len)}")
    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
        f.write(f"max window length: {np.max(wins_len)}\n")
        f.write(f"mean window length: {np.mean(wins_len)}\n")

    all_empty_source = run_log[0].empty_bls_num + run_log[0].empty_buffer_num
    norm_goodput = last_dec / (T - int(rtt/2) * (N-1) - all_empty_source)

    period_len = 100
    num_periods = int(T / period_len)
    decoded_counts, _ = np.histogram(dec_times, bins=np.linspace(0, T, num_periods + 1))
    bw = np.mean(decoded_counts)/period_len

    all_empty = sum([run_log[i].empty_bls_num + run_log[i].empty_buffer_num for i in range(0, N-1)])
    ch_util_rate = all_empty / ((N-1)*T - int(rtt/2) * (N-2)*(N-1)/2)

    arrival_times = run_log[0].arrival_times
    delay = dec_times - arrival_times[:last_dec]
    mean_delay = np.mean(delay)
    max_delay = np.max(delay)

    nc_delay = dec_times - run_log[0].trans_times[:last_dec]
    mean_nc_delay = np.mean(nc_delay)
    max_nc_delay = np.max(nc_delay)

    print("---- Full System ----")
    print(f"Normalized Goodput: {norm_goodput:.2f}")
    print(f"Delivery Rate: {bw:.2f}")
    print(f"Channel Util Rate: {ch_util_rate:.2f}")
    print(f"Mean Delay: {mean_delay:.2f}")
    print(f"Max Delay: {max_delay:.2f}")
    print(f"Mean NC Delay: {mean_nc_delay:.2f}")
    print(f"Max NC Delay: {max_nc_delay:.2f}")

    # save print to file:
    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
        f.write("---- Full System ----\n")
        f.write(f"Normalized Goodput: {norm_goodput:.2f}\n")
        f.write(f"Delivery Rate: {bw:.2f}\n")
        f.write(f"Channel Util Rate: {ch_util_rate:.2f}\n")
        f.write(f"Mean Delay: {mean_delay:.2f}\n")
        f.write(f"Max Delay: {max_delay:.2f}\n")
        f.write(f"Mean NC Delay: {mean_nc_delay:.2f}\n")
        f.write(f"Max NC Delay: {max_nc_delay:.2f}\n")
        f.write("-----------------------------------\n\n")
    ###########################################################

    # RETURN RESULTS #########################################
    dict = {
        "Node": -1,
        "Normalized Goodput": norm_goodput,
        "Delivery Rate": bw,
        "Channel Utilization Rate": ch_util_rate,
        "Mean Delay": mean_delay,
        "Max Delay": max_delay,
        "Mean NC Delay": mean_nc_delay,
        "Max NC Delay": max_nc_delay,
    }
    df = pd.DataFrame(dict, index=[0])
    run_log_results.append(df)
    ###########################################################

    combined_df = pd.concat(run_log_results, keys=range(len(run_log_results)), names=["Run", "Index"])

    return combined_df
