import itertools
import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend

from matplotlib import pyplot as plt
from utils.config import CFG
from utils.config_setup import Config
from run import data_storage
from run import run_1
from run import plot_results


def generate_er_rates_grid(varying_indices, fixed_er_rates, start=0.1, stop=0.9, steps=9):
    """
    Generate a grid of er_rates where one or two specified indices vary between start and stop (inclusive),
    and the rest are fixed manually.

    Parameters:
    varying_indices (list): Indices of the er_rates that will vary (either one or two indices).
    fixed_er_rates (list): The er_rates with manually fixed values.
    start (float): The starting value of the erasure rate.
    stop (float): The stopping value of the erasure rate.
    steps (int): The number of values in the grid (default is 9, from 0.1 to 0.9).

    Returns:
    grid_er_rates (list of lists): All combinations of er_rates in the grid.
    """
    varying_er_rates = np.round(np.linspace(start, stop, steps), 2)

    # Check the number of varying indices and generate corresponding product combinations
    if len(varying_indices) == 1:
        # If only one index varies, we don't need combinations, just the list of varying_er_rates
        er_rates_combinations = [(value,) for value in varying_er_rates]
    elif len(varying_indices) == 2:
        # If two indices vary, generate combinations of both
        er_rates_combinations = list(itertools.product(varying_er_rates, repeat=2))
    else:
        raise ValueError("varying_indices should have a length of 1 or 2.")

    grid_er_rates = []
    for combination in er_rates_combinations:
        er_rates = fixed_er_rates.copy()
        for idx, value in zip(varying_indices, combination):
            er_rates[idx] = value
        grid_er_rates.append(er_rates)

    return grid_er_rates


def save_results(cfg, rtt, er_rates, eta, mean_delay, max_delay):
    """
    Save the results (eta, mean_delay, max_delay) to a specified folder using the config file details.
    The filenames will indicate the RTT and er_rates used.

    Args:
        cfg: Config object containing the parameters including the results folder and filename.
        rtt: Current RTT value.
        er_rates: List of erasure rates used in the current run.
        eta: Eta values.
        mean_delay: Mean delay values.
        max_delay: Max delay values.
    """
    # Create the folder path using config parameters
    results_folder = os.path.join(cfg.param.results_folder, cfg.param.results_filename)

    # Create folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Generate filenames indicating the RTT and er_rates
    rtt_str = f"rtt_{rtt}"
    er_rates_str = "_".join([f"er_{rate:.1f}" for rate in er_rates])

    # Save the results as .npy files
    np.save(os.path.join(results_folder, f"eta_{rtt_str}_{er_rates_str}.npy"), eta)
    np.save(os.path.join(results_folder, f"mean_delay_{rtt_str}_{er_rates_str}.npy"), mean_delay)
    np.save(os.path.join(results_folder, f"max_delay_{rtt_str}_{er_rates_str}.npy"), max_delay)


def load_results(cfg, rtt, er_rates):
    """
    Load the results (eta, mean_delay, max_delay) from the specified folder based on RTT and er_rates.

    Args:
        cfg: Config object containing the parameters including the results folder and filename.
        rtt: The RTT value for which the results were saved.
        er_rates: List of erasure rates used in the current run.

    Returns:
        eta: Loaded eta values.
        mean_delay: Loaded mean delay values.
        max_delay: Loaded max delay values.
    """
    # Create the folder path using config parameters
    results_folder = os.path.join(cfg.param.results_folder, cfg.param.results_filename)

    # Generate filenames indicating the RTT and er_rates
    rtt_str = f"rtt_{rtt}"
    er_rates_str = "_".join([f"er_{rate:.1f}" for rate in er_rates])

    # Load the .npy files
    eta = np.load(os.path.join(results_folder, f"eta_{rtt_str}_{er_rates_str}.npy"))
    mean_delay = np.load(os.path.join(results_folder, f"mean_delay_{rtt_str}_{er_rates_str}.npy"))
    max_delay = np.load(os.path.join(results_folder, f"max_delay_{rtt_str}_{er_rates_str}.npy"))

    return eta, mean_delay, max_delay


def run_all(prot_type=None):
    cfg = Config.from_json(CFG)
    if prot_type is not None:
        cfg.param.prot_type = prot_type
        cfg.param.results_filename = f"{cfg.param.results_filename_base}/{cfg.param.prot_type}"

    # Initialize: ###########################################################################################
    new_folder = os.path.join(cfg.param.results_folder, cfg.param.results_filename)
    if os.path.exists(new_folder):
        print("ERROR: NEW FOLDER NAME ALREADY EXISTS. CHANGE DIRECTORY TO AVOID OVERWRITE TRAINED MODEL")
        exit()
    else:
        os.makedirs(new_folder)

    # Prevent problems loading data:
    if cfg.param.er_estimate_type == "genie":
        cfg.param.er_load = "from_csv"
        print("WARNING: CHANGING ER_LOAD TO from_csv FOR GENIE ESTIMATION")

    ##########################################################################################################

    config_path = os.path.join(cfg.param.results_folder, cfg.param.results_filename_base, 'config.txt')
    with open(config_path, 'a') as text_file:
        json.dump(CFG, text_file, indent=4)

    print(f"Running {cfg.param.prot_type} protocol")
    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
        f.write(f"Running {cfg.param.prot_type} protocol\n")

    # Load Parameters: ########################################################################################
    rep = cfg.param.rep
    rtt_list = cfg.param.rtt
    er_var_values = cfg.param.er_var_values
    ###########################################################################################################

    # Generate Erasures Grid: #################################################################################
    varying_indices = cfg.param.er_var_ind  # Define the indices of the er_rates that will vary
    all_er_rates = cfg.param.er_rates
    start = er_var_values[0]
    stop = er_var_values[1]
    steps = er_var_values[2]
    er_rates_grid = generate_er_rates_grid(varying_indices, all_er_rates, start=start, stop=stop, steps=steps)
    ############################################################################################################

    # Run the simulations: #####################################################################################
    all_df = pd.DataFrame()
    for rtt_idx, rtt in enumerate(rtt_list):
        cfg.param.rtt = rtt  # Update the RTT value for the current run
        cfg.run_index.rtt_index = rtt_idx  # Update the erasure rates for the current run

        # Run all erasure rates options:
        for er_idx, er_rates in enumerate(er_rates_grid):
            cfg.param.er_rates = er_rates  # Update the erasure rates for the current run
            cfg.run_index.er_var_index = er_idx  # Update the erasure rates for the current run

            # Run all repetitions:
            all_r_list = []
            cfg.param.data_storage = data_storage.DataStorage()
            for r in range(rep):
                cfg.run_index.rep_index = r  # Update the repetition number for the current run
                # cfg.run_index.ber_process = generate_ber_events(rate=1-max(cfg.param.er_rates), T=cfg.param.T)

                if cfg.param.in_type == "ber":
                    # ber_events = os.path.join(cfg.param.project_folder, "Data", f"upsample",
                    #                                  f"series_{r+1}.txt")

                    ber_events = os.path.join(cfg.param.project_folder, "Data", f"ber_series_less",
                                              f"rate_{np.round(1 - max(cfg.param.er_rates), 2)}", f"series_{r + 1}.txt")

                    # max(cfg.param.er_rates)

                    cfg.run_index.ber_process = np.array(np.loadtxt(ber_events, dtype=int))

                    print(f"in rate: {np.mean(cfg.run_index.ber_process)}")
                    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                        f.write(f"in rate: {np.mean(cfg.run_index.ber_process)}\n")

                print(f"--- RTT={int(rtt)}, er_rates={er_rates}, Repetition {r + 1} ---")
                with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                    f.write(f"--- RTT={int(rtt)}, er_rates={er_rates}, Repetition {r + 1} ---\n")

                # Run one repetition and log results ###################################################
                df_list = run_1.run_1(cfg, rtt, er_rates, new_folder)

                # Skip the repetition if it failed - e.g. nothing decoded.
                if df_list is None:
                    continue

                # Save the results for the current repetition
                cfg.param.data_storage.save_to_files(os.path.join(new_folder, f"RTT={rtt}_ER_I={er_idx}"))
                all_r_list.append(df_list)
                shutil.copy("ff_log.txt", os.path.join(new_folder,
                                                       f"RTT={rtt}_ER_I={er_idx}", f"ff_log_r={r}.txt"))
                ########################################################################################

            # Concatenate all DataFrames row-wise for each index across the sublists
            new_df = pd.concat(all_r_list, keys=range(len(all_r_list)), names=["Rep"])
            new_df = new_df.reset_index()
            new_df = new_df.drop(columns=['Index', 'Run'])
            new_df["Eps"] = cfg.param.er_rates[cfg.param.er_var_ind[0]]
            new_df = new_df[["Eps", "Rep"] + [col for col in new_df.columns if col not in ["Rep", "Eps"]]]

            all_df = pd.concat([all_df, new_df], ignore_index=True)
            # Save to results folder:
            file_path = os.path.join(cfg.param.results_folder, cfg.param.results_filename,
                                     f"RTT={rtt}_ER_I={er_idx}_all_df.csv")
            all_df.to_csv(file_path, index=False)

        # Save the results for the current RTT value
        all_df.to_csv(os.path.join(new_folder, f"all_df_rtt_{rtt}.csv"), index=False)

        # Print the mean and std values for each node #################################################
        print("/n-------------------------------")
        print(f"Mean and std values for RTT={rtt}")
        with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
            print("/n-------------------------------")
            f.write(f"Mean and std values for RTT={rtt}\n")

        # print the mean and std values for each node
        for eps in all_df['Eps'].unique():
            for node in all_df['Node'].unique():

                print(f"Node {node}, Eps {eps}")
                with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                    f.write(f"Node {node}, Eps {eps}\n")

                for col in all_df.columns:
                    if col in ['Node', 'Eps', 'Rep']:
                        continue
                    mean_value = all_df[(all_df['Node'] == node) & (all_df['Eps'] == eps)][col].mean()
                    std_value = all_df[(all_df['Node'] == node) & (all_df['Eps'] == eps)][col].std()
                    print(f"mean {col}: {mean_value:.2f} ± {std_value:.2f}")
                    # print to file:
                    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                        f.write(f"mean {col}: {mean_value:.2f} ± {std_value:.2f}\n")

                print("-------------------------------")
                with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                    f.write("-------------------------------\n")
        ########################################################################################

    a = 5
    return all_df


def print_1_metric(df1, node_value=-1):
    # Plot ################################################################################################
    # Calculate the mean across all DataFrames for each row (grouped by Index)
    mean_df1 = df1.groupby(["Node", 'Eps']).mean().reset_index()

    # Filter both DataFrames for the selected Node
    df_node1 = mean_df1[mean_df1['Node'] == node_value]

    # Create a figure with subplots (1 row, 5 columns for each metric)
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharex=True, sharey=False)

    # List of columns to plot
    metrics = ['Normalized Goodput', 'Delivery Rate', 'Channel Utilization Rate', 'Mean Delay', 'Max Delay',
               'Mean NC Delay', 'Max NC Delay']

    # Plot each metric for both DataFrames
    for i, column in enumerate(metrics):
        # Plot from the first DataFrame
        axes[i].plot(df_node1['Eps'], df_node1[column], label=f'{column} (df)', linestyle='-', color='blue')
        # Plot from the second DataFrame
        axes[i].set_xlabel('Eps')
        axes[i].set_ylabel(column)
        axes[i].set_title(f'{column} vs Eps')

        # Set y-axis limits based on the column
        if column in ['Normalized Goodput', 'Bandwidth']:
            axes[i].set_ylim(0, 1)  # Limit for Goodput and Bandwidth
        elif column == 'Channel Utilization Rate':
            axes[i].set_ylim(0, 0.15)  # Limit for Channel Utilization Rate

        axes[i].grid(True)  # Add grid to each subplot
        axes[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f'Node {node_value} - Metrics vs Eps', fontsize=16)
    plt.subplots_adjust(top=0.85)  # Adjust to make room for the main title

    plt.show()
    a = 5
    ######################################################################################################
    # Single plot for each metric
    # for column in mean_df.columns:
    #     if column not in ['eps', 'Node']:  # Skip 'eps' and 'Node' as they are index levels
    #         plt.figure(figsize=(8, 6))
    #
    #         # Loop through each unique 'Node'
    #         for node in mean_df.index.get_level_values('Node').unique():
    #             node_data = mean_df[mean_df.index.get_level_values('Node') == node]
    #             plt.plot(node_data.index.get_level_values('Eps'), node_data[column], label=f'Node {node}')
    #
    #         plt.title(f'{column} as a function of eps')
    #         plt.xlabel('eps')
    #         plt.ylabel(column)
    #         plt.legend(title="Node")
    #         plt.grid(True)
    #         plt.show()
    ######################################################################################################


def print_metrics_for_all_dfs(dfs, labels, er_rates, node_value=None, filename=None, type_mid='Real'):
    """
    Plot metrics for any number of DataFrames with labels.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames to plot metrics from.
    labels (list of str): List of labels corresponding to each DataFrame.
    node_value (int): Node value to filter data.
    """

    # Validate input
    if len(dfs) < 1:
        raise ValueError("At least one DataFrame is required.")
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames must match the number of labels.")

    # Prepare data: Calculate means and filter for the selected Node
    dfs_mean = []
    for df in dfs:
        dfs_mean.append(df.groupby(['Node', 'Eps']).mean().reset_index())

    if node_value is None:
        node_value = dfs_mean[0]['Node'].unique()
    else:
        node_value = [node_value]

    # Add plot of the capacity:
    all_eps = df['Eps'].unique()
    eps_bn_fixed = max(er_rates)
    eps_bn_changed = np.array([max(eps_bn_fixed, eps) for eps in all_eps])
    capacity = 1 - eps_bn_changed

    for n in node_value:  # Plot for each node

        if n == -1:
            metrics_to_plot = ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate', 'Mean Delay', 'Max Delay']
        else:
            if type_mid == 'Real':
                metrics_to_plot = ['Normalized Goodput Real', 'Delivery Rate Real', 'Channel Usage Rate Real',
                                   'Mean Real Delay', 'Max Real Delay']

            else:  # semi decoding
                metrics_to_plot = ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate', 'Mean Delay',
                                   'Max Delay']

        dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 6), sharex=True,
                                 sharey=False)

        for i, column in enumerate(metrics_to_plot):

            for j, (df_node, label) in enumerate(zip(dfs_mean_node, labels)):
                axes[i].plot(df_node['Eps'], df_node[column], label=f'{label}', linestyle='-', alpha=0.7)

            axes[i].set_xlabel('Eps')
            axes[i].set_ylabel(column)
            axes[i].set_title(f'{column} vs Eps')
            # axes[i].set_xlim(df_node['Eps'].min(), df_node['Eps'].max())

            # Add capacity line
            if n == -1:
                if column in ['Delivery Rate']:
                    axes[i].plot(all_eps, capacity, label='Capacity', linestyle='--', color='black')

            # Set y-axis limits based on the column
            if column in ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate']:
                axes[i].set_ylim(0, 1.01)

            if column in ['Mean Real Delay']:
                axes[i].set_ylim(0, 200, 50)
            if column in ['Max Real Delay']:
                axes[i].set_ylim(0, 400, 50)
            if column in ['Delivery Rate Real']:
                axes[i].set_ylim(0, 1.01)
            if column in ['Normalized Goodput Real']:
                axes[i].set_ylim(0, 1.01)
            if column in ['Channel Usage Rate Real']:
                axes[i].set_ylim(0, 1.01)

            # if column in ['Mean Delay']:
            #     axes[i].set_ylim(0, 400, 100)
            # if column in ['Max Delay']:
            #     axes[i].set_ylim(0, 400, 100)

            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()
        plt.suptitle(f'Node {n} - Metrics vs Eps', fontsize=16)
        plt.subplots_adjust(top=0.85)
        # plt.show()

        # Save figure:
        # plt.savefig(f'{filename}/Node_{n}_metrics_vs_eps.png')  # , dpi=300)
        plt.savefig(os.path.join(filename, f"Node_{n}_metrics_vs_eps.png"))

    return


def print_metrics_paper(dfs, labels, er_rates, node_value=None, filename=None):
    """
    Plot metrics for any number of DataFrames with labels, generating separate figures for each metric.

    Parameters:
    dfs (list of pd.DataFrame): List of DataFrames to plot metrics from.
    labels (list of str): List of labels corresponding to each DataFrame.
    er_rates (list of float): Error rates for calculating capacity.
    node_value (int): Node value to filter data. If None, include all nodes.
    filename (str): Directory to save figures.
    """

    # Validate input
    if len(dfs) < 1:
        raise ValueError("At least one DataFrame is required.")
    if len(dfs) != len(labels):
        raise ValueError("The number of DataFrames must match the number of labels.")

    # Prepare data: Calculate means and filter for the selected Node
    dfs_mean = [df.groupby(['Node', 'Eps']).mean().reset_index() for df in dfs]

    if node_value is None:
        node_value = dfs_mean[0]['Node'].unique()
    else:
        node_value = [node_value]

    # Add plot of the capacity:
    all_eps = dfs[0]['Eps'].unique()
    eps_bn_fixed = max(er_rates)
    eps_bn_changed = np.array([max(eps_bn_fixed, eps) for eps in all_eps])
    capacity = 1 - eps_bn_changed

    for n in node_value:  # Plot for each node

        # Filter to only 'Channel Usage Rate' when node_value != -1
        if n == -1:
            metrics_to_plot = ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate', 'Mean Delay', 'Max Delay']
        else:
            metrics_to_plot = ['Channel Usage Rate']

        dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean]

        for column in metrics_to_plot:
            # Create a separate figure for each metric
            plt.figure(figsize=(8, 6))

            for df_node, label in zip(dfs_mean_node, labels):
                plt.plot(df_node['Eps'], df_node[column], label=f'{label}', linestyle='-', alpha=0.7, linewidth=2)

            plt.xlabel('Epsilon', fontsize=14)
            # plt.ylabel(column, fontsize=14)

            # if n == -1:
            # plt.title(f'{column}', fontsize=16)\
            # if n != -1:
            #     plt.title(f'{column} - Channel {n-1}', fontsize=16)

            # Add capacity line for Delivery Rate
            if n == -1 and column == 'Delivery Rate':
                plt.plot(all_eps, capacity, label='Capacity', linestyle='--', color='black', linewidth=2)

            # Set y-axis limits for specific metrics
            if column in ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate']:
                plt.ylim(0, 1.01)

            # Set x-axis limits to match epsilon limits
            plt.xlim(min(all_eps), max(all_eps))

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()

            # Save figure for the current metric
            if filename:
                # plt.savefig(f'{filename}/Node_{n}_{column.replace(" ", "_")}_vs_eps.png', dpi=300)
                plt.savefig(os.path.join(filename, "Channel_Usage_Rate_All_vs_eps.png"), dpi=300)

    # def print_metrics_separate(dfs, labels, er_rates, node_value=None, filename=None):
    # """
    # Plot metrics for any number of DataFrames with labels, generating separate figures for each metric.
    # Also creates a combined plot for Channel Usage Rate for all n != -1.
    #
    # Parameters:
    # dfs (list of pd.DataFrame): List of DataFrames to plot metrics from.
    # labels (list of str): List of labels corresponding to each DataFrame.
    # er_rates (list of float): Error rates for calculating capacity.
    # node_value (int): Node value to filter data. If None, include all nodes.
    # filename (str): Directory to save figures.
    # """
    #
    # # Validate input
    # if len(dfs) < 1:
    #     raise ValueError("At least one DataFrame is required.")
    # if len(dfs) != len(labels):
    #     raise ValueError("The number of DataFrames must match the number of labels.")
    #
    # # Prepare data: Calculate means and filter for the selected Node
    # dfs_mean = [df.groupby(['Node', 'Eps']).mean().reset_index() for df in dfs]
    #
    # custom_colors = ['#00A2E8', '#D10056', '#C4A000', '#000000',
    #                  '#1D3557', '#006400', '#FF6347',
    #                  '#20B2AA']
    #
    # # Add plot of the capacity:
    # all_eps = dfs[0]['Eps'].unique()
    # eps_bn_fixed = max(er_rates)
    # eps_bn_changed = np.array([max(eps_bn_fixed, eps) for eps in all_eps])
    # capacity = 1 - eps_bn_changed
    #
    # if node_value is None:
    #     node_value = dfs_mean[0]['Node'].unique()
    #
    #     # Combined plot for all n != -1 (Channel Usage Rate)
    #     non_minus_one_nodes = [n for n in dfs_mean[0]['Node'].unique() if n != -1]
    #     plt.figure(figsize=(10, 8))
    #
    #     for idx, n in enumerate(non_minus_one_nodes):
    #         color = custom_colors[idx % len(custom_colors)]
    #         dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean]
    #         for df_node, label in zip(dfs_mean_node, labels):
    #
    #             if label == "BS-EMPTY":
    #                 plt_label = 'BS-ACRLNC'
    #             elif label == "AC-FEC":
    #                 plt_label = 'Local-ACRLNC'
    #             elif label == "MIXALL":
    #                 plt_label = 'Baseline'
    #             else:
    #                 plt_label = label
    #
    #             plt.plot(df_node['Eps'], df_node['Channel Usage Rate'],
    #                      label=f'{plt_label} - Channel {n - 1}', linestyle='-', alpha=0.7, linewidth=2, color=color)
    #
    #     plt.xlabel('Epsilon', fontsize=14)
    #     # plt.ylabel('Channel Usage Rate', fontsize=14)
    #     plt.ylim(0, 1.01)
    #     plt.xlim(min(all_eps), max(all_eps))
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend(fontsize=12)
    #     plt.tight_layout()
    #
    #     # Save figure if filename is provided
    #     if filename:
    #         plt.savefig(f'{filename}/Channel_Usage_Rate_All_vs_eps.png', dpi=300)
    #
    # else:
    #     node_value = [node_value]
    #     # Separate plots for each metric and node
    #     for n in node_value:  # Plot for each node
    #         # Filter to only 'Channel Usage Rate' when node_value != -1
    #         if n == -1:
    #             metrics_to_plot = ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate', 'Mean Delay',
    #                                'Max Delay']
    #         else:
    #             metrics_to_plot = ['Channel Usage Rate']
    #
    #         dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean]
    #
    #         for column in metrics_to_plot:
    #             # Create a separate figure for each metric
    #             plt.figure(figsize=(8, 6))
    #
    #             for df_node, label in zip(dfs_mean_node, labels):
    #
    #                 if label == "BS-EMPTY":
    #                     plt_label = 'BS-ACRLNC'
    #                     color = custom_colors[-1]
    #                 elif label == "AC-FEC":
    #                     plt_label = 'Local-ACRLN'
    #                     color = custom_colors[-2]
    #                 elif label == "MIXALL":
    #                     plt_label = 'Baseline'
    #                     color = custom_colors[-3]
    #                 else:
    #                     plt_label = label
    #                     color = custom_colors[-4]
    #
    #                 plt.plot(df_node['Eps'], df_node[column], label=f'{plt_label}', linestyle='-', alpha=0.7,
    #                          linewidth=2, color=color)
    #
    #             plt.xlabel('Epsilon', fontsize=14)
    #             # plt.ylabel(column, fontsize=14)
    #
    #             # Add capacity line for Delivery Rate
    #             if n == -1 and column == 'Delivery Rate':
    #                 plt.plot(all_eps, capacity, label='Capacity', linestyle='--', color='black', linewidth=2)
    #
    #             # Set y-axis limits for specific metrics
    #             if column in ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate']:
    #                 plt.ylim(0, 1.01)
    #
    #             # Set x-axis limits to match epsilon limits
    #             plt.xlim(min(all_eps), max(all_eps))
    #
    #             plt.grid(True, linestyle='--', alpha=0.7)
    #             plt.legend(fontsize=12)
    #             plt.tight_layout()
    #
    #             # Save figure for the current metric
    #             if filename:
    #                 # plt.savefig(f'{filename}/Node_{n}_{column.replace(" ", "_")}_vs_eps.png', dpi=300)
    #                 plt.savefig(os.path.join(filename, f"Node_{n}_{column.replace(' ', '_')}_vs_eps.png"), dpi=300)


def load_and_plot(dfs_names, node_value=None):
    cfg = Config.from_json(CFG)
    filename = os.path.join(cfg.param.results_folder, cfg.param.results_filename_base)

    # # Read the configuration from the JSON file - make sure to read the correct configuration
    # with open('config.json', 'r') as json_file:
    #     loaded_config = json.load(json_file)
    #     cfg = Config.from_json(loaded_config)

    er_rates = cfg.param.er_rates

    df_list = []
    for df_name in dfs_names:
        # df = pd.read_csv(f'{filename}/{df_name}/all_df_rtt_{cfg.param.rtt[0]}.csv')
        path = os.path.join(filename, df_name, f"all_df_rtt_{cfg.param.rtt[0]}.csv")
        df = pd.read_csv(path)
        df_list.append(df)

    # print_metrics_for_all_dfs(dfs=df_list, labels=dfs_names, er_rates=er_rates, filename=filename, node_value=node_value)
    plot_results.print_for_paper(cfg=cfg, dfs=df_list, labels=dfs_names, er_rates=er_rates, filename=filename,
                                 node_value=node_value)
    plot_results.print_for_arxiv(cfg=cfg, dfs=df_list, labels=dfs_names, er_rates=er_rates, filename=filename,
                                 node_value=node_value)


def main_run():
    # Choose one option by comment\uncomment

    ###### 1. Run and Plot  ######
    labels = ['MIXALL', 'BS-EMPTY']  # Options: 'MIXALL', 'AC-FEC', 'BS-EMPTY', 'AC-EMPTY', 'BS-FEC'

    # Run:
    all_df = []
    for label in labels:
        df = run_all(prot_type=label)
        all_df.append(df)

    # Plot:
    cfg = Config.from_json(CFG)
    filename = os.path.join(cfg.param.results_folder, cfg.param.results_filename_base)
    print_metrics_for_all_dfs(dfs=all_df,
                              labels=labels,
                              er_rates=cfg.param.er_rates,
                              filename=filename)

    ###### 2. Load and Plot For Paper  ######
    # dfs_names = ['MIXALL', 'AC-FEC', 'BS-EMPTY']  # Only for these options.
    # load_and_plot(dfs_names, node_value=-1)

if __name__ == '__main__':
    main_run()
