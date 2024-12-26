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
from run import run_1


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

    # Save the basic configuration to a JSON file
    with open('config.json', 'w') as json_file:
        json.dump(CFG, json_file, indent=4)

    # Initialize: #####################################################################################
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
    #########################################################################################################

    print(f"Running {cfg.param.prot_type} protocol")
    with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
        f.write(f"Running {cfg.param.prot_type} protocol\n")

    # Load Parameters: ######################################################################################
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

    # Run the simulations: #########################################################################################
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
            for r in range(rep):
                cfg.run_index.rep_index = r  # Update the repetition number for the current run

                print(f"--- RTT={int(rtt)}, er_rates={er_rates}, Repetition {r + 1} ---")
                with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                    f.write(f"--- RTT={int(rtt)}, er_rates={er_rates}, Repetition {r + 1} ---\n")

                # # Read Data From A File #################################################################
                # # When reading data from a file, determine the relevant path using {r}.
                # # Later: AAA=Channel index and BBB=eps_hist.
                # cfg.param.er_series_path = f"{cfg.param.project_folder}" /
                #                            f"/Data/{cfg.param.er_type}/AAA/" /
                #                            f"erasure_series_eps_BBB_series_{r}.csv"
                # ########################################################################################

                # Run one repetition and log results ###################################################
                df_list = run_1.run_1(cfg, rtt, er_rates, new_folder)

                if df_list is None:  # Skip the repetition if it failed - e.g. nothing decoded.
                    continue

                all_r_list.append(df_list)
                shutil.copy("ff_log.txt", os.path.join(cfg.param.results_folder, cfg.param.results_filename,
                                                       f"RTT={rtt}_ER_I={er_idx}_r={r}_ff_log.txt"))
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
                                     f"RTT={rtt}_ER_I={er_idx}_r={r}_all_df.csv")
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


def print_metrics_for_all_dfs(dfs, labels, er_rates, node_value=None, filename=None):
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

    # List of columns to plot
    metrics = ['Normalized Goodput', 'Delivery Rate', 'Channel Usage Rate', 'Mean Delay', 'Max Delay']

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

        dfs_mean_node = [df[df['Node'] == n] for df in dfs_mean]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6), sharex=True, sharey=False)

        for i, column in enumerate(metrics):
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
            # if column in ['Mean Delay']:
            #     axes[i].set_ylim(300, 800)
            # if column in ['Max Delay']:
            #     axes[i].set_ylim(600, 1200)

            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()
        plt.suptitle(f'Node {n} - Metrics vs Eps', fontsize=16)
        plt.subplots_adjust(top=0.85)
        # plt.show()

        # Save figure:
        plt.savefig(f'{filename}/Node_{n}_metrics_vs_eps.png')  # , dpi=300)

    return


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
        df = pd.read_csv(f'{filename}/{df_name}/all_df_rtt_{cfg.param.rtt[0]}.csv')
        df_list.append(df)

    print_metrics_for_all_dfs(dfs=df_list, labels=dfs_names, er_rates=er_rates, filename=filename,
                              node_value=node_value)


def main_run():
    df_mixall = run_all(prot_type="MIXALL")

    df_ac_fec = run_all(prot_type="AC-FEC")

    df_bs_empty = run_all(prot_type="BS-EMPTY")

    df_bs_fec = run_all(prot_type="BS-FEC")

    df_ac_empty = run_all(prot_type="AC-EMPTY")

    cfg = Config.from_json(CFG)
    filename = os.path.join(cfg.param.results_folder, cfg.param.results_filename_base)

    print_metrics_for_all_dfs(dfs=[df_ac_fec, df_ac_empty, df_bs_fec, df_bs_empty],
                              labels=['AC-FEC', 'AC-EMPTY', 'BS-FEC', 'BS-EMPTY'],
                              er_rates=cfg.param.er_rates,
                              filename=filename)

    print_metrics_for_all_dfs(dfs=[df_ac_fec, df_ac_empty, df_bs_fec, df_bs_empty, df_mixall],
                              labels=['AC-FEC', 'AC-EMPTY', 'BS-FEC', 'BS-EMPTY', 'MIXALL'],
                              er_rates=cfg.param.er_rates,
                              filename=filename, node_value=-1)

    # dfs_names = ['AC-FEC', 'AC-EMPTY', 'BS-FEC', 'BS-EMPTY']
    # load_and_plot(dfs_names)
    # dfs_names = ['AC-FEC', 'AC-EMPTY', 'BS-FEC', 'BS-EMPTY', 'MIXALL']
    # load_and_plot(dfs_names, node_value=-1)


if __name__ == '__main__':
    main_run()