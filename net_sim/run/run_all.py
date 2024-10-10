import itertools
import os
import numpy as np
from utils.config import CFG
from utils.config_setup import Config
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


def run_all():
    cfg = Config.from_json(CFG)

    new_folder = os.path.join(cfg.param.results_folder, cfg.param.results_filename)
    if os.path.exists(new_folder):
        print("ERROR: NEW FOLDER NAME ALREADY EXISTS. CHANGE DIRECTORY TO AVOID OVERWRITE TRAINED MODEL")
        exit()
    else:
        os.makedirs(new_folder)

    rep = cfg.param.rep
    rtt_list = cfg.param.rtt

    varying_indices = cfg.param.er_var_ind  # Define the indices of the er_rates that will vary
    all_er_rates = cfg.param.er_rates

    start = cfg.param.er_var_values[0]
    stop = cfg.param.er_var_values[1]
    steps = cfg.param.er_var_values[2]
    er_rates_grid = generate_er_rates_grid(varying_indices, all_er_rates, start=start, stop=stop, steps=steps)

    eta = np.zeros([len(rtt_list), len(er_rates_grid), rep])
    mean_delay = np.zeros([len(rtt_list), len(er_rates_grid), rep])
    max_delay = np.zeros([len(rtt_list), len(er_rates_grid), rep])

    for rtt_idx, rtt in enumerate(rtt_list):

        cfg.param.rtt = rtt  # needed in the AC nodes.

        for er_idx, er_rates in enumerate(er_rates_grid):
            for r in range(rep):
                print(f"--- RTT={int(rtt)}, er_rates={er_rates}, Repetition {r + 1} ---")

                cfg.param.er_rates = er_rates

                # When reading data from a file, determine the relevant path using {r}.
                # Later: AAA=Channel index and BBB=eps.
                cfg.param.er_series_path = f"{cfg.param.project_folder}" \
                                           f"\\Data\\{cfg.param.er_type}\\AAA\\" \
                                           f"erasure_series_eps_BBB_series_{r}.csv"

                eta_1, mean_delay_1, max_delay_1 = run_1.run_1(cfg, rtt, er_rates, new_folder)

                eta[rtt_idx, er_idx, r] = eta_1
                mean_delay[rtt_idx, er_idx, r] = mean_delay_1
                max_delay[rtt_idx, er_idx, r] = max_delay_1

            # Print final mean:
            print("----Final Results----")
            print(f"RTT={rtt}")
            print(f"Channel rates: {1 - np.array(er_rates)}")
            print(f"Mean eta: {np.mean(eta[rtt_idx, er_idx, :]):.2f} +- {np.std(eta[rtt_idx, er_idx, :]):.2f}")
            print(
                f"Mean delay: {np.mean(mean_delay[rtt_idx, er_idx, :]):.2f} +- {np.std(mean_delay[rtt_idx, er_idx, :]):.2f}")
            print(
                f"Max delay: {np.mean(max_delay[rtt_idx, er_idx, :]):.2f} +- {np.std(max_delay[rtt_idx, er_idx, :]):.2f}")
            print(f"\n")

            # Save results for this RTT and er_rates configuration
            save_results(cfg, rtt, er_rates, eta[rtt_idx, :, :], mean_delay[rtt_idx, :, :], max_delay[rtt_idx, :, :])

        # Plot mean results vs er_rate: (mean over all reps)
        eta_plt = np.mean(eta[rtt_idx, :, :], axis=-1)
        mean_delay_plt = np.mean(mean_delay[rtt_idx, :, :], axis=-1)
        max_delay_plt = np.mean(max_delay[rtt_idx, :, :], axis=-1)

        er_var_num = len(varying_indices)
        channels_num = len(cfg.param.er_rates)

        if er_var_num == channels_num: # all channels are varying
            # Choose a fixed channel rate to plot - Can be changed manually.
            fixed_ind = 0
            fixed_er_rate = cfg.param.er_var_values[fixed_ind]

        elif er_var_num < channels_num:  # some channels are fixed
            fixed_ind = [i for i in range(len(cfg.param.er_rates)) if i not in varying_indices][0]
            fixed_er_rate = cfg.param.er_rates[fixed_ind]

        else:
            print("ERROR: The number of varying indices is larger than the number of channels.")
            return

        plot_results.plot_2d_vs_er_rate(cfg, rtt, er_rates_grid, eta_plt, mean_delay_plt, max_delay_plt,
                                        fix_ind=fixed_ind,
                                        fix_er_rate=fixed_er_rate)

        if er_var_num == 2:
            plot_results.plot_3d_vs_er_rates(cfg, rtt, er_rates_grid, eta_plt, mean_delay_plt, max_delay_plt)

        # TODO: Add the following plots:
        # Plot mean results vs RTT: (mean over all reps)
        # er_idx_plt = 0 # Choose a specific er_rate to plot
        # eta_plt = np.mean(eta[:, er_idx_plt, :], axis=-1)
        # mean_delay_plt = np.mean(mean_delay[:, er_idx_plt, :], axis=-1)
        # max_delay_plt = np.mean(max_delay[:, er_idx_plt, :], axis=-1)
        # plot_results.plot_2d_vs_rtt(cfg, rtt_list, er_rates_grid[er_idx_plt], eta_plt, mean_delay_plt, max_delay_plt)

        a = 5


if __name__ == '__main__':
    run_all()
