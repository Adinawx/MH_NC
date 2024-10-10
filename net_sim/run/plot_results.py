import os

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_2d_vs_er_rate(cfg, rtt, er_rates_grid, eta, mean_delay, max_delay, fix_ind=0, fix_er_rate=0.1):

    # Move all to numpy arrays
    eta = np.array(eta)
    mean_delay = np.array(mean_delay)
    max_delay = np.array(max_delay)
    er_rates_grid = np.array(er_rates_grid)

    # Find indices of fix_er_rate in er_rates_grid
    fix_er_rate_idx = np.where(er_rates_grid[:, fix_ind] == fix_er_rate)[0]

    # The varying erasure rate index
    all_varying_indices = cfg.param.er_var_ind
    vary_ind = [i for i in all_varying_indices if i != fix_ind][0]

    # Extract the varying erasure rate values
    sub_er_grid = er_rates_grid[fix_er_rate_idx]
    x_axis = sub_er_grid[:, vary_ind]
    eta_values = eta[fix_er_rate_idx]
    mean_delay_values = mean_delay[fix_er_rate_idx]
    max_delay_values = max_delay[fix_er_rate_idx]

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(x_axis, eta_values, marker='o', label='eta')
    ax[0].set_ylabel('Normalized Throughput')
    ax[0].set_xlabel(f'Erasure Rate of node: {vary_ind + 1}')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(x_axis, mean_delay_values, marker='o', label='Mean Delay')
    ax[1].set_ylabel('Mean Delay')
    ax[1].set_xlabel(f'Erasure Rate of node: {vary_ind + 1}')
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(x_axis, max_delay_values, marker='o', label='Max Delay')
    ax[2].set_ylabel('Max Delay')
    ax[2].set_xlabel(f'Erasure Rate of node: {vary_ind + 1}')
    ax[2].grid(True)
    ax[2].legend()

    # Set the title for the figure
    fig.suptitle(f'RTT={int(rtt)}, Fixed Erasure Rate of node: {fix_ind + 1} = {fix_er_rate}')

    plt.show()

    save_folder = os.path.join(cfg.param.results_folder, cfg.param.results_filename)
    fig.savefig(os.path.join(save_folder, f'2D_RTT={int(rtt)}_Fixed_ER={fix_er_rate}_Node={fix_ind + 1}.png'))


def plot_3d_vs_er_rates(cfg, rtt, er_rates_grid, eta, mean_delay, max_delay):
    """
    Plot a 3D graph for the varying erasure rates and the result metrics (eta, mean_delay, max_delay).

    Args:
        rtt: The RTT value for which the results are plotted.
        er_rates_grid: List of er_rates combinations (grid) used in the simulation.
        eta: Eta values for all er_rates combinations.
        mean_delay: Mean delay values for all er_rates combinations.
        max_delay: Max delay values for all er_rates combinations.
    """

    # Move all to numpy arrays
    eta = np.array(eta)
    mean_delay = np.array(mean_delay)
    max_delay = np.array(max_delay)
    er_rates_grid = np.array(er_rates_grid)

    # Extract the two varying erasure rates
    all_varying_indices = cfg.param.er_var_ind
    er_rate_1 = er_rates_grid[:, all_varying_indices[0]]
    er_rate_2 = er_rates_grid[:, all_varying_indices[1]]

    save_folder = os.path.join(cfg.param.results_folder, cfg.param.results_filename)

    # eta
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(er_rate_1, er_rate_2, eta.squeeze(), cmap='viridis', edgecolor='none')
    ax.set_xlabel('Erasure Rate 1')
    ax.set_ylabel('Erasure Rate 2')
    ax.set_zlabel('Normalized Throughput')
    fig.savefig(os.path.join(save_folder, f'eta_3D_RTT={int(rtt)}.png'))

    # mean delay
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(er_rate_1, er_rate_2, mean_delay.squeeze(), cmap='viridis', edgecolor='none')
    ax.set_xlabel('Erasure Rate 1')
    ax.set_ylabel('Erasure Rate 2')
    ax.set_zlabel('Mean Delay [Slots]')
    fig.savefig(os.path.join(save_folder, f'mean_delay_3D_RTT={int(rtt)}.png'))

    # max delay
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(er_rate_1, er_rate_2, max_delay.squeeze(), cmap='viridis', edgecolor='none')
    ax.set_xlabel('Erasure Rate 1')
    ax.set_ylabel('Erasure Rate 2')
    ax.set_zlabel('Max Delay [Slots]')
    fig.savefig(os.path.join(save_folder, f'max_delay_3D_RTT={int(rtt)}.png'))

    # Show plot
    plt.show()
