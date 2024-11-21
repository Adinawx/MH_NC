import numpy as np
import os
import matplotlib.pyplot as plt


def generate_erasure_series(length, erasure_prob):
    """
    Generates a single erasure series for a binary erasure channel.

    Parameters:
    length (int): Length of the erasure series.
    erasure_prob (float): Probability of erasure (0 <= erasure_prob <= 1).

    Returns:
    np.ndarray: The generated erasure series, where 0 denotes erasure and 1 denotes success.
    """
    # Generate random numbers and compare to erasure probability
    random_values = np.random.rand(length)
    erasure_series = np.where(random_values < erasure_prob, 0, 1)
    return erasure_series


def create_and_save_erasure_series_for_eps(r, length, eps_list, main_path, channels_num):
    """
    Generates R erasure series for varying epsilons and saves them in different subfolders under the main path.

    Parameters:
    r (int): Number of erasure series to generate per epsilon.
    length (int): Length of each erasure series.
    eps_list (list of float): List of erasure probabilities (eps_hist values).
    main_path (str): Main directory path ('BEC'), containing subfolders where the series will be saved.

    Returns:
    None
    """
    # Create the main path if it doesn't exist
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    # Loop over 5 channels (ch_0, ch_1, ..., ch_<channels_num>)
    for channel_num in range(channels_num):
        channel_path = os.path.join(main_path, f"ch_{channel_num}")

        # Ensure the subfolder exists
        if not os.path.exists(channel_path):
            os.makedirs(channel_path)

        for eps in eps_list:
            for i in range(r):
                # Generate a single erasure series
                series = generate_erasure_series(length, eps)

                # Create a filename based on the epsilon value and series index
                filename = os.path.join(channel_path, f"erasure_series_eps_{eps:.2f}_series_{i}.csv")

                # Save the series to a CSV file
                np.savetxt(filename, series, fmt='%d', delimiter=',')
                print(f"Series {i + 1} for eps_hist={eps} saved in {channel_path}")


def read_erasure_series_for_eps(eps, r, channel_num, main_path):
    """
    Reads R erasure series from a specific channel's subfolder for a given epsilon.

    Parameters:
    eps_hist (float): The erasure probability (epsilon) to load the series for.
    r (int): The number of series to read.
    channel_num (int): The channel number (1-5) corresponding to the subfolder.
    main_path (str): Main directory path ('BEC'), containing subfolders where the series are stored.

    Returns:
    list of np.ndarray: A list of R erasure series read from the files.
    """
    channel_path = os.path.join(main_path, f"ch_{channel_num}")
    series_list = []

    for i in range(r):
        # Create the filename based on the epsilon value and series index
        filename = os.path.join(channel_path, f"erasure_series_eps_{eps:.2f}_series_{i}.csv")

        if os.path.exists(filename):
            # Load the series from the file
            series = np.loadtxt(filename, delimiter=',').astype(int)
            series_list.append(series)
            print(f"Series {i} for eps_hist={eps} loaded from {filename}")
        else:
            print(f"File {filename} not found.")

    return series_list


def plot_erasure_series(series_list, eps):
    """
    Plots the erasure series and displays the amount of erasures (zeros) in the legend.

    Parameters:
    series_list (list of np.ndarray): The erasure series to plot.
    eps_hist (float): The epsilon value for the series being plotted.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    for i, series in enumerate(series_list, 1):
        # Count the number of erasures (zeros)
        num_erasures = np.sum(series == 0)

        # Plot the series with markers
        plt.plot(series, label=f"Series {i}, Erasures: {num_erasures}")

    # Set plot title and labels
    plt.title(f"Erasure Series for eps_hist={eps}")
    plt.xlabel("Timestep")
    plt.ylabel("Value (0=Erasure, 1=Success)")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


# Example usage:
# Define the parameters
r = 1  # Number of series to generate per epsilon
channels_num = 1  # Number of channels to generate series for
length = 3000  # Length of each series
eps_list = np.arange(0, 1.1, 0.1)  # Different erasure probabilities
main_path = "C:\\Users\\adina\\Technion\\Research\\MH_Project\\Code\\Data\\BEC"  # Main directory to store the series files

# Create and save erasure series in different subfolders under the 'BEC' folder
create_and_save_erasure_series_for_eps(r, length, eps_list, main_path, channels_num)

# # Now, let's read r series for a specific epsilon (e.g., eps_hist = 0.3) from channel 2
# eps = 0.3  # Epsilon value to load series for
# r_to_read = 3  # Number of series to read
# channel_num = 2  # Specify which channel's folder to read from (ch_2)
#
# # Read the series from the specific channel folder
# series_list = read_erasure_series_for_eps(eps, r_to_read, channel_num, main_path)
#
# # Plot the loaded series with the number of erasures in the legend
# if series_list:
#     plot_erasure_series(series_list, eps)

