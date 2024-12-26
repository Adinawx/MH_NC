import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def generate_erasure_series_BEC(length, erasure_prob):
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


def generate_erasure_series_GE(length, p, q, erasure_prob_good, erasure_prob_bad):
    """
    Generates an erasure series for a Gilbert-Elliot channel.

    Parameters:
    length (int): Length of the erasure series.
    p (float): Probability of transitioning from "good" to "bad" state.
    q (float): Probability of transitioning from "bad" to "good" state.
    erasure_prob_good (float): Erasure probability in the "good" state.
    erasure_prob_bad (float): Erasure probability in the "bad" state.

    Returns:
    np.ndarray: The generated erasure series, where 0 denotes erasure and 1 denotes success.
    """
    # Initialize the series and the starting state (0 for good, 1 for bad)
    series = np.zeros(length, dtype=int)
    state = 0  # Start in the "good" state

    for i in range(length):
        # Generate the erasure based on the current state
        if state == 0:  # Good state
            series[i] = 1 if np.random.rand() > erasure_prob_good else 0
            # Transition to bad state with probability p
            if np.random.rand() < p:
                state = 1
        else:  # Bad state
            series[i] = 1 if np.random.rand() > erasure_prob_bad else 0
            # Transition to good state with probability q
            if np.random.rand() < q:
                state = 0

    return series


def create_and_save_erasure_series(r, length, eps_list, main_path, channels_num, channel_type, **kwargs):
    """
    Generates R erasure series for varying epsilons and saves them in different subfolders under the main path.
    Allows selection between BEC and Gilbert-Elliot channels.

    Parameters:
    r (int): Number of erasure series to generate per epsilon.
    length (int): Length of each erasure series.
    eps_list (list of float): List of erasure probabilities (eps values for BEC, reference for GE).
    main_path (str): Main directory path ('BEC' or 'GilbertElliot'), containing subfolders where the series will be saved.
    channels_num (int): Number of channels to create.
    channel_type (str): Type of channel ('BEC' or 'GE').
    **kwargs: Additional parameters for Gilbert-Elliot channel (p, q, erasure_prob_good, erasure_prob_bad).

    Returns:
    None
    """
    # Create the main path if it doesn't exist
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    # Loop over the channels (e.g., ch_0, ch_1, ...)
    for channel_num in range(channels_num):
        channel_path = os.path.join(main_path, f"ch_{channel_num}")

        # Ensure the subfolder exists
        if not os.path.exists(channel_path):
            os.makedirs(channel_path)

        for eps in eps_list:
            for i in range(r):
                # Generate the erasure series based on channel type
                if channel_type == "BEC":
                    series = generate_erasure_series_BEC(length, eps)
                elif channel_type == "GE":
                    series = generate_erasure_series_GE(length, **kwargs)
                else:
                    raise ValueError(f"Unsupported channel type: {channel_type}")

                # Create a filename based on the epsilon value and series index
                filename = os.path.join(channel_path, f"{channel_type}_series_eps_{eps:.2f}_series_{i}.csv")

                # Save the series to a CSV file
                np.savetxt(filename, series, fmt='%d', delimiter=',')
                print(f"{channel_type} Series {i + 1} for eps={eps} saved in {channel_path}")


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
        filename = os.path.join(channel_path, f"GE_series_eps_{eps:.2f}_series_{i}.csv")

        if os.path.exists(filename):
            # Load the series from the file
            series = np.loadtxt(filename, delimiter=',').astype(int)
            series_list.append(series)
            print(f"Series {i} for eps_hist={eps} loaded from {filename}")
        else:
            print(f"File {filename} not found.")

    return series_list


def plot_erasure_series(series, eps):
    """
    Plots the erasure series and displays the amount of erasures (zeros) in the legend.

    Parameters:
    series_list (list of np.ndarray): The erasure series to plot.
    eps_hist (float): The epsilon value for the series being plotted.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    # Count the number of erasures (zeros)
    num_erasures = np.sum(series == 0)

    bursts = np.diff(series)
    burts_indices = np.where(bursts != 0)[0]
    burst_lengths = np.diff(burts_indices)
    mean_burst_length = np.mean(burst_lengths)
    std_burst_length = np.std(burst_lengths)

    # Plot the series with markers
    plt.plot(series[0: 1500])

    # Set plot title and labels
    plt.title(f"")
    plt.xlabel("Timestep")
    plt.ylabel("Value (0=Erasure, 1=Success)")
    plt.grid(True)

    # Show the plot
    plt.show()


# Parameters for BEC
r = 20  # Number of series to generate per epsilon
channels_num = 5  # Number of channels to generate series for
length = 50000  # Length of each series
channel_type = "GE"

# Generate and save BEC series
if channel_type == "BEC":
    eps_list = np.arange(0, 1.1, 0.1)  # Different erasure probabilities
    main_path = "C:\\Users\\adina\\Technion\\Research\\MH_Project\\Code\\Data\\BEC"
    create_and_save_erasure_series(r, length, eps_list, main_path, channels_num, channel_type="BEC")

elif channel_type == "GE":

    # Parameters for Gilbert-Elliot channel
    p = 0.01  # Probability of transitioning from "good" to "bad"
    q = 0.02  # Probability of transitioning from "bad" to "good"
    erasure_prob_good = 0.1  # Erasure probability in the "good" state
    erasure_prob_bad = 1  # Erasure probability in the "bad" state

    stat_good = q/(p+q)
    stat_bad = p/(p+q)
    erasure_prob = stat_good*erasure_prob_good + stat_bad*erasure_prob_bad
    print(f"Stationary distribution: good={stat_good:.2f}, bad={stat_bad:.2f}")
    print(f"Erasure probability: {erasure_prob:.2f}")
    print("Average burst length: ", 1/q)
    eps_list = [np.round(erasure_prob, 2)]

    main_path = f"C:\\Users\\adina\\Technion\\Research\\MH_Project\\Code\\Data\\GE_p_{p}_q_{q}_g_{erasure_prob_good}_b_{erasure_prob_bad}"

    # Generate and save GE series
    # create_and_save_erasure_series(
    #     r,
    #     length,
    #     eps_list,
    #     main_path,
    #     channels_num,
    #     channel_type="GE",
    #     p=p,
    #     q=q,
    #     erasure_prob_good=erasure_prob_good,
    #     erasure_prob_bad=erasure_prob_bad,
    # )

    # Read and plot GE series
    series = read_erasure_series_for_eps(eps_list[0], r, 0, main_path)
    series = series[0]
    plot_erasure_series(series, 0.4)

