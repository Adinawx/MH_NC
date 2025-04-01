import numpy as np
import os
np.random.seed(44)


def save_bernoulli_series(folder_path, N, T, p):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    for i in range(1, N + 1):
        # Generate a binary series of length T with probability p
        series = np.random.binomial(1, p-0.1, T)

        # Create the file path
        file_path = os.path.join(folder_path, f"series_{i}.txt")

        # Save the series to a file
        np.savetxt(file_path, series, fmt='%d')

def upsample_(N, series, factor):
    os.makedirs(folder_path, exist_ok=True)

    for i in range(1, N + 1):
        # Generate a binary series of length T with probability p
        series_long = np.tile(series, factor)

        # Create the file path
        file_path = os.path.join(folder_path, f"series_{i}.txt")

        # Save the series to a file
        np.savetxt(file_path, series_long, fmt='%d')

# Example usage
folder_path = r"C:\Users\adina\Technion\Research\MH_Project\Code\Data\upsample"  # Folder to save the series
N = 100  # Number of series
T = 10000  # Length of each series

for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    folder_path_ = os.path.join(folder_path, f"rate_{p}")

    # Generate and save the series
    # save_bernoulli_series(folder_path_, N, T, p)

    series = np.zeros(100)
    series[0] = 1
    factor = 100
    upsample_(N, series, factor)


print(f"Series saved in folder: {folder_path}")
