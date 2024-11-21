import numpy as np
from ns.port.fifo_store import FIFO_Store


class EpsEstimator:
    def __init__(self, cfg, env):

        self.cfg = cfg
        self.genie_helper = None
        self.t = 0
        self.ch = 0
        self.rtt = cfg.param.rtt

        # memory holder of relevant packets. (get() when decoding happened):
        self.acks_tracker = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

    def update_acks_tracker(self, ack):
        self.acks_tracker.put(ack)
        return

    def eps_estimate(self, t, ch, est_type='stat', *args):

        self.t = t
        self.ch = ch

        if est_type == 'genie':
            return self.genie(*args)
        elif est_type == 'stat':
            return self.stat()
        elif est_type == 'stat_max':
            return self.stat_max()
        else:
            raise ValueError('Unknown erasure estimate type')

    def stat(self):

        # New - contains all forward nodes
        acks = 1 - np.array(self.acks_tracker.fifo_items(), dtype=object)
        if len(acks) > 0:
            max_len = max(arr.shape[0] for arr in acks)  # Find the longest array
            # Stack arrays with NaN padding to match the max length
            padded = np.full((len(acks), max_len), np.nan)  # Create an empty array filled with NaN
            for i, arr in enumerate(acks):  # Fill in the values
                padded[i, :arr.shape[0]] = arr

            # Compute the mean across each position, ignoring NaNs
            eps = np.nanmean(padded, axis=0)
        else:
            eps = [0.5]

        return eps

    def stat_max(self):

        # New - contains all forward nodes
        acks = 1 - np.array(self.acks_tracker.fifo_items(), dtype=object)
        if len(acks) > 0:
            max_len = max(arr.shape[0] for arr in acks)  # Find the longest array
            # Stack arrays with NaN padding to match the max length
            padded = np.full((len(acks), max_len), np.nan)  # Create an empty array filled with NaN
            for i, arr in enumerate(acks):  # Fill in the values
                padded[i, :arr.shape[0]] = arr

            # Compute the mean across each position, ignoring NaNs
            all_eps_mean = np.nanmean(padded, axis=0)
            v = self.rtt * (1 - all_eps_mean) * all_eps_mean  # variance for BEC
            eps_max = all_eps_mean + self.cfg.param.sigma * np.sqrt(v) / self.rtt
        else:
            eps_max = np.zeros(1)

        return eps_max

    def genie_old(self):

        # New - contains all forward nodes - not finished
        # Initialization - Load the full series if it has not been loaded yet
        channels_num = len(self.cfg.param.er_rates)
        if self.genie_helper is None:
            for ch in range(channels_num):
                # Read path
                eps = self.cfg.param.er_rates[ch]
                path = self.cfg.param.er_series_path
                path = path.replace('AAA', f"ch_{ch}")
                path = path.replace('BBB', f'{eps:.2f}')

                # Load current series
                ch_genie = np.genfromtxt(path, delimiter=',')  # full series

                # Save all the series
                if self.genie_helper is None:
                    self.genie_helper = ch_genie
                else:
                    self.genie_helper = np.vstack((self.genie_helper, ch_genie))

        # Calculate the subseries from t-RTT to t, with RTT aggregated delay
        ch_in_delay = [int((n + 1) * (self.cfg.param.rtt / 2 + 1)) for n in
                       range(channels_num - self.ch)]  # Each channel has its initial delay.
        starts = [max(0, int(self.t - ch_in_delay[ch] - 1)) + 1 for ch in range(channels_num - self.ch)]
        ends = [int(self.t) + 1] * (channels_num - self.ch)
        subseries = [self.genie_helper[ch, starts[ch]:ends[ch]] for ch in range(channels_num - self.ch)]

        # if start == end:
        #     return np.zeros(1)
        #
        # # Get the subseries from start to end
        # subseries = 1-self.genie_helper[self.ch, start:end]

        # New - contains all forward nodes
        acks = 1 - subseries
        if len(acks) > 0:
            max_len = max(arr.shape[0] for arr in acks)  # Find the longest array
            # Stack arrays with NaN padding to match the max length
            padded = np.full((len(acks), max_len), np.nan)  # Create an empty array filled with NaN
            for i, arr in enumerate(acks):  # Fill in the values
                padded[i, :arr.shape[0]] = arr

            # Compute the mean across each position, ignoring NaNs
            eps = np.nanmean(padded, axis=0)
        else:
            eps = np.zeros(1)

        # rtt = self.cfg.param.rtt
        # arr = self.genie[0, :]
        # arr_truncated = arr[:len(arr) - (len(arr) % rtt)]
        # reshaped_arr = arr_truncated.reshape(-1, rtt)
        # fecs_0 = reshaped_arr.mean(axis=1)

        # #####
        # # Old - contains only the current forward node
        # # Only load the full series the first time the function is called
        # if self.genie_ is None:
        #     eps_hist = self.cfg.param.er_rates[self.ch]
        #     path = self.cfg.param.er_series_path
        #     path = path.replace('AAA', f"ch_{self.ch}")
        #     path = path.replace('BBB', f'{eps_hist:.2f}')
        #     self.genie_ = np.genfromtxt(path, delimiter=',')  # full series
        #
        # # Calculate the subseries from t-RTT to t
        # start = max(0, int(self.t - self.cfg.param.rtt-1))+1
        # end = int(self.t)+1
        #
        # if start == end:
        #     return np.zeros(1)
        #
        # # Get the subseries from start to end
        # subseries = 1-self.genie_[start:end]

        # Return the mean of the subseries
        return np.mean(subseries)

    def genie(self, win_length=None, empty_indexes=None):

        # 0. Initialization - Load the full series if it has not been loaded yet
        channels_num = len(self.cfg.param.er_rates)
        if self.genie_helper is None:
            for ch in range(channels_num):
                # Read path
                eps = self.cfg.param.er_rates[ch]
                path = self.cfg.param.er_series_path
                path = path.replace('AAA', f"ch_{ch}")
                path = path.replace('BBB', f'{eps:.2f}')

                # Load current series
                ch_genie = np.genfromtxt(path, delimiter=',')  # full series

                # Save all the series
                if self.genie_helper is None:
                    self.genie_helper = ch_genie
                else:
                    self.genie_helper = np.vstack((self.genie_helper, ch_genie))

        # Iterate through each time step
        eps_mean = []
        for ch in range(self.ch, channels_num):
            in_delay = int((ch + 1) * (self.cfg.param.rtt / 2)) -1 - ch*2

            # If current time step is greater than or equal to the delay for this series
            if self.t >= 0: #in_delay:

                if win_length is None or win_length == 0:
                    win_length = self.cfg.param.rtt + 2

                start = max(0, int(self.t - win_length))
                end = int(self.t) #+ 1

                # cut the first element of the series due to the +1 inherent delay of the system.
                if self.genie_helper.ndim > 1:
                    series = self.genie_helper[ch, in_delay:].squeeze()
                else:
                    series = self.genie_helper[in_delay:]

                mask = np.ones(len(series), dtype=bool)  # Create a mask of True
                if len(empty_indexes) > 0:
                    mask[np.array(empty_indexes)] = False  # Mark indices in empty_index as False

                if end < len(series):
                    eps_mean.append(
                        np.mean(1-series[start:end][mask[start:end]]))
                else:
                    print("Error: Genie Index out of bounds")

        return eps_mean
