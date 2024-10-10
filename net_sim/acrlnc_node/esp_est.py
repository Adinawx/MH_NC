import numpy as np
from ns.port.fifo_store import FIFO_Store

class EpsEstimator:
    def __init__(self, cfg, env):

        self.cfg = cfg
        self.genie_ = None
        self.t = 0
        self.ch = 0
        self.rtt = cfg.param.rtt

        # memory holder of relevant packets. (get() when decoding happened):
        self.acks_tracker = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

    def update_acks_tracker(self, ack):
        self.acks_tracker.put(ack)
        return

    def eps_estimate(self, t, ch):

        self.t = t
        self.ch = ch

        if self.cfg.param.er_estimate_type == 'genie':
            return self.genie()
        elif self.cfg.param.er_estimate_type == 'stat':
            return self.stat()
        elif self.cfg.param.er_estimate_type == 'stat_max':
            return self.stat_max()
        else:
            raise ValueError('Unknown erasure estimate type')

    def stat(self):

        # Old:
        # acks = 1-np.array(self.acks_tracker.fifo_items())
        # if len(acks) > 0:
        #     eps = np.mean(acks)
        # else:
        #     eps = 0

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
            eps = np.zeros(1)

        return eps

    def stat_max(self):

        # Old:
        # acks = 1-np.array(self.acks_tracker.fifo_items())
        # if len(acks) > 0:
        #     eps = np.mean(acks)
        # else:
        #     eps = np.zeros(1)
        # v = self.rtt * (1-eps)*eps  # variance for BEC
        # eps_max = eps + self.cfg.param.sigma * np.sqrt(v) / self.rtt

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

    def genie(self):

        # New - contains all forward nodes - not finished
        # # Load the full series if it has not been loaded yet
        # channels_num = len(self.cfg.param.channels)
        # if self.genie is None:
        #     for ch in range(channels_num):
        #         # Read path
        #         eps = self.cfg.param.er_rates[ch]
        #         path = self.cfg.param.er_series_path
        #         path = path.replace('AAA', f"ch_{ch}")
        #         path = path.replace('BBB', f'{eps:.2f}')
        #
        #         # Load the full series
        #         ch_genie = np.genfromtxt(path, delimiter=',')  # full series
        #
        #         # Save all the series
        #         if self.genie is None:
        #             self.genie = ch_genie
        #         else:
        #             self.genie = np.vstack((self.genie, ch_genie))
        #
        # # Calculate the subseries from t-RTT to t, with RTT aggregated delay
        # channels_delay = np.arange(channels_num) * self.cfg.param.rtt
        # start = max(0, int(self.t - channels_delay - 1)) + 1
        # end = int(self.t - channels_delay) + 1

        # Old - contains only the current forward node
        # Only load the full series the first time the function is called
        if self.genie_ is None:
            eps = self.cfg.param.er_rates[self.ch]
            path = self.cfg.param.er_series_path
            path = path.replace('AAA', f"ch_{self.ch}")
            path = path.replace('BBB', f'{eps:.2f}')
            self.genie_ = np.genfromtxt(path, delimiter=',')  # full series

        # Calculate the subseries from t-RTT to t
        start = max(0, int(self.t - self.cfg.param.rtt-1))+1
        end = int(self.t)+1

        if start == end:
            return np.zeros(1)

        # Get the subseries from start to end
        subseries = 1-self.genie_[start:end]

        # Return the mean of the subseries
        return np.mean(subseries)

