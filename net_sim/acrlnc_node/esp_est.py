import numpy as np
from ns.port.fifo_store import FIFO_Store

class EpsEstimator:
    def __init__(self, cfg, env):

        self.cfg = cfg
        self.genie_folder = None
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

        acks = 1-np.array(self.acks_tracker.fifo_items())
        if len(acks) > 0:
            eps = np.mean(acks)
        else:
            eps = 0

        return eps

    def stat_max(self):

        acks = 1-np.array(self.acks_tracker.fifo_items())
        if len(acks) > 0:
            eps = np.mean(acks)
        else:
            eps = 0

        v = self.rtt * (1-eps)*eps  # variance for BEC
        eps_max = eps + self.cfg.param.sigma * np.sqrt(v) / self.rtt

        return eps_max

    def genie(self):

        if self.t == 8:
            a=5

        # Only load the full series the first time the function is called
        if self.genie_folder is None:
            eps = self.cfg.param.er_rates[self.ch]
            path = self.cfg.param.er_series_path
            path = path.replace('AAA', f"ch_{self.ch}")
            path = path.replace('BBB', f'{eps:.2f}')
            self.genie_folder = np.genfromtxt(path, delimiter=',')  # full series

        # Calculate the subseries from t-RTT to t
        start = max(0, int(self.t - self.cfg.param.rtt-1))+1
        end = int(self.t)+1

        if start == end:
            return 0

        # Get the subseries from start to end
        subseries = 1-self.genie_folder[start:end]

        # Return the mean of the subseries
        return np.mean(subseries)

