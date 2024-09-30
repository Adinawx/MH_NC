import numpy as np
from ns.port.fifo_store import FIFO_Store

class EpsEstimator:
    def __init__(self, cfg, env):

        self.cfg = cfg
        self.eps = None
        self.t = 0
        self.ch = 0

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
        else:
            raise ValueError('Unknown erasure estimate type')

    def stat(self):

        acks = 1-np.array(self.acks_tracker.fifo_items())
        if len(acks) > 0:
            self.eps = np.mean(acks)
        else:
            self.eps = 0

        return self.eps

    def genie(self):

        if self.t == 8:
            a=5

        # Only load the full series the first time the function is called
        if self.eps is None:
            eps = self.cfg.param.er_rate[self.ch]
            path = self.cfg.param.er_series_path
            path = path.replace('AAA', f"ch_{self.ch}")
            path = path.replace('BBB', f'{eps:.2f}')
            self.eps = np.genfromtxt(path, delimiter=',')  # full series

        # Calculate the subseries from t-RTT to t
        start = max(0, int(self.t - self.cfg.param.rtt-1))+1
        end = int(self.t)+1

        if start == end:
            return 0

        # Get the subseries from start to end
        subseries = 1-self.eps[start:end]

        # Return the mean of the subseries
        return np.mean(subseries)

