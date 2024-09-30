import torch

class Decoder:
    def __init__(self):
        self.t = None
        self.rtt = None
        self.dec_ind = None  # Index of the last decoded packet
        self.dof_num = None  # Number of missing Dofs at the current receiver
        self.waiting_packets_start = None  # Index of the first packet that is waiting to be decoded
        self.waiting_packets_end = None  # Index of the last packet that is waiting to be decoded
        self.received_hist = None  # Log of the received packets [t, [p_start_ind, p_end_ind]]
        self.decoded_times_hist = None  # Log of the time of decoding each packet

    def reset(self, T, rtt):
        self.t = 0
        self.rtt = rtt
        self.dec_ind = 0
        self.dof_num = 0
        self.waiting_packets_start = 0
        self.waiting_packets_end = 0
        self.received_hist = torch.zeros([T, 2])
        self.decoded_times_hist = torch.zeros(T)

    def receive(self, ct):
        if ct is not None:
            self.received_hist[self.t, :] = ct
            if ct[1] > self.dec_ind:  # w_end that is arrived is not decoded yet
                self.dof_num += 1
            self.waiting_packets_end = ct[1]

    def decode(self):
        waiting_num = self.waiting_packets_end - self.dec_ind
        if self.dof_num == waiting_num > 0:  # Decode all the waiting packets
            last_dec = self.waiting_packets_end
            self.decoded_times_hist[int(self.dec_ind): int(self.waiting_packets_end)] = self.t  # Decode with forward_tt dec_timea
            self.dof_num = 0
            self.dec_ind += waiting_num
            return last_dec
        else:
            return 0
