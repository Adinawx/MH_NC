import torch

class Encoder:

    def __init__(self):
        self.t = 0
        self.RTT = 0

        self.ct_buffer = None
        self.pt_buffer = None
        self.erasures_buffer = None

        self.p_th = 0
        self.ad = 0
        self.md = 0
        self.cnew = 0
        self.csame = 0
        self.eps = 0
        self.delta = 0

        self.fec_flag = False
        self.fec_num = 0
        self.EOW = 2*(self.RTT-1)

        self.criterion = False
        self.type = ''
        self.transmission_line = None

        self.w_min = 0
        self.w_max = 0

    def RTT_set(self, RTT):
        self.RTT = RTT
        return

    def update_t(self):
        self.t += 1
        return

    def update_w_min(self):
        self.w_min = self.pt_buffer[0].nc_id

    def update_eps(self):
        self.eps = torch.mean(self.erasures_buffer)
        return self.eps

    def update_delta(self):
        self.ad = len([ct for ct in self.ct_buffer if ct.fec and ct.ack])
        self.md = len([ct for ct in self.ct_buffer if not ct.fec and not ct.ack])
        self.cnew = len([ct for ct in self.ct_buffer if ct.fec and ct.ack is None])
        self.csame = len([ct for ct in self.ct_buffer if not ct.fec and ct.ack is None])
        self.delta = (self.md + self.eps * self.cnew) - (self.ad + (1-self.eps) * self.csame) - self.p_th
        return self.delta

    def update_w_max(self):

        self.type=''
        self.update_eps()
        self.update_delta()

        # end of "generation", start FEC transmission
        if self.t % self.RTT == 0:
            self.fec_num = self.cnew * self.eps  # number of FEC packets
            self.fec_flag = True
            # self.fec_flag = False  # Terminate FEC transmission

        # Check FEC transmission:
        if self.fec_flag and self.fec_num - 1 >= 0:  # FEC transmission
            self.criterion = True
            self.fec_num = self.fec_num - 1
            self.type = 'FEC'
            if self.fec_num - 1 <= 0:
                self.fec_flag = False  # End FEC transmission

        # Check FB-FEC transmission:
        self.criterion = (self.delta > 0)  # True is FB-FEC

        # Final transmission:
        if self.criterion and not self.fec_flag and self.type=='':
            self.type = 'FB-FEC'
        else:
            w_max_new = self.w_max + 1
            if w_max_new <= self.EOW:
                self.w_max = w_max_new
                self.type = 'NEW'
                self.w_max += 1
            else:
                self.type = 'EOW'

        return

    def run(self, p_buffer, c_buffer, erasures_buffer):
        self.update_t()

        self.pt_buffer = p_buffer
        self.ct_buffer = c_buffer
        self.erasures_buffer = erasures_buffer

        self.update_w_min()

        # No Packet arrived and transmission line is not empty: Send scheduled packet.
        if self.erasures_buffer[-1] == 0:
            if len(self.transmission_line) > 0:
                ct = self.transmission_line[0]
                self.type = self.transmission_line[1]
                return ct

        self.update_w_max()
        ct = [self.w_min, self.w_max]
        self.transmission_line.append([ct, self.type])

        return ct, self.type
