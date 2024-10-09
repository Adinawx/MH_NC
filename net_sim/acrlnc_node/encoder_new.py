from ns.port.fifo_store import FIFO_Store
import numpy as np

class Encoder:

    def __init__(self, cfg, env, t=0):

        self.cfg = cfg

        self.t = t
        self.rtt = cfg.param.rtt
        self.EOW = 2 * (self.rtt - 1)
        self.p_id = 0

        # memory holder of relevant packets. (get() when decoding happened):
        # ct = [p_id, fec_type,[w_min_t, w_max_t], ack/nack]
        self.ct_buffer = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

        # tranmsmission line of packets. (get() every time step):
        self.transmission_line = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

        self.pt_buffer = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.erasure_tracker = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

        self.p_th = 0
        self.ad = 0
        self.md = 0
        self.cnew = 0
        self.csame = 0
        self.eps = 0
        self.delta = 0

        self.fec_flag = False
        self.fec_num = 0
        self.EOW = 2 * (self.rtt)

        self.criterion = False
        self.type = ''

        self.w_min = 0
        self.w_max = -1

    def update_t(self):
        self.t += 1
        return

    def update_w_min(self):

        # Update w_min:
        # take w_min from the buffer.
        first_packet = self.pt_buffer.fifo_items()[0].nc_serial
        self.w_min = first_packet

        # Update ct_buffer:
        # if packets were decoded, update the c_t buffer.
        for ct in self.ct_buffer.fifo_items():
            if ct[1][1] < self.w_min:
                self.ct_buffer.get()
            else:
                break
        return

    def update_delta(self):

        all_ct = self.ct_buffer.fifo_items()
        self.ad = len([ct for ct in all_ct if ct[2] != 'NEW' and ct[3] == 1])
        self.md = len([ct for ct in all_ct if ct[2] == 'NEW' and ct[3] == 0])
        self.csame = len([ct for ct in all_ct if ct[2] != 'NEW' and ct[3] is None])
        self.cnew = len([ct for ct in all_ct if ct[2] == 'NEW' and ct[3] is None])

        self.delta = (self.md + self.eps * self.cnew) - (self.ad + (1 - self.eps) * self.csame) - self.p_th
        self.delta = np.round(self.delta, 4)  # Avoid numerical errors.

        if self.cfg.param.print_flag:
            print(f"ad={self.ad}, md={self.md}, csame={self.csame}, cnew={self.cnew}, delta={self.delta}")

        return self.delta

    def update_w_max(self, in_fb):

        # 1. Decoding happened:
        if self.w_max == self.w_min - 1:  # Decoding happened and w_max < w_min.
            self.w_max = self.w_min
            self.type = 'NEW'
            return

        self.type = ''

        if self.t == 8:
            a=5

        # 2.1 FEC rule:
        # end of "generation", start FEC transmission
        if self.t % self.rtt == 0 and self.t > 0:
            self.fec_num = self.cnew * self.eps #self.cnew * self.eps  # number of FEC packets
            if self.fec_num - 1 >= 0:  # Activate FEC transmission
                self.fec_flag = True
            self.fec_flag = False  # Manually terminate FEC transmission

        # Check FEC transmission:
        if self.fec_flag and self.fec_num - 1 >= 0:  # FEC transmission
            self.criterion = True
            self.type = 'FEC'
            # Reduce the number of FEC packets for the next time step:
            self.fec_num = self.fec_num - 1
            if self.fec_num - 1 < 0:
                self.fec_flag = False  # End FEC transmission
            return

        # 2.2 FB-FEC rule:
        # if in_fb[0] is not None:  # feedback received
        self.update_delta()
        self.criterion = (self.delta > 0)  # True is FB-FEC

        # 3. Set transmission type:
        if self.criterion and not self.fec_flag and self.type == '':
            self.type = 'FB-FEC'

        else:

            if self.w_max == -1:  # First packet
                self.w_max = 0
                self.type = 'NEW'

            else:
                last_packet = self.pt_buffer.fifo_items()[-1].nc_serial
                w_max_new = self.w_max + 1
                if w_max_new > last_packet:
                    self.type = 'EMPTY_FEC'
                    return

                if w_max_new-self.w_min > self.EOW:
                    self.type = 'EOW'
                    return

                self.w_max = w_max_new
                self.type = 'NEW'

                if self.w_max == self.w_min - 1:
                    self.w_max = self.w_min
                elif self.w_max < self.w_min:
                    print('ERROR: w_max < w_min')
                    return
        return

    def run(self, pt_buffer: [], in_fb: [], eps: float, t=0, print_file=None):
        '''
        :param pt_buffer: [is_arrived, [packet1, packet2, ...]]
        :param in_fb: [ack/nack, packet_id]
        :param eps: erasure probability
        :param t: time step
        :param print_file: file to print the output
        '''

        self.t = t
        self.eps = eps

        is_arrived = pt_buffer[0]
        self.pt_buffer = pt_buffer[1]

        if is_arrived:
            self.erasure_tracker.put(1)
        else:
            self.erasure_tracker.put(0)

        if len(self.pt_buffer) > 0:  # Packet buffer is not empty

            # A feedback arrived: Update the corresponding ct ack
            ack_id = in_fb[0]
            ack = in_fb[1]

            for ind_c, ct in enumerate(self.ct_buffer.fifo_items()):
                channels_num = len(self.cfg.param.er_rates)
                if ct[0] == self.p_id - int(self.rtt + channels_num):
                    self.ct_buffer.fifo_items()[ind_c][3] = ack
                    break

            # Update w_min:
            self.update_w_min()

            # Update w_max:
            self.update_w_max(in_fb)

            # Next transmission:
            ct = [self.w_min, self.w_max]

            # Update the ct_buffer:
            self.ct_buffer.put([self.p_id, ct, self.type, None])  # p_id, ct, fec/new, ack
            self.p_id += 1

        else:  # Packet buffer is empty

            # Buffer is beginning empty - unable to send anything.
            if self.w_max == -1:
                ct, type = None, None

            # Buffer is empty, but "stored" the last sent packets.
            else:
                ct = [self.w_min, self.w_max]
                self.type = 'EM_B_FEC'  # Empty buffer FEC
                # Update the ct_buffer:
                self.ct_buffer.put([self.p_id, ct, self.type, None])  # p_id, ct, fec/new, ack
                self.p_id += 1

        self.print(in_fb=in_fb, is_arrived=pt_buffer[0], print_file=print_file)

        return ct, self.type

    def print(self, in_fb, is_arrived, print_file=None):

        with open(print_file, 'a') as f:

            ack_id = in_fb[0]
            ack = in_fb[1]
            dec_id = in_fb[2]
            if ack != 0:
                ack_line = f'ACK({ack_id}, dec_id={dec_id})'
            else:
                ack_line = f'NACK({ack_id}, dec_id={dec_id})'

            if is_arrived:
                if len(self.pt_buffer) > 0:
                    arrive_line = f'Reception, Packet({self.pt_buffer.fifo_items()[-1]})'
                else:
                    arrive_line = 'Reception, Empty Buffer'
            else:
                arrive_line = 'Erasure'

            print(f'---- t = {int(self.t)} ----\n', file=f)

            print(f'Feedback IN: {ack_line}', file=f)

            print(
                f'Packet IN: {arrive_line}', file=f)

            print(f'Packet OUT: {self.type}[{self.w_min}, {self.w_max}]', file=f)

            print(
                f'eps = {self.eps:.2f}',
                f'ad = {self.ad}',
                f'md = {self.md}',
                f'cnew = {self.cnew}',
                f'csame = {self.csame}',
                f'delta = {self.delta :.2f}',
                f'criterion = {self.criterion}',
                f'fec_flag = {self.fec_flag}',
                f'fec_num = {self.fec_num :.2f}\n', file=f)

            print(f'erasure number = {np.sum(1-np.array(self.erasure_tracker.fifo_items()))}', file=f)

            # print(
                # f'ct_buffer = {self.ct_buffer.fifo_items()}',
                # f'pt_buffer = {self.pt_buffer.fifo_items()}',
                # f'erasure_tracker = {self.erasure_tracker.fifo_items()}',
                # f'transmission_line = {self.transmission_line.fifo_items()}')

        return

