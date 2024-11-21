from ns.port.fifo_store import FIFO_Store
import numpy as np
from .esp_est import EpsEstimator


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
        # self.eps = 0
        self.delta = [0, 0]  # [delta_mean, delta_max]

        self.fec_num = 0
        self.fec_flag = False
        self.EOW = 2 * (self.rtt)
        self.bls_num = 0
        self.bls_flag = False
        self.bls_max = 0

        self.criterion = False
        self.type = ''

        self.w_min = 0
        self.w_max = -1

        self.eps_est = EpsEstimator(cfg=cfg, env=env)  # Estimator of eps_hist.
        self.eps_hist = []
        self.eps_mean = []
        self.eps_max = []
        self.empty_indx = []  # for the genie estimation - remember when transmitted empty and ignore it in the eps mean.

        self.curr_ch = 0  # channel index

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
        ct_temp = self.ct_buffer.fifo_items().copy()
        for ct in ct_temp:
            if ct[1][1] is not None: ## TRY 2
                if ct[1][1] < self.w_min:
                    self.ct_buffer.get()
                else:
                    break
        return

    def update_delta(self):

        all_ct = self.ct_buffer.fifo_items()
        self.ad = len([ct for ct in all_ct if(ct[2] != 'NEW' and 'EMPTY' not in ct[2]) and ct[3] == 1])
        self.md = len([ct for ct in all_ct if ct[2] == 'NEW' and ct[3] == 0])
        self.csame = len([ct for ct in all_ct if (ct[2] != 'NEW' and 'EMPTY' not in ct[2]) and ct[3] is None])
        self.cnew = len([ct for ct in all_ct if (ct[2] == 'NEW') and ct[3] is None])

        # # New code:
        # eps_hist pf the current channel (index 0 at eps_ list):
        eps_mean = self.eps_mean[0]
        eps_max = self.eps_max[0]

        # Calculate delta for mean and max eps_hist:
        delta_mean = (self.md + eps_mean * self.cnew) - (self.ad + (1 - eps_mean) * self.csame) - self.p_th
        delta_mean = np.round(delta_mean, 4)  # Avoid numerical errors.

        delta_max = (self.md + eps_max * self.cnew) - (self.ad + (1 - eps_max) * self.csame) - self.p_th
        delta_max = np.round(delta_max, 4)  # Avoid numerical errors.

        self.delta = [delta_mean, delta_max]

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

        # 2.1 FEC rule:
        # end of "generation", start FEC transmission
        eps_ = self.eps_mean[0]

        if self.cfg.param.er_estimate_type == 'stat_max':
            eps_ = self.eps_max[0]

        self.update_delta()  # anyway update so c_new will be accurate for a-priori FEC

        ######## Option 1: FEC by Time. Dis-Activate in main the other option ########
        if self.t % self.rtt == 0 and self.t > 0:
            self.fec_num = (self.cnew * eps_)  # self.rtt * eps_  #self.cnew * self.eps_hist  # number of FEC packets
            print(f"---fec_num={self.fec_num}, eps={eps_}, cnew={self.cnew}---")
            if self.fec_num - 1 >= 0:  # Activate FEC transmission
                self.fec_flag = True
            # self.fec_flag = False  # Manually terminate FEC transmission
            else:
                self.fec_flag = False
        # Check FEC transmission:
        if self.fec_flag and self.fec_num - 1 >= 0:  # FEC transmission
            self.criterion = True
            self.type = 'FEC'
            # Reduce the number of FEC packets for the next time step:
            self.fec_num = self.fec_num - 1

            # End FEC transmission
            if self.fec_num - 1 < 0:
                self.fec_flag = False

                # # BLS-FEC transmission:
                if self.cfg.param.empty_space_flag:
                    if len(self.eps_mean) > 1:
                        self.bls_flag = True
                        self.bls_max = self.t
                        # BLS distributed:
                        # n = np.round(np.sum(self.rtt * self.eps_mean[1:]))
                        # self.bls_num = np.round(
                        #     np.linspace(self.rtt/(1+n), self.rtt-self.rtt/(1+n), int(n))
                        # )
                        # BLS all in one stream
                        self.bls_num = np.round(np.sum((self.rtt) * self.eps_mean[1:]))
            return
        ##########################################################

        ######## Option 2: FEC by packets. Activate in main as well ########
        # if self.fec_flag == 'Start FEC Transmit':
        #     self.fec_num = (self.cnew * eps_)  # self.rtt * eps_  #self.cnew * self.eps_hist  # number of FEC packets
        #     if self.fec_num - 1 >= 0:  # Activate FEC transmission
        #         self.fec_flag = 'FEC Transmit'
        #     #     self.fec_flag = 'End FEC Transmit'  # Manually terminate FEC transmission
        #     else:
        #         self.fec_flag = 'End FEC Transmit'
        #
        # # Check FEC transmission:
        # if self.fec_flag == 'FEC Transmit' and self.fec_num - 1 >= 0:  # FEC transmission
        #     self.criterion = True
        #     self.type = 'FEC'
        #     # Reduce the number of FEC packets for the next time step:
        #     self.fec_num = self.fec_num - 1
        #
        #     # End FEC transmission
        #     if self.fec_num - 1 < 0:
        #         self.fec_flag = 'End FEC Transmit'
        #
        #         # # BLS-FEC transmission:
        #         if self.cfg.param.empty_space_flag:
        #             if len(self.eps_mean) > 1:
        #                 self.bls_flag = True
        #                 self.bls_max = self.t
        #                 # BLS distributed:
        #                 # n = np.round(np.sum(self.rtt * self.eps_mean[1:]))
        #                 # self.bls_num = np.round(
        #                 #     np.linspace(self.rtt/(1+n), self.rtt-self.rtt/(1+n), int(n))
        #                 # )
        #                 # BLS all in one stream
        #                 self.bls_num = np.round(np.sum((self.rtt) * self.eps_mean[1:]))
        #     return
        ##########################################################

        # 2.2 Blank Spaces - Reduce Max Delay?
        if self.bls_flag:  # for stream
        # if self.bls_flag and len(self.bls_num) > 0:  # for distributed BLS

            # # # BLS distributed:
            # if (self.t - self.bls_max) in self.bls_num:
            #     self.criterion = True
            #     self.type = 'BLS-FEC'
            #     # Debug: Transmit Nothing:
            #     # self.type = 'EMPTY-BLS'
            #     return

            # # BLS all in one stream
            if (self.t - self.bls_max) <= self.bls_num:

                delta_hat = 1 / (self.bls_num * (1 - self.eps_mean[0]))
                r_bn = max(self.cfg.param.er_rates[self.curr_ch:])
                criterion = (delta_hat <= r_bn)  # Continue bls transmission.
                if criterion:
                    self.type = 'EMPTY-BLS'
                    return

                else:
                    self.bls_flag = False
                    self.bls_num = 0
                    self.bls_max = 0

                # Stream - old:
                # self.type = 'EMPTY-BLS'
                # # self.criterion = True
                # # self.type = 'BLS-FEC'
                # return

            # End BLS-FEC transmission:
            if (self.t - self.bls_max) > np.max(self.bls_num):
                self.bls_flag = False
                self.bls_num = 0
                self.bls_max = 0

        # 2.2 Blank Spaces:
        # indication to start a BLS-FEC transmission.
        # if self.bls_flag:  # and self.bls_num > self.rtt/2:
        # # if self.bls_flag and self.bls_num > self.rtt/2:
        #
        #     # Limit the number of blank spaces:
        #     if self.bls_num > self.cfg.param.COST:
        #         self.bls_num = self.cfg.param.COST
        #
        #     # If there are enough blank spaces, start BLS-FEC transmission:
        #     if self.bls_num <= self.cfg.param.COST and self.bls_num - 1 >= 0:
        #         # Option 1: Decide by criterion:
        #         criterion_max = (self.delta[0] > 0)  # Calculate criterion with eps_max
        #         if criterion_max:
        #             self.criterion = True
        #             self.type = 'BLS-FEC'
        #             self.bls_num -= 1
        #             return
        #         else:
        #             self.type = 'EMPTY-BLS'
        #             self.bls_num -= 1
        #             return
        #     else:
        #         self.criterion = False
        #         self.bls_flag = False

        # 2.3 FB-FEC rule:

        if in_fb[0] is not None:  # Feedback is arrived
            self.criterion = (self.delta[0] > 0)  # True is FB-FEC

        # Optional: Use stat_max for criterion:
        if self.cfg.param.er_estimate_type == 'stat_max':
            self.criterion = (self.delta[1] > 0)  # True is FB-FEC

        # # Set transmission type:
        if self.criterion and self.type == '':
            self.type = 'FB-FEC'
            return

        # 3. ELSE
        if self.w_max == -1:  # First packet
            self.w_max = 0
            self.type = 'NEW'

        else:
            last_packet = self.pt_buffer.fifo_items()[-1].nc_serial
            w_max_new = self.w_max + 1
            if w_max_new > last_packet:
                self.type = 'EMPTY_BUFFER'
                return

            if w_max_new - self.w_min > self.EOW:
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

    def epsilon_estimation(self, fb_packet, in_cur_fb):

        # Update the acks tracker.
        if in_cur_fb[1] is not None:  # If there is a feedback.
            all_acks = np.array(fb_packet.nc_header)
            self.eps_est.update_acks_tracker(ack=all_acks)

        if fb_packet is None:
            self.curr_ch = 0
        else:
            self.curr_ch = len(self.cfg.param.er_rates) if fb_packet.src == 'd_fb' else \
                int(fb_packet.src[-1]) - 1

        est_type = self.cfg.param.er_estimate_type
        if est_type == 'genie':
            win_length = min(len(self.ct_buffer.fifo_items()), self.rtt+1)
            eps_mean = self.eps_est.eps_estimate(self.t, self.curr_ch, 'genie', win_length, self.empty_indx)
            eps_max = self.eps_est.eps_estimate(self.t, self.curr_ch, 'genie', win_length, self.empty_indx)
        else:
            eps_mean = self.eps_est.eps_estimate(t=self.t, ch=self.curr_ch, est_type='stat')
            eps_max = self.eps_est.eps_estimate(t=self.t, ch=self.curr_ch, est_type='stat_max')

        if self.cfg.param.print_flag:
            print(f'eps_hist mean: {eps_mean[0]:.2f}')
            print(f'eps_hist max: {eps_max[0]:.2f}')

        eps = [eps_mean, eps_max]
        self.eps_mean = eps[0].tolist() if isinstance(eps[0], np.ndarray) else eps[0]
        self.eps_max = eps[1].tolist() if isinstance(eps[1], np.ndarray) else eps[1]

        return

    def run(self, pt_buffer: [], in_fb: [], fb_packet=None, t=0, print_file=None):

        self.t = t

        is_arrived = pt_buffer[0]
        self.pt_buffer = pt_buffer[1]

        if is_arrived:
            self.erasure_tracker.put(1)
        else:
            self.erasure_tracker.put(0)

        if len(self.pt_buffer) > 0:  # Packet buffer is not empty

            # Old version - probably can delete - Activate for FEC by time
            # last_packet = self.pt_buffer.fifo_items()[-1]
            # if last_packet.src == 's_ff':
            #     if self.t % self.rtt == 0 and self.t > 0:
            #         self.fec_flag = 'Start FEC Transmit'
            # else:
            #     if last_packet.fec_type == 'FEC' or (last_packet.nc_header[0][0] % self.rtt == 1):
            #         self.fec_flag = 'FEC Receive'
            #     elif last_packet.fec_type != 'FEC' and self.fec_flag == 'FEC Receive':
            #         self.fec_flag = 'Start FEC Transmit'

            #  New version - Activate for FEC by time
            # last_packet = self.pt_buffer.fifo_items()[-1]
            # if self.fec_flag != 'Start FEC Transmit':
            #     if last_packet.src == 's_ff':
            #         if self.t % self.rtt == 0 and self.t > 0:
            #             self.fec_flag = 'Start FEC Transmit'
            #     elif last_packet.nc_header[0][1] % (self.rtt-1) == 0 and last_packet.nc_header[0][1] > 0:
            #         self.fec_flag = 'Start FEC Transmit'

            # A feedback arrived: Update the corresponding ct ack
            ack = in_fb[1]
            if ack is not None:
                for ind_c, ct in enumerate(self.ct_buffer.fifo_items()):
                    channels_num = len(self.cfg.param.er_rates)
                    if ct[0] == self.p_id - int(self.rtt + 2):  # 2 instead of + channels_num + 1
                        self.ct_buffer.fifo_items()[ind_c][3] = ack
                        break

            # Update w_min:
            self.update_w_min()

            # Update w_max:
            self.epsilon_estimation(fb_packet=fb_packet, in_cur_fb=in_fb)
            self.update_w_max(in_fb)

            # Next transmission:
            if 'EMPTY' not in self.type:
                ct = [self.w_min, self.w_max]
                self.ct_buffer.put([self.p_id, ct, self.type, None])  # p_id, ct, fec/new, ack
            else:
                ct = [None, None]
                self.empty_indx.append(int(self.t))

            # Update the ct_buffer:
            self.p_id += 1

        else:  # Packet buffer is empty - Buffer begins empty - has nothing to transmit
            ct = None
            self.type = 'EMPTY_BEGIN'
            self.eps_mean = [1]

        self.print(in_fb=in_fb, is_arrived=pt_buffer[0], print_file=print_file)

        return ct, self.type, self.eps_mean

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
                # f'eps_hist = {self.eps}',
                f'ad = {self.ad}',
                f'md = {self.md}',
                f'cnew = {self.cnew}',
                f'csame = {self.csame}',
                f'delta = {self.delta}',
                f'criterion = {self.criterion}',
                f'fec_flag = {self.fec_flag}',
                f'fec_num = {self.fec_num :.2f}\n', file=f)

            print(f'erasure number = {np.sum(1 - np.array(self.erasure_tracker.fifo_items()))}', file=f)

            # print(
            # f'ct_buffer = {self.ct_buffer.fifo_items()}',
            # f'pt_buffer = {self.pt_buffer.fifo_items()}',
            # f'erasure_tracker = {self.erasure_tracker.fifo_items()}',
            # f'transmission_line = {self.transmission_line.fifo_items()}')

        return

