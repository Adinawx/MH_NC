from ns.port.fifo_store import FIFO_Store
import numpy as np
from .esp_est import EpsEstimator
import os


class Encoder:

    def __init__(self, cfg, env, t=0):

        self.cfg = cfg

        self.t = t
        self.rtt = cfg.param.rtt
        self.p_id = 0

        capacity = self.cfg.param.T

        # memory holder of relevant packets. (get() when decoding happened):
        # ct = [p_id, fec_type,[w_min_t, w_max_t], ack/nack]
        self.ct_buffer = FIFO_Store(env, capacity=capacity, memory_size=float('inf'), debug=False)

        # tranmsmission line of packets. (get() every time step):
        self.transmission_line = FIFO_Store(env, capacity=capacity, memory_size=float('inf'), debug=False)

        self.pt_buffer = FIFO_Store(env, capacity=capacity, memory_size=float('inf'), debug=False)

        self.p_th = -0.2  # threshold for the delta calculation
        self.ad = 0
        self.md = 0
        self.cnew = 0
        self.csame = 0
        self.delta = [0, 0]  # [delta_mean, delta_max]

        self.fec_num = 0
        self.fec_flag = False
        self.EOW = int(1.5 * (self.rtt + 1))
        self.bls_num = 0
        self.bls_flag = False
        self.bls_time_start = 0
        self.in_fec_hold_flag = False

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

    def update_w_min(self):

        # Update w_min:
        # take w_min from the buffer.
        first_packet = self.pt_buffer.fifo_items()[0].nc_serial
        self.w_min = first_packet

        # Update ct_buffer:
        # if packets were decoded, update the c_t buffer.
        ct_temp = self.ct_buffer.fifo_items().copy()
        for ct in ct_temp:
            if ct[1][1] is not None:
                if ct[1][1] < self.w_min:
                    self.ct_buffer.get()
                else:
                    break
        return

    def update_delta(self):

        # all_ct = self.ct_buffer.fifo_items()
        # self.ad = len([ct for ct in all_ct if(ct[2] != 'NEW' and 'EMPTY' not in ct[2]) and ct[3] == 1])
        # self.md = len([ct for ct in all_ct if ct[2] == 'NEW' and ct[3] == 0])
        # self.csame = len([ct for ct in all_ct if (ct[2] != 'NEW' and 'EMPTY' not in ct[2]) and ct[3] is None])
        # self.cnew = len([ct for ct in all_ct if (ct[2] == 'NEW') and ct[3] is None])

        all_ct = self.ct_buffer.fifo_items()
        self.ad = len([ct for ct in all_ct if ('NEW' not in ct[2] and 'EMPTY' not in ct[2]) and ct[3] == 1])
        self.md = len([ct for ct in all_ct if 'NEW' in ct[2] and ct[3] == 0])
        self.csame = len([ct for ct in all_ct if ('NEW' not in ct[2] and 'EMPTY' not in ct[2]) and ct[3] is None])
        self.cnew = len([ct for ct in all_ct if ('NEW' in ct[2]) and ct[3] is None])

        # # New code:
        # eps_hist pf the current channel (index 0 at eps_ list):
        eps_mean = self.eps_mean[0]
        eps_max = self.eps_max[0]

        # Calculate delta for mean and max eps_hist:
        delta_mean = (self.md + eps_mean * self.cnew) - (self.ad + (1 - eps_mean) * self.csame) - self.p_th * (
                    self.ad + (1 - eps_mean) * self.csame)
        delta_mean = np.round(delta_mean, 4)  # Avoid numerical errors.

        delta_max = (self.md + eps_max * self.cnew) - (self.ad + (1 - eps_max) * self.csame) - self.p_th * (
                    self.ad + (1 - eps_max) * self.csame)
        delta_max = np.round(delta_max, 4)  # Avoid numerical errors.

        self.delta = [delta_mean, delta_max]

        if self.cfg.param.print_flag:
            print(f"ad={self.ad}, md={self.md}, csame={self.csame}, cnew={self.cnew}, delta={self.delta}")

        return self.delta

    def update_w_max_old(self, in_fb):

        # 1. Decoding happened:
        if self.w_max == self.w_min - 1:  # Decoding happened and w_max < w_min.
            self.w_max = self.w_min
            self.type = 'NEW'
            return

        self.type = ''

        # # Check for an IN-FEC packet:
        # last_packet = self.pt_buffer.fifo_items()[-1]
        # # Meaning there is a packet that wasn't sent yet in the buffer
        # if self.w_max < last_packet.nc_serial:
        #     # if it's a FEC packet, send it
        #     if 'NEW' not in last_packet.fec_type:
        #         self.w_max = self.w_max + 1
        #         self.type = 'NEW-F2'
        #         return

        # 2.1 FEC rule:
        # end of "generation", start FEC transmission
        eps_ = self.eps_mean[0]

        if self.cfg.param.er_estimate_type == 'stat_max':
            eps_ = self.eps_max[0]

        self.update_delta()  # anyway update so c_new will be accurate for a-priori FEC

        ######## Option 1: FEC by Time. Dis-Activate in main the other option ########
        # begin_delay = self.curr_ch * int(self.rtt / 2 + 1) #+ 1
        # if (self.t - begin_delay) % self.rtt == 0 and self.t > 0:
        #     self.fec_num = (self.cnew * eps_)  # self.rtt * eps_  #self.cnew * self.eps_hist  # number of FEC packets
        #     if self.cfg.param.print_flag:
        #         print(f"---fec_num={self.fec_num}, eps={eps_}, cnew={self.cnew}---")
        #     if self.fec_num - 1 >= 0:  # Activate FEC transmission
        #         self.fec_flag = True
        #         # self.fec_flag = False  # Manually terminate FEC transmission
        #     else:
        #         self.fec_flag = False
        #
        # # Check FEC transmission:
        # if self.fec_flag and self.fec_num - 1 >= 0:  # FEC transmission
        #     self.criterion = True
        #     self.type = 'FEC'
        #     # Reduce the number of FEC packets for the next time step:
        #     self.fec_num = self.fec_num - 1
        #
        #     # End FEC transmission
        #     if self.fec_num - 1 < 0:
        #         self.fec_flag = False
        #
        #         # # BLS-FEC transmission:
        #         if self.cfg.param.prot_type:
        #             if len(self.eps_mean) > 1:
        #                 self.bls_flag = True
        #                 self.bls_time_start = self.t
        #
        #                 # BLS all in one stream
        #                 ep_rate_cut = self.cfg.param.er_rates[self.curr_ch:]
        #                 ep_bn = max(ep_rate_cut)
        #                 # ep_bn = max(self.eps_mean[1:])
        #                 ep_bn_idx = ep_rate_cut.index(ep_bn)
        #                 self.bls_num = np.round(np.sum((self.rtt) * self.eps_mean[1:ep_bn_idx+1]))
        #     return
        ####### Option 2: FEC by packets. Activate in main as well ########
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
        #         if self.cfg.param.prot_type:
        #             if len(self.eps_mean) > 1:
        #                 self.bls_flag = True
        #                 self.bls_time_start = self.t
        #                 # BLS distributed:
        #                 # n = np.round(np.sum(self.rtt * self.eps_mean[1:]))
        #                 # self.bls_num = np.round(
        #                 #     np.linspace(self.rtt/(1+n), self.rtt-self.rtt/(1+n), int(n))
        #                 # )
        #                 # BLS all in one stream
        #                 self.bls_num = np.round(np.sum((self.rtt) * self.eps_mean[1:]))
        #     return
        #########################################################

        ######## Option 3: FEC by Time But with FEC Delay. Dis-Activate in main the other option ########
        # Activate FEC period every RTT timeslots.
        begin_delay = self.curr_ch * int(self.rtt / 2 + 1)  # + 1
        if (self.t - begin_delay) % self.rtt == 0 and self.t > 0:

            # sent_new = len([ct for ct in self.ct_buffer.fifo_items() if ('NEW' in ct[2])]) # and ct[3] is None
            # if sent_new == self.rtt-1 and self.t > 0:
            self.fec_num = (self.rtt * eps_)  # number of FEC packets

            if self.cfg.param.print_flag:
                print(f"---fec_num={self.fec_num}, eps={eps_}, cnew={self.cnew}---")
            if self.fec_num - 1 >= 0:  # Activate FEC transmission
                self.fec_flag = True
                self.in_fec_hold_flag = True
                # self.fec_flag = False  # Activate to manually terminate FEC transmission
            else:
                self.fec_flag = False

        # Check for an IN-FEC packet:
        # if self.fec_flag and self.in_fec_hold_flag:
        #     last_packet = self.pt_buffer.fifo_items()[-1]
        #     # Meaning there is a packet that wasn't sent yet in the buffer
        #     if self.w_max < last_packet.nc_serial:
        #         # if it's a new packet, ignore and send all FEC packets
        #         if 'NEW' == last_packet.fec_type:
        #             self.in_fec_hold_flag = False
        #         # if it's an IN-FEC packet, send it
        #         # elif 'NEW' != last_packet.fec_type:
        #         elif 'FEC' == last_packet.fec_type:
        #             self.w_max = self.w_max + 1
        #             self.type = 'NEW-IN-FEC'
        #             return

        # Send A FEC:
        if self.fec_flag and self.fec_num - 1 >= 0:  # FEC transmission
            self.criterion = True
            self.type = 'FEC'
            # Reduce the number of FEC packets for the next time step:
            self.fec_num = self.fec_num - 1

            # End FEC period (For next timestep):
            if self.fec_num - 1 < 0:
                self.fec_flag = False

                # Activate BLS period (For next timestep):
                if self.cfg.param.prot_type == "BS":
                    if len(self.eps_mean) > 1:
                        self.bls_flag = True
                        self.bls_time_start = self.t
                        # BLS all in one stream
                        ep_rate_cut = self.cfg.param.er_rates[self.curr_ch:]
                        ep_bn = max(ep_rate_cut)
                        # ep_bn = max(self.eps_mean[1:])
                        ep_bn_idx = ep_rate_cut.index(ep_bn)
                        self.bls_num = np.round(np.sum((self.rtt) * self.eps_mean[1:ep_bn_idx + 1]))
            return

        ###### Blank Space ####################################################
        if self.bls_flag:  # BLS transmission is active
            if self.bls_num - 1 >= 0:  # Enough timesteps for BLS transmission

                # 1. Check for fb-fec:
                if in_fb[0] is not None:  # Feedback is arrived
                    self.criterion = (self.delta[0] > 0)  # True is FB-FEC
                    if self.criterion and self.type == '':
                        self.type = 'FB-FEC'
                        self.bls_num = self.bls_num - 1
                        return

                # 2. Check for empty space:
                curr_ep = self.cfg.param.er_rates[self.curr_ch]

                # Option 1: Local bottleneck
                ep_bn = max(self.cfg.param.er_rates[self.curr_ch:])
                # ep_bn = max(self.eps_mean[1:])
                if curr_ep >= ep_bn:
                    # Stop BLS transmission
                    self.bls_flag = False
                    self.bls_num = 0
                    self.type = ''
                    return

                # # Option 2: Global bottleneck
                # ep_bn = max(self.cfg.param.er_rates)
                # global_bn = self.cfg.param.er_rates.index(ep_bn)
                # if self.curr_ch >= global_bn:
                #     # Stop BLS transmission
                #     self.bls_flag = False
                #     self.bls_num = 0
                #     self.type = ''
                #     return

                delta_hat = 1 / (self.bls_num * (1 - self.eps_mean[0]))
                self.criterion = (delta_hat <= 1 - ep_bn)  # Continue bls transmission.
                if self.criterion:
                    self.type = 'EMPTY-BLS'
                    self.bls_num = self.bls_num - 1
                    return

                # Stop BLS transmission
                self.bls_flag = False
                self.bls_num = 0
                self.type = ''
        #######################################################################

        # 2.3 FB-FEC rule:
        if in_fb[0] is not None:  # Feedback is arrived
            self.criterion = (self.delta[0] > 0)  # True is FB-FEC

        # Optional: Use stat_max for criterion:
        if self.cfg.param.er_estimate_type == 'stat_max':
            self.criterion = (self.delta[1] > 0)  # True is FB-FEC

        # Set transmission type:
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
                # self.type = 'FEC_BUFFER'
                return

            if w_max_new - self.w_min > self.EOW:
                self.type = 'EOW'
                return

            self.w_max = w_max_new

            # Check for an IN-FEC packet:
            last_packet = self.pt_buffer.fifo_items()[-1]
            ##############################
            if 'NEW' not in last_packet.fec_type:
                self.type = 'NEW-F1'
            else:
                self.type = 'NEW'
            ##############################

            if self.w_max == self.w_min - 1:
                self.w_max = self.w_min
            elif self.w_max < self.w_min:
                print('ERROR: w_max < w_min')
                return
        return

    def update_w_max(self, in_fb):

        # First packet: ########################################################################
        if self.w_max == -1:
            self.w_max = 0
            self.type = 'NEW'
            return
        ########################################################################################

        # Decoding happened and need to update w_max. ##########################################
        if self.w_max == self.w_min - 1:
            self.w_max = self.w_min
            self.type = 'NEW'
            return
        ########################################################################################

        # Check for an IN-FEC packet: ##########################################################
        # last_packet = self.pt_buffer.fifo_items()[-1]
        # gap = last_packet.nc_serial - self.w_max
        # if gap > 0:
        #     semi_new_packet = self.pt_buffer.fifo_items()[-gap]
        #     # Meaning there is a packet that wasn't sent yet in the buffer
        #     if 'NEW' not in semi_new_packet.fec_type:
        #         self.w_max = self.w_max + 1
        #         self.type = 'NEW-F'
        #         return
        ########################################################################################

        # End Of Window: #######################################################################
        if self.w_max - self.w_min >= self.EOW:
            self.type = 'EOW'
            return
        ########################################################################################

        self.type = ''
        self.update_delta()  # anyway update so c_new will be accurate for a-priori FEC

        eps_vec = self.eps_mean if self.cfg.param.er_estimate_type == 'stat' else self.eps_max
        # A-prior FEC: #########################################################################
        # Activate FEC transmission every RTT timeslots from the transmission start at curr_ch
        begin_delay = self.curr_ch * int(self.rtt / 2 + 1)
        if (self.t - begin_delay) % self.rtt == 0 and self.t > 0:

            # number of FEC packets to send:
            eps_ = eps_vec[0]
            self.fec_num = np.floor(self.cnew * eps_)

            if self.cfg.param.print_flag:
                print(f"---fec_num={self.fec_num}, eps={eps_}, cnew={self.cnew}---")

            # Check if there is a need for FEC:
            if self.fec_num - 1 >= 0:
                self.fec_flag = True
                self.in_fec_hold_flag = True
                # self.fec_flag = False  # Activate to manually terminate FEC transmission
            else:
                self.fec_flag = False

        # Send A FEC:
        if self.fec_flag and self.fec_num - 1 >= 0:  # FEC transmission
            self.criterion = True
            self.type = 'FEC'
            # Reduce the number of FEC packets for the next time step:
            self.fec_num = self.fec_num - 1

            # End FEC period (For next timestep):
            if self.fec_num - 1 < 0:
                self.fec_flag = False

                # Activate BLS period (For next timestep):
                if self.cfg.param.prot_type == 'BS-FEC' or self.cfg.param.prot_type == 'BS-EMPTY':
                    if len(self.eps_mean) > 1:
                        self.bls_flag = True

                        ep_bn = max(eps_vec)
                        ep_bn_idx = eps_vec.index(ep_bn)

                        if ep_bn_idx == 0:  # Current channel is the bottleneck
                            self.bls_num = 0

                        else:
                            self.bls_num = np.round((self.rtt / 2) * np.sum(eps_vec[1:ep_bn_idx + 1]))

                            # sum_eps = [e/(1+e) for e in eps_vec]
                            # self.bls_num = np.round((self.rtt) * np.sum(sum_eps[1:ep_bn_idx+1]))

                        # self.bls_fix_num = self.bls_num
                        # self.bls_num = np.round((self.rtt) * np.sum(eps_vec[1:ep_bn_idx+1]))

            return
        ##########################################################################################

        ###### Blank Space ####################################################
        if self.bls_flag:
            if self.bls_num > 0:  # Enough timesteps for BLS transmission

                # 1. Check for fb-fec:
                if in_fb[0] is not None:  # Feedback is arrived
                    self.criterion = (self.delta[0] >= 0)  # True is FB-FEC
                    if self.criterion and self.type == '':
                        self.type = 'FB-FEC'
                        self.bls_num = self.bls_num - 1
                        return

                ep_bn = max(eps_vec)
                ep_bn_idx = eps_vec.index(ep_bn)
                curr_ep = eps_vec[0]
                hop_gap = ep_bn_idx

                if curr_ep < ep_bn:  # Make sure the current channel is not the bottleneck (may be due to updated estimation)

                    delta_hat = 1 / (self.bls_num * (1 - eps_vec[0]) + (hop_gap) * np.log10((ep_bn - curr_ep)))
                    self.criterion = (delta_hat < 1 - ep_bn)  # Continue bls transmission.

                    if self.criterion:
                        self.type = 'EMPTY-BLS'
                        self.bls_num = self.bls_num - 1
                        return

                # LOG:
                # else:
                #     new_folder = os.path.join(self.cfg.param.results_folder, self.cfg.param.results_filename)
                #     with open(os.path.join(new_folder, f"metrics.txt"), "a") as f:
                #         f.write(f'\nt: {self.t}')
                #         f.write(f'\ncurr_ch: {self.curr_ch}')
                #         f.write('\nBLS Criterion Terminated')

                # Stop BLS transmission
                self.bls_flag = False
                self.bls_num = 0
                self.type = ''

        #######################################################################

        # FB-FEC: ################################################################################
        if self.cfg.param.er_estimate_type == 'oracle':
            self.criterion = (self.delta[0] >= 0)
            if self.criterion:
                self.type = 'FB-FEC'
                return
        else:
            if in_fb[0] is not None:  # Feedback is arrived
                self.criterion = (self.delta[0] >= 0)  # True is FB-FEC
                if self.criterion:
                    self.type = 'FB-FEC'
                    return
        ###########################################################################################

        # Empty buffer (no new): ##################################################################
        last_packet = self.pt_buffer.fifo_items()[-1].nc_serial
        if self.w_max + 1 > last_packet:
            if self.cfg.param.prot_type == 'AC-FEC' or self.cfg.param.prot_type == 'BS-FEC':
                self.type = 'FEC_BUFFER'
                return
            if self.cfg.param.prot_type == 'AC-EMPTY' or self.cfg.param.prot_type == 'BS-EMPTY':
                self.type = 'EMPTY_BUFFER'
                return
        ##########################################################################################

        # New packet: ############################################################################
        self.w_max += 1
        self.type = 'NEW'
        ##########################################################################################

        # ##############################
        # last_packet = self.pt_buffer.fifo_items()[-1]
        # if 'NEW' not in last_packet.fec_type:
        #     self.type = 'NEW-F1'
        # else:
        #     self.type = 'NEW'
        # ##############################

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
        if est_type == 'oracle':
            # win_length = min(len(self.ct_buffer.fifo_items()), self.rtt+1)
            if len(self.ct_buffer.fifo_items()) == 0:
                win_length = 0
            else:
                sent_len = self.p_id - self.ct_buffer.fifo_items()[0][0]
                win_length = min(sent_len, self.rtt + 1)
            eps_mean = self.eps_est.eps_estimate(self.t, self.curr_ch, 'oracle', win_length, self.empty_indx)
            eps_max = eps_mean

        elif est_type == 'genie':
            eps_mean = self.eps_est.eps_estimate(t=self.t, ch=self.curr_ch, est_type='genie')
            eps_max = eps_mean

        else:
            if self.cfg.param.er_estimate_type == 'stat':
                eps_mean = self.eps_est.eps_estimate(t=self.t, ch=self.curr_ch, est_type='stat')
                eps_max = eps_mean
            else:
                eps_max = self.eps_est.eps_estimate(t=self.t, ch=self.curr_ch, est_type='stat_max')
                eps_mean = eps_max

        if self.cfg.param.print_flag:
            print(f'eps_hist mean: {eps_mean[0]:.2f}')
            print(f'eps_hist max: {eps_max[0]:.2f}')

        eps = [eps_mean, eps_max]
        self.eps_mean = eps[0].tolist() if isinstance(eps[0], np.ndarray) else eps[0]
        self.eps_max = eps[1].tolist() if isinstance(eps[1], np.ndarray) else eps[1]

        return

    def run(self, pt_buffer: [], in_fb: [], fb_packet=None, t=0, print_file=None, info=None):

        self.t = t
        self.curr_ch = info[0]

        self.pt_buffer = pt_buffer[1]

        if len(self.pt_buffer) > 0:  # Packet buffer is not empty

            ack = in_fb[1]
            if ack is not None:
                items = self.ct_buffer.fifo_items()
                target_id = self.p_id - int(self.rtt + 2)

                # Create a dictionary for quick lookup
                nc_serial_to_index = {ct[0]: ind_c for ind_c, ct in enumerate(items)}

                if target_id in nc_serial_to_index:
                    items[nc_serial_to_index[target_id]][3] = ack

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

        return ct, self.type, self.eps_mean
