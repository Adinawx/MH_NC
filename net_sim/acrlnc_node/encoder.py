import torch

class Encoder:

    def __init__(self):
        self.fec_print = None
        self.T = None
        self.t = None
        self.rtt = None
        self.tran_ind = None
        self.last_relev_slot = None
        self.transmission_line = None
        self.transmission_times_hist = None
        self.feedback_hist = None
        self.redun_track = None
        self.added_dof = None
        self.missing_dof = None
        self.c_t_new = None
        self.c_t_same = None
        self.pred = None
        self.eps = None
        self.criterion = None
        self.w_start = None
        self.w_end = None
        self.th = None
        self.o = None
        self.fec_flag = None
        self.fec_num = None
        self.status_hist = None

    def reset(self, T, rtt, th):
        # Time Variables
        self.T = T  # Max number of transmissions
        self.t = 0  # Current time
        self.rtt = rtt  # Round Trip Time

        # Window Variables - Determines c_t
        self.w_start = 1  # First raw packet in the current window
        self.w_end = 0  # Last raw packet in the current window

        # Transmission Variables
        self.tran_ind = 0  # Index of the last transmission
        self.last_relev_slot = 0  # Index of the last slot contains any undecoded packet.
        self.transmission_line = torch.zeros([T, 2])  # Scheduled Transmissions: [w_start, w_end], at time i transmitting the i's row of this vector.
        self.redun_track = torch.ones(T)  # 1=redundancy, 0=new packet in combination
        self.feedback_hist = torch.zeros([T, 2])   # Log of the feedbacks [ack, dec]. Used to update w_start.
        self.transmission_times_hist = torch.zeros(T)  # At index t, the first time raw packet p_t is transmitted.Used to comupte the dec_timea.

        # A-priori Criteria Variables
        self.fec_flag = 1  # 1 is no fec transmission, 0 is FEC transmission
        self.fec_num = torch.tensor(0)  # Number of FEC packets to be transmitted
        self.fec_print = False  # Flag for printing a FEC transmission

        # Posteriori Criteria Variables
        self.added_dof = 0  # Number of ACK new relevant packets before t-rtt
        self.missing_dof = 0  # Number of NACK new relevant packets before t-rtt
        self.c_t_new = 0  # Number of new packets in the last rtt
        self.c_t_same = 0  # Number of same packets in the last rtt
        self.th = th  # Threshold for the feedback criterion
        self.pred = torch.zeros([1, rtt])  # Prediction of the erasure probability
        self.eps = 0  # Prediction of the erasure probability
        self.criterion = False  # Feedback FEC criterion. If True, FB-FEC is transmitted.

        # End Of Window Variables
        self.o = int(1.5 * (rtt - 1))  # Maximum allowed number of packets in the window

        # Log Variables
        self.status_hist = []  # Log of the status of each transmission - No FB, FEC, FB-FEC, AddPacket, EoW

    def read_fb_and_update_w_start(self, t_minus:int) -> tuple:
        """
        Read the feedback at time t_minus and update w_start accordingly.
        :param t_minus: time to read the feedback (t-rtt)
        :return: [ack, dec] = [feedback, last decoded packet at t_minus]
        """
        ack = None
        if t_minus >= 0:
            ack = self.feedback_hist[t_minus, 0]

        dec = self.feedback_hist[t_minus, 1]
        if dec:  # If a packet is decoded, update w_start
            self.w_start = int(dec) + 1

        return ack, dec

    def update_last_relev_slot(self) -> None:
        """
        Update the last relevant slot, that is the last slot that may contain an undecoded packet.
        :return: None
        """

        # 1. find all slots that may be relevant
        all_w_ends = self.transmission_line[self.last_relev_slot: self.t, 1]
        # 2. find all slots that are relevant
        relevant_slots = torch.nonzero(self.w_start <= all_w_ends)
        # 3. find the oldest slot that is relevant
        if relevant_slots.nelement() != 0:
            self.last_relev_slot = int(self.last_relev_slot + relevant_slots[0]) # increment to the oldest relevant slot
        else:
            self.last_relev_slot = self.t # if decoding happened and w_start is now greater than any w_end ever sent.

    def update_md_ad(self) -> tuple:
        """
        Update the missing and added degrees of freedom by the feedback and according to the last relevant slot.
        :return: [missing_dof, added_dof] = [#Nack of New packets before last relevant slot, #Ack of Redundant packets before last relevant slot]
        """
        if self.last_relev_slot <= self.t - self.rtt:
            t_minus = self.t - self.rtt
            all_fb = self.feedback_hist[self.last_relev_slot: t_minus, 0]
            self.missing_dof = (1 - all_fb) @ (1 - self.redun_track[self.last_relev_slot: t_minus])  # Nack x New
            self.added_dof = all_fb @ self.redun_track[self.last_relev_slot: t_minus]  # Ack x redun
        else:
            self.missing_dof = 0
            self.added_dof = 0

        return self.missing_dof, self.added_dof

    def update_ct_new_same(self) -> tuple:
        """
        Update the number of new and same packets in the last rtt.
        :return: [c_t_new, c_t_same] = [Number of New packets in the last rtt, Number of Redundant packets in the last rtt]
        """
        if self.last_relev_slot <= self.t - self.rtt:
            t_minus = self.t - self.rtt
        else:
            t_minus = self.last_relev_slot

        self.c_t_same = sum(self.redun_track[t_minus: self.t])  # Redun
        self.c_t_new = self.t - t_minus - self.c_t_same  # New = amount of relevant slots minus the redun slots

        return self.c_t_new, self.c_t_same

    def get_pred(self, pred):
        """
        Get the prediction of the erasure probability.
        :param pred: vector of predictions of the erasure probability
        :return: eps_hist: erasure rate in the last rtt time slots.
        """
        self.pred = pred
        return self.pred

    def update_eps(self, pred=None) -> torch.tensor:
        """
        Update the prediction of the erasure probability.
        :param pred: vector of predictions of the erasure probability
        :return: eps_hist: erasure rate in the last rtt time slots.
        """
        self.eps = torch.mean(1 - self.pred[0, 1:self.rtt])
        return self.eps

    def fb_criterion(self):
        # Initialize
        criterion = False
        win_fec_size = int(self.rtt)
        if self.t == 0:
            self.fec_flag = 1

        if 0 <= self.t < self.T:

            self.update_last_relev_slot()
            self.update_md_ad()
            self.update_ct_new_same()
            self.update_eps()

            if self.t % win_fec_size == 0:  # end of generation, start fec transmission
                self.fec_num = self.c_t_new * self.eps #(self.rtt-1) * eps0
                # if self.fec_num - 1 < 0:  # Force at least one fec:
                #     self.fec_num += 1
                self.fec_flag = 0  # 0 is fec transmission, 1 is no fec transmission
                # self.fec_flag = 1  # Terminate fec transmission

            # debug
            if self.t == 2448:
                a=5

            # print(f"flag: {self.fec_flag}, fec num: {self.fec_num}")
            miss = torch.round(self.missing_dof + self.eps  * self.c_t_new, decimals=4)  # round after 4 digits to prevent numerical problems.
            add = torch.round(self.added_dof + (1 - self.eps ) * self.c_t_same, decimals=4)

            delta_t = (miss - add) - self.th

            # debug
            if torch.isnan(delta_t):
                a=5

            self.criterion = (delta_t.detach() > 0)  # True is FB-FEC

            if self.fec_flag == 0 and self.fec_num.detach() - 1 >= 0:  # fec transmission
                self.fec_print = True
                criterion = True
                self.fec_num = self.fec_num - 1
                if self.fec_num - 1 <= 0:
                    self.fec_flag = 1  # End fec transmission

        return delta_t, self.criterion

    def update_transmission_line(self, tran_num, status):
        if self.tran_ind < self.T:

            for i in range(tran_num):
                self.transmission_line[self.tran_ind, :] = torch.tensor([self.w_start, self.w_end])
                self.status_hist.append(status)
                self.tran_ind += 1

                if self.tran_ind >= self.transmission_line.shape[0]:  # End of transmission, reached T scheduled transmissions.
                    return True

        return False

    def enc_step(self):

        if self.t == 0:
            self.w_end += 1
            self.transmission_times_hist[self.w_end - 1] = self.tran_ind
            self.redun_track[self.tran_ind] = 0
            transmission_num = 1
            status = 'No FB'
            done = self.update_transmission_line(transmission_num, status)
            if done:
                return True

        else:
            if self.w_end - self.w_start > self.o:  # Eow
                transmission_num = 1
                status = 'EoW'
            else:
                if self.w_start <= self.w_end and self.criterion:  # FEC or FB-FEC
                    transmission_num = 1
                    # if delta_t <= 0:
                    if self.fec_print:
                        status = 'FEC'
                        self.fec_print = False
                        # print('FECCC')
                    else:
                        status = 'FB-FEC'

                else:  # Add A Packet
                    transmission_num = 1
                    self.w_end += 1
                    self.transmission_times_hist[self.w_end - 1] = self.tran_ind
                    self.redun_track[self.tran_ind] = 0
                    status = 'AddPacket'

            done = self.update_transmission_line(transmission_num, status)
            if done:
                return True

    def get_ct(self, t):
        c_t = self.transmission_line[t, :].clone()
        return c_t