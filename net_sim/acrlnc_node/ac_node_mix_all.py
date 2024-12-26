from acrlnc_node.encoder_mix_all import Encoder
from ns.port.fifo_store import FIFO_Store

import numpy as np
np.random.seed(42)

import copy

class ACRLNC_Node_Mix_All():

    ### Each node has a buffer of packets.
    ### Each packet has a nc_id, updated here. Ct = [nc_id, nc_id]
    ### Each packet has a header. Pt =  [id, id]. id= nc_id of previous Node.
    ### Each packet has an ACK flag: 0 = NACK, 1 = ACK, None = No ACK.
    ### Each packet has a FEC flag: NEW, FEC, FB-FEC, EOW

    def __init__(self, env, cfg, t=-1):
        self.cfg = cfg
        self.env = env
        self.t = t
        self.node_type = None  # Transmitter, Receiver, Intermediate
        self.node_index = None  # Channel index.

        self.print_file = None

        self.fb_buffer = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'),
                                    debug=False)  # Buffer of erasures, vector of size Memory.
        self.pt_buffer = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'),
                                    debug=False)  # Packet buffer, buffer of [p_t]
        self.in_pt = None  # Received ct from the sender, None = no ct, [p_start_ind, p_end_ind] = ct.
        self.out_ct = [None, None]  # [w_min, w_max] = ct to be sent to the receiver.

        self.in_cur_fb = [None, None, None]  # Feedback from the next node [ack_id, ack, dec_id].

        self.out_cur_fb = [None, None, None]  # Feedback to send out to the previous node [ack_id, ack, dec_id]
        self.out_all_fb = []  # Feedback to send out to all following nodes [packets].
        self.fb_dec_id = -1

        self.last_nc_id = -1  # Last nc_id of the packet.
        self.last_packet_store = None  # Last packet stored in the buffer.
        self.info_packet_nc_header = None  # nc_id of the info packet header.

        # Encoder:
        self.enc = Encoder(cfg=cfg, env=env)
        self.eps_hist = []
        self.trans_buffer = FIFO_Store(env, capacity=float('inf'),
                                       memory_size=float('inf'))  # Buffer of source packets in the transmitter

        # Decoder:
        # self.packets_num = 0  # Number of info packets.
        self.last_dec_id_info = -1  # Last decoded packet in the receiver.
        self.last_dec_id = -1  # Last decoded packet.

        # Receiver:
        self.rece_buffer = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'),
                                      debug=False)  # Buffer of received packets.

        # First time to send an info packet: packet i is sent at time send_times[i].
        self.send_times = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.arrival_times = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        # Time of decoding of each packet: packet i is decoded at time dec_times[i].
        self.dec_times = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.semi_dec_times = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.semi_dec_times.put(1) # For the MIXALL case - semi-dec here is nonsense.
        self.ct_type_hist = []

    def run(self, in_packet_info, in_packet_recep_flag, fb_packet):
        '''
        :param in_packet_info: ct = packet.
        :param in_packet_recep_flag: 0 = Erasure, 1 = Reception, None
        :param fb_packet: fb = [[ack], [ack_id], dec_id], ack = 0 = NACK, 1 = ACK, None = No FB, ack_id = nc_id of the coming packet, dec_id = nc_id of the decoded packet.

        out_ct = [[w_min, w_max], fec_type] = ct to be sent to the receiver.
        out_cur_fb = [[ack_id], [ack], dec_id] = fb to be sent to the sender.
        '''

        if len(self.pt_buffer) == 0:
            brk = 5

        self.set_node_type(in_packet_info, fb_packet)

        self.get_fb(fb_packet)

        # Print the inputs
        if self.cfg.param.print_flag:
            if self.node_type == 'Intermediate' or self.node_type == 'Receiver' or self.node_type == 'Transmitter':
                # if self.node_type == 'Transmitter':
                # if curr_ch == 1 or curr_ch == 0:
                print('\n----t: ', self.t, '----',
                      '\nSRC Node: ', in_packet_info.src,
                      '\n---Inputs:---',
                      '\nin_packet_info: ', in_packet_info,
                      '\nin_packet_recep_flag: ', in_packet_recep_flag,
                      '\nfb_packet: ', fb_packet)

        # Decide if to accept the coming packet and update the nc_id.
        self.update_pt_buffer_add(in_packet_info, in_packet_recep_flag)

        if self.cfg.param.print_flag:
            if self.node_type == 'Intermediate' or self.node_type == 'Receiver' or self.node_type == 'Transmitter':
                # if self.node_type == 'Transmitter':
                # if curr_ch == 1 or curr_ch == 0:
                print('pt_buffer: dof_num:', self.relevant_dof(), ', packets_num:', self.update_packet_num())
                if self.node_type == 'Receiver':
                    print('rece_buffer: dof_num:', len(self.rece_buffer), ', packets_info_num:',
                          self.update_packet_info_num())

        # Semi-decode the packets and create feedback packet.
        self.decode_and_create_fb(in_packet_recep_flag, fb_packet)

        # Read FB and remove decoded packets from the pt buffer.
        self.update_pt_buffer_discard(fb_packet)

        # TODO: Adina: Now - makes ct for receiver's node anyway because it creates the log file - Fix this.
        # Call the encoder to generate the output packet.
        if self.node_type != 'Receiver':
            self.output_packet_processing(in_packet_recep_flag, fb_packet, in_packet_info)

        # Print the outputs
        if self.cfg.param.print_flag:
            if self.node_type == 'Intermediate' or self.node_type == 'Receiver' or self.node_type == 'Transmitter':
                # if self.node_type == 'Transmitter':
                # if curr_ch == 1 or curr_ch == 0:
                print('---Outputs:---')
                if self.node_type != 'Receiver':
                    print('out_ct: ', self.out_ct, )
                print("out_cur_fb: ",
                      f'ack_id:{self.out_cur_fb[0]} || ack:{self.out_cur_fb[1]} || dec_id:{self.out_cur_fb[2] + 1 if self.out_cur_fb[2] is not None else None}')

        return self.out_ct, self.out_all_fb

    def update_t(self, t):
        self.t = t
        return

    def set_node_type(self, in_packet_info, fb_packet):

        if in_packet_info.src == 's_ff':
            self.node_type = 'Transmitter'
            self.node_index = 0

        elif fb_packet.src == 'd_fb':
            self.node_type = 'Receiver'
            self.node_index = -1

        else:
            self.node_type = 'Intermediate'
            self.node_index = int(in_packet_info.src[-1]) + 1

        return

    def accept_fec(self):

        dof_num = self.relevant_dof()
        packets_num = self.update_packet_num()
        if dof_num < packets_num != 0:
            return True
        else:
            if self.last_packet_store is None:
                return True
            else:
                if self.last_packet_store.nc_header[0][1] < self.in_pt.nc_header[0][1]:
                    return True

            return False

    def get_fb(self, fb_packet):

        if fb_packet is not None:  # Some feedback is arrived.

            if fb_packet.fec_type is not None:
                ack_id = fb_packet.fec_type[0]
                ack = fb_packet.nc_header[0] if self.node_type != 'Receiver' else 1
                dec_id = fb_packet.nc_serial

                self.in_cur_fb = [ack_id, ack, dec_id]  # corresponding packet.
        return

    # OG
    # def update_pt_buffer_add(self, in_packet_info, in_packet_recep_flag):
    #
    #     # In packet info:
    #     self.in_pt = in_packet_info
    #
    #     # Do not accept packets that contain nothing.
    #     if isinstance(in_packet_info.nc_header, list):
    #         if in_packet_info.nc_header[1][0] is None:
    #             return
    #     else:
    #         if in_packet_info.nc_header is None:
    #             return
    #
    #     if in_packet_recep_flag:
    #         # Accept the packet:
    #         # if self.in_pt.fec_type == 'NEW' or (self.in_pt.fec_type != 'NEW' and self.accept_fec()):
    #         if self.node_index == 1 or 'NEW' in self.in_pt.fec_type or ('NEW' not in self.in_pt.fec_type and self.accept_fec()):
    #
    #             # Update the nc_id:
    #             self.last_nc_id = self.last_nc_id + 1
    #             self.in_pt.nc_serial = self.last_nc_id
    #
    #             # Insert the packet to the buffer:
    #             self.pt_buffer.put(self.in_pt)
    #
    #             if self.node_type == 'Receiver':
    #                 curr_info_wmax = self.in_pt.nc_header[1][1]
    #                 if curr_info_wmax > self.last_dec_id_info:
    #                     self.rece_buffer.put(self.in_pt)
    #
    #             self.last_packet_store = self.pt_buffer.fifo_items()[-1]
    #
    #             # Update Arrival time:
    #             # if 'NEW' in self.in_pt.fec_type:
    #             self.arrival_times.put(self.t)
    #     return

    # Poisson IN
    def update_pt_buffer_add(self, in_packet_info, in_packet_recep_flag):

        # poisson buffer distribution
        if self.node_type == 'Transmitter':
            self.trans_buffer.put(copy.deepcopy(in_packet_info))
            lam = max(self.cfg.param.er_rates)
            poisson = 1-np.random.poisson(lam=lam, size=1)
            # print('poisson:', poisson)
            if poisson > 0:
                self.in_pt = self.trans_buffer.fifo_items()[0]
                self.trans_buffer.get()
            else:
                self.in_pt = in_packet_info
                self.in_pt.fec_type = 'EMPTY_SOURCE'
                self.in_pt.nc_header = None
                return
        else:
            # In packet info:
            self.in_pt = in_packet_info

        # Do not accept packets that contain nothing.
        if isinstance(in_packet_info.nc_header, list):
            if in_packet_info.nc_header[1][0] is None:
                return
        else:
            if in_packet_info.nc_header is None:
                return

        if in_packet_recep_flag:
            # Accept the packet:
            # if self.in_pt.fec_type == 'NEW' or (self.in_pt.fec_type != 'NEW' and self.accept_fec()):
            if self.node_index == 1 or 'NEW' in self.in_pt.fec_type or ('NEW' not in self.in_pt.fec_type and self.accept_fec()):

                # Update the nc_id:
                self.last_nc_id = self.last_nc_id + 1
                self.in_pt.nc_serial = self.last_nc_id

                # Insert the packet to the buffer:
                self.pt_buffer.put(self.in_pt)

                if self.node_type == 'Receiver':
                    curr_info_wmax = self.in_pt.nc_header[1][1]
                    if curr_info_wmax > self.last_dec_id_info:
                        self.rece_buffer.put(self.in_pt)

                self.last_packet_store = self.pt_buffer.fifo_items()[-1]

                # Update Arrival time:
                # if 'NEW' in self.in_pt.fec_type:
                self.arrival_times.put(self.t)
        return

    def decode_and_create_fb(self, in_packet_recep_flag, fb_packet):

        # 1. Packet info:
        ack_id = self.in_pt.packet_id
        if ack_id is None:
            ack = None
        else:
            # 2. Ack:
            ack = 1  # Arrived packet (ack=True)
            if not in_packet_recep_flag:  # if NOT Arrived packet (ack=False)
                ack = 0

        # 3. Decoding:
        dec_id = -1  # No decoding (dec_id will be -1).

        if fb_packet is not None:  # Feedback packet is not empty.

            ###### Semi decode: use pt_buffer ######
            dof_num = self.relevant_dof()
            packets_num = self.update_packet_num()
            if dof_num >= packets_num > 0:  # and dof_num != 0:  # Enough DOF in an intermediate node.
                # id of the last info packet in the buffer.
                if isinstance(self.pt_buffer.fifo_items()[-1].nc_header, list):
                    dec_id = self.pt_buffer.fifo_items()[-1].nc_header[0][1]
                else:
                    dec_id = self.pt_buffer.fifo_items()[-1].nc_header

            if dec_id > self.last_dec_id:  # decoding.
                #If the node is a receiver, we need to remove the decoded packets from the buffer.
                # Otherwise, we remove them in "update_pt_buffer_discard".
                if self.node_type == 'Receiver':
                    for i in range(dec_id - self.last_dec_id):
                        if len(self.pt_buffer) == 1:
                            self.last_packet_store = self.pt_buffer.fifo_items()[0]
                        self.pt_buffer.get()
                        # Log the semi-decoding time.
                        self.semi_dec_times.put(self.t)
                self.last_dec_id = dec_id
            #############################################################################

            ###### Real decode: use rece_buffer ######
            if self.node_type == 'Receiver':  # Actual decoding at the receiver.
                packets_info_num = self.update_packet_info_num()
                dof_num = len(self.rece_buffer)
                if dof_num >= packets_info_num > 0:  # and dof_num != 0:  # Enough DOF in the receiver node.
                    # eliminate decoded packets and update their decode timing.
                    dec_id_info = self.rece_buffer.fifo_items()[-1].nc_header[1][1]
                    for i in range(dec_id_info - self.last_dec_id_info):
                        self.dec_times.put(self.t)
                        if self.cfg.param.print_flag:
                            print('dec_id: ', dec_id_info)
                        self.rece_buffer.get()
                    self.last_dec_id_info = dec_id_info

        # dec with a minus one to match the later +1 in the nc_serial field.
        self.out_cur_fb = [ack_id, ack, self.last_dec_id_info - 1]

        # 4. Feedback:
        if self.node_type == 'Receiver':  # Send a real feedback
            if self.in_cur_fb[1] is not None:  # If there is feedback, add it to the feedback packet.
                all_ack_id = [ack_id] + fb_packet.fec_type
                all_ack = [ack] + fb_packet.nc_header
                self.out_all_fb = [all_ack_id, all_ack, self.last_dec_id_info - 1]
            else:  # First feedback to transmit.
                self.out_all_fb = [[ack_id], [ack], self.last_dec_id_info - 1]

        else:  # Propagate the feedback
            if self.in_cur_fb[1] is not None:
                self.out_all_fb = [[self.in_cur_fb[0]], [self.in_cur_fb[1]], self.in_cur_fb[2]-1]
            else:
                self.out_all_fb = [[-1], [None], -3]
        return

    def update_pt_buffer_discard(self, fb_packet):

        if self.in_cur_fb[1] is not None:  # Some feedback is arrived.
            # The receiver feedback is not an indicator of decoding (its dec_id always increases). Update its buffer
            # in the decoding process.

            # if self.node_type != 'Receiver':\
            if self.node_type == 'Transmitter':
                # Remove decoded packets from the buffer:
                dec_id = self.in_cur_fb[2]
                if dec_id >= 0 and self.out_cur_fb[2] >= -1:  # Indicator of decoding in the current node, meaning it is indeed OK to discard packets.

                    while len(self.pt_buffer) > 0:  # Buffer is not empty.
                        first_pt_id = self.pt_buffer.fifo_items()[0].nc_serial
                        if first_pt_id <= dec_id:

                            # Store the last packet stored in the buffer if it is removed.
                            if len(self.pt_buffer) == 1:
                                self.last_packet_store = self.pt_buffer.fifo_items()[-1]

                            # Remove the packet from the buffer.
                            self.pt_buffer.get()

                        else:
                            break

                # dec_info = self.in_cur_fb[2]
                # if dec_info >= 0 and self.out_cur_fb[2] >= -1:
                #
                #     while len(self.pt_buffer) > 0:
                #         first_pt_wmax = self.pt_buffer.fifo_items()[0].nc_header[1][1]
                #
                #         if first_pt_wmax <= dec_info:
                #             if len(self.pt_buffer) == 1:
                #                 self.last_packet_store = self.pt_buffer.fifo_items()[-1]
                #             self.pt_buffer.get()
                #
                #         else:
                #             break

            elif self.node_type == 'Intermediate':

                while len(self.pt_buffer) > 0:

                    last_packet = self.pt_buffer.fifo_items()[-1]
                    incoming_w_min = last_packet.nc_header[0][0]

                    first_pt = self.pt_buffer.fifo_items()[0]
                    first_pt_w_min = first_pt.nc_header[0][0]
                    if first_pt_w_min < incoming_w_min:
                        if len(self.pt_buffer) == 1:
                            self.last_packet_store = last_packet
                        self.pt_buffer.get()
                    else:
                        break
        return

    def update_packet_num(self):

        if len(self.pt_buffer) > 0:  # Buffer is not empty.
            last_packet = self.pt_buffer.fifo_items()[-1]  # Last packet in the buffer.
        elif self.last_packet_store is not None:
            last_packet = self.last_packet_store
        else:
            return 0

        if isinstance(last_packet.nc_header, list):
            last_packet_wmax = last_packet.nc_header[0][1]

            if self.last_dec_id == -1:  # First decoding.
                packets_num = last_packet_wmax + 1
            else:
                packets_num = last_packet_wmax - self.last_dec_id
        else:
            if self.last_dec_id == -1:  # First decoding.
                packets_num = last_packet.nc_header + 1
            else:
                packets_num = last_packet.nc_header - self.last_dec_id

        return packets_num

    def update_packet_info_num(self):

        packets_info_num = 0
        if len(self.rece_buffer) > 0:  # Buffer is not empty.
            last_packet = self.rece_buffer.fifo_items()[-1]  # Last packet in the buffer.
            last_packet_wmax = last_packet.nc_header[1][1]  ### THIS IS A CHANGE from the previous function

            if self.last_dec_id_info == -1:  # First decoding.
                packets_info_num = last_packet_wmax + 1
            else:
                packets_info_num = last_packet_wmax - self.last_dec_id_info

        return packets_info_num

    def relevant_dof(self):

        """
        This function returns the number of relevant degrees of freedom (dof) in the buffer.
        :return: dof
        Since the buffer is emptied according to a feedback, but can decode anyways,
        some packets in the buffer may be irrelevant for the current window.
        Meaning - we keep them in the buffer and keep sending them forward,
        but don't necessarily count them for decoding incoming packets.
        """

        dof = 0
        for i in range(len(self.pt_buffer)):
            if self.pt_buffer.fifo_items()[i].nc_serial > self.last_dec_id:
                dof += 1

        return dof

    def output_packet_processing(self, in_packet_recep_flag, fb_packet, in_packet_info):

        # Cut the buffer to discard packets that are not relevant for the next node.
        # If some packets are acked as decoded - we do not need them in c_t
        # We may still want them to be in the pt_buffer

        pt_buffer_cut = self.pt_buffer
        if self.node_type == 'Transmitter':  # For MIXALL, do it only in the encoder
            if self.in_cur_fb[2] is not None and self.in_cur_fb[2] >= 0:
                self.fb_dec_id = self.in_cur_fb[2]
            if self.fb_dec_id is not None:  # If some decoding was done.
                pt_buffer_cut = FIFO_Store(self.env, capacity=float('inf'), memory_size=float('inf'), debug=False)
                for i in range(len(self.pt_buffer)):
                    if self.pt_buffer.fifo_items()[i].nc_serial > self.fb_dec_id:
                        pt_buffer_cut.put(self.pt_buffer.fifo_items()[i])
            else:
                pt_buffer_cut = self.pt_buffer

        # Encoding:
        ct, fec_type, eps_mean = self.enc.run(pt_buffer=[in_packet_recep_flag, pt_buffer_cut],
                                              in_fb=self.in_cur_fb,
                                              fb_packet=fb_packet,
                                              t=self.t,
                                              print_file=self.print_file,
                                              )
        if isinstance(eps_mean, list):
            self.eps_hist.append(eps_mean[0])
        else:
            self.eps_hist.append(eps_mean)

        # Info-packets header:
        [info_wmin, info_wmax] = [None, None]
        if ct is not None:  # meaning there is a packet to transmit. We want to pass its info-packets.
            ct_wmin = ct[0]
            ct_wmax = ct[1]
            if len(self.pt_buffer) > 0:  # Buffer is not empty.
                # find the info-packets for these nc_id:
                for i in range(len(self.pt_buffer)):

                    # Not the first node:
                    if isinstance(self.pt_buffer.fifo_items()[i].nc_header, list):
                        if self.pt_buffer.fifo_items()[i].nc_serial == ct_wmin:
                            info_wmin = self.pt_buffer.fifo_items()[i].nc_header[1][0]
                        if self.pt_buffer.fifo_items()[i].nc_serial == ct_wmax:
                            info_wmax = self.pt_buffer.fifo_items()[i].nc_header[1][1]

                    # First node:
                    else:
                        if self.pt_buffer.fifo_items()[i].nc_serial == ct_wmin:
                            info_wmin = self.pt_buffer.fifo_items()[i].nc_header
                        if self.pt_buffer.fifo_items()[i].nc_serial == ct_wmax:
                            info_wmax = self.pt_buffer.fifo_items()[i].nc_header

                self.info_packet_nc_header = [info_wmin, info_wmax]

            else:  # Buffer is empty.
                # take from the memory.
                info_wmin = self.info_packet_nc_header[0]
                info_wmax = self.info_packet_nc_header[1]

        if info_wmax is None and self.node_type != 'Receiver':
            brk = 5

        # Output packet info:
        self.out_ct = [[ct, [info_wmin, info_wmax]], fec_type]

        # Update the send times for the delay calculation.
        if 'NEW' in fec_type:
            self.send_times.put(self.t)

        self.ct_type_hist.append(fec_type)

        return
