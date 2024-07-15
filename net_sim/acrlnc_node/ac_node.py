from encoder_new import Encoder


class ACRLNC_Node():

    ### Each node has a buffer of packets.
    ### Each packet has a nc_id, updated here. Ct = [nc_id, nc_id]
    ### Each packet has a header. Pt =  [id, id]. id= nc_id of previous Node.
    ### Each packet has an ACK flag: 0 = NACK, 1 = ACK, None = No ACK.
    ### Each packet has a FEC flag: 0 = Data, 1 = FEC.

    def __init__(self):
        self.T = None
        self.t = None
        self.RTT = None

        self.enc = Encoder()
        self.ct_buffer = None  # Buffer of [c_t]
        self.erasure_buffer = None  # Buffer of erasures, vector of size Memory.

        self.pt_buffer = None  # Packet buffer, buffer of [p_t]
        self.arrival_num = None  # Number of packets arrived at the node, int.

        self.in_pt = None  # Received ct from the sender, None = no ct, [p_start_ind, p_end_ind] = ct.
        self.out_ct = None  # ct to be sent to the receiver, [p_start_ind, p_end_ind] = ct.
        self.in_fb = None  # Feedback from the receiver, 0 = erasure, 1 = success, None = no feedback, vector of size Memory.
        self.out_fb = None  # Feedback to the sender, 0 = erasure, 1 = success, None = no feedback, int.

    def reset(self):
        self.T = 0
        self.t = 0
        self.RTT = 0

        self.ct_buffer = []

        self.pt_buffer = []
        self.arrival_num = 0

        self.in_pt = None
        self.out_ct = None
        self.in_fb = None
        self.out_fb = None
        return

    def run(self, in_pt, in_fb):

        self.update_t()

        # Receive the packet:
        self.in_pt = in_pt
        is_reception = self.is_reception()  # Check if a packet is received and update the erasure buffer.
        # Update the arrival number and the in buffer:
        if is_reception:
            self.update_pt_buffer_end()
        # Out Feedback:
        self.out_fb[0] = [self.in_pt.id, is_reception]  # TODO: Check the syntax.

        # Receive the feedback:
        self.in_fb = in_fb
        # Update the in buffer:
        self.update_pt_buffer_start()
        # Update ACKs:
        self.update_ack()
        # Out Feedback:
        self.out_fb[1] = self.decode()

        # Update the out buffer:
        ct, type = self.enc.encode(self.pt_buffer, self.ct_buffer, self.erasure_buffer)
        self.out_ct.header = [ct[0], ct[1]]
        if type == 'FEC' or type == 'FB-FEC' or type == 'EOW':
            self.out_ct.fec = True
        self.ct_buffer.append(self.out_ct)

        return self.out_ct, self.out_fb

    def update_t(self):
        self.t += 1
        return

    def accept_fec(self):
        # decide if to accept the FEC packet. # TODO: Implement the function.
        return True

    def update_pt_buffer_start(self):
        dec = self.in_fb.dec
        if dec is not None:
            # Update the buffer:
            buffer_ids = [pt.nc_id for pt in self.pt_buffer]
            dec_id = buffer_ids.index(dec)
            self.pt_buffer = self.pt_buffer[dec_id+1:]
            # Update the arrival number:
            self.arrival_num -= dec  # TODO: Check.
            return True # Decoded packet
        else:
            return False # No decoded packet

    def update_pt_buffer_end(self):
        if self.in_pt is not None:  # Packet Arrives:
            # A new packet OR a relevant FEC packet arrives:
            if not self.in_pt.fec or (self.in_pt.fec and self.accept_fec()):
                # Update NC id:
                last_nc_id = self.pt_buffer[-1].nc_id
                self.in_pt.nc_id = last_nc_id + 1
                # Insert the packet to the buffer:
                self.pt_buffer.append(self.in_pt)
                # Update the arrival number:
                self.arrival_num += 1
            return True # Packet is accepted.

        else:
            return False # Packet is not accepted.

    def update_ack(self):
        pt_id = self.in_fb[0][0]
        ack = self.in_fb[0][1]
        self.pt_buffer[pt_id].ack = ack
        return ack

    def is_reception(self):
        self.erasure_buffer[:-1] = self.erasure_buffer[1:]
        if self.in_pt is not None:
            self.erasure_buffer[-1] = 1
            return 1  # ACK
        else:
            self.erasure_buffer[-1] = 0
            return 0  # NACK

    def decode(self):

        id1, id2 = self.pt_buffer[0].header
        packets_num = id2 - id1 + 1
        if self.arrival_num == packets_num > 0:
            dec = self.pt_buffer[-1].id
        else:
            dec = 0

        return dec