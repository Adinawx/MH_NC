"""
Implements a port with an output buffer, given an output rate and a buffer size (in either bytes
or the number of packets). This implementation uses the simple tail-drop mechanism to drop packets.
"""
import simpy
from ns.port.fifo_store import FIFO_Store
import random  # For debug only. Erase eventually
from acrlnc_node.ac_node1 import ACRLNC_Node
from acrlnc_node.ac_node_mix_all import ACRLNC_Node_Mix_All
import numpy as np

class NCEncoder:
    """Models an output port on a switch with a given rate and buffer size (in either bytes
    or the number of packets), using the simple tail-drop mechanism to drop packets.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    debug: bool
        If True, prints more verbose debug information.
    """

    def __init__(
            self,
            env,
            cfg,
            element_id: int = None,
            debug: bool = False,
            enc_default_len: float = float('inf'),
            channel_default_len: float = float('inf'),
    ):
        self.env = env
        self.cfg = cfg
        self.out_ff = None
        self.out_fb = None
        self.element_id = element_id
        self.ff_packets_received = 0  # Number of packets received on the forward channel
        self.ff_packets_dropped = 0  # Number of packets dropped on the forward channel
        self.fb_packets_received = 0  # Number of packets received on the feedback channel
        self.fb_packets_dropped = 0  # Number of packets dropped on the feedback channel
        self.enc_default_len = enc_default_len
        self.channel_default_len = channel_default_len
        self.debug = debug
        self.hist_store_ff_time = []  # New - 18/8
        self.hist_store_fb_time = []  # New - 18/8
        self.hist_erasures = []

        ## Choose Protocol ###########################################
        if cfg.param.prot_type != "MIXALL":
            self.ac_node = ACRLNC_Node(env=env, cfg=cfg)
        else:
            self.ac_node = ACRLNC_Node_Mix_All(env=env, cfg=cfg)
        ##############################################################


        # Typically, the following FIFO_stores are used in order to pass the current packet to the nc encoder
        # (so, for example, len = 0 or 1):
        self.store_fb = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.store_ff = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.store_fb_ch = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.store_ff_ch = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

        # Typically, the following FIFO_stores are used in order to keep track of the arriving packet history:
        self.store_fb_hist = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.store_ff_hist = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.store_fb_ch_hist = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)
        self.store_ff_ch_hist = FIFO_Store(env, capacity=float('inf'), memory_size=float('inf'), debug=False)

        # Typically, the following FIFO_stores is used to store packets that were used in the
        # current linear combination:
        self.store_nc_enc = FIFO_Store(env, capacity=float('inf'), memory_size=self.enc_default_len, debug=False)

        # Typically, the following FIFO_stores is used to keep track of the noise used for
        # prediction in the RLNC algorithm:
        self.store_channel_stats = FIFO_Store(env, capacity=float('inf'), memory_size=self.channel_default_len,
                                              debug=False)

        self.action = env.process(self.run())

    def update(self, packet):
        """
        The packet has just been retrieved from this element's own buffer by a downstream
        node that has no buffers.
        """
        # There is nothing that needs to be done, just print a debug message

    def delete_pct_and_reindex(self):
        pass

    def run(self):
        """The generator function used in simulations."""

        res_folder = r"{}\{}".format(self.cfg.param.results_folder, self.cfg.param.results_filename)

        while True:

            ff_packets = yield self.store_ff.get()
            ff_ch = yield self.store_ff_ch.get()

            if self.fb_packets_received > 0:
                fb_packets = yield self.store_fb.get()
                fb_ch = yield self.store_fb_ch.get()
            else:  # ADINA
                fb_packets = None
                fb_ch = None

            if ff_packets.nc_header is not None:
                self.store_ff_hist.put(ff_packets)
                self.store_ff_ch_hist.put(ff_ch)
                self.hist_store_ff_time.append(self.env.now)
            ff_packets_hist = self.store_ff_hist.fifo_items()
            ff_ch_hist = self.store_ff_ch_hist.fifo_items()

            if self.fb_packets_received > 0:
                if fb_packets.nc_header is not None:
                    self.store_fb_hist.put(fb_packets)
                    self.store_fb_ch_hist.put(fb_ch)
                    self.hist_store_fb_time.append(self.env.now)
                fb_packets_hist = self.store_fb_hist.fifo_items()
                fb_ch_hist = self.store_fb_ch_hist.fifo_items()

            if self.debug:
                print(str(self.env.now) + '| ' + str(self.element_id))
                print('-----------------------    Packet (in)   ------------------------------------')
                for curr_ff_packet, curr_ff_ch in zip(ff_packets_hist, ff_ch_hist):
                    print(f"{curr_ff_packet} || {curr_ff_ch}")
                print('-----------------------    FB (in)     ------------------------------------')
                if self.fb_packets_received > 0:
                    for curr_fb_packet, curr_fb_ch in zip(fb_packets_hist, fb_ch_hist):
                        print(f"{curr_fb_packet} || {curr_fb_ch}")
                # print('-----------------------    END (in)     ------------------------------------')
                # print('\n')

            ######ADINA########################################################################################
            if ff_packets.nc_header is not None:

                print_file = r"{}\{}.txt".format(res_folder, self.element_id)
                # print(f"---{str(self.element_id)}---")

                # 1. Erasure Gate: False=erasure, True=reception
                ff_recep_flag = False
                curr_ch_state = ff_ch[-1] if isinstance(ff_ch, list) else ff_ch
                if curr_ch_state or curr_ch_state is None:
                    ff_recep_flag = True

                # 2. Run the AC node
                self.ac_node.print_file = print_file
                self.ac_node.update_t(self.env.now)
                out_ff, out_fb = self.ac_node.run(in_packet_info=ff_packets,
                                                  in_packet_recep_flag=ff_recep_flag,
                                                  fb_packet=fb_packets)

                # 3.1 Update next packet to be sent
                self.out_ff.nc_header = out_ff[0]  # [[w_min, w_max], [pinfo_min, pinfo_max]]
                self.out_ff.fec_type = out_ff[1]  # 'NEW' / 'FEC'

                # 3.2 Update next feedback packet to be sent
                self.out_fb.fec_type = out_fb[0]  # ack_id
                self.out_fb.nc_header = out_fb[1]  # ack / nack
                self.out_fb.nc_serial = out_fb[2]  # dec

                # 4. log
                # save ct type history:
                curr_ch = 0 if self.element_id == 'enc_node0' else int(
                    ff_packets.src[-1]) + 1

                # tran_times in the Transmitter
                # if self.element_id == 'enc_node0':  # the first node = Transmitter
                tran_times = np.array(self.ac_node.send_times.fifo_items())
                np.save(r"{}\trans_times_ch={}.npy".format(res_folder, curr_ch), tran_times)

                # Arrival times in each node
                arrival_times = np.array(self.ac_node.arrival_times.fifo_items())
                np.save(r"{}\arrival_times_ch={}.npy".format(res_folder, curr_ch), arrival_times)

                # Semi dec_time in each node
                dec_times = np.array(self.ac_node.semi_dec_times.fifo_items())
                np.save(r"{}\semi_dec_times_ch={}.npy".format(res_folder, curr_ch), dec_times)

                # dec_time - in the receiver only
                if fb_packets is not None and fb_packets.src == 'd_fb':  # the last node = Receiver
                    dec_times = np.array(self.ac_node.dec_times.fifo_items())
                    np.save(r"{}\dec_times.npy".format(res_folder), dec_times)

                ct_type = self.ac_node.ct_type_hist
                np.save(r"{}\trans_types_ch={}.npy".format(res_folder, curr_ch), ct_type)

                # log erasure series:
                erasure_ = 1 if ff_recep_flag else 0  # 0=erasure, 1=reception
                self.hist_erasures.append(erasure_)
                np.save(r"{}\erasures_ch={}.npy".format(res_folder, curr_ch), np.array(self.hist_erasures))

                # log epsilons:
                eps_mean_hist = self.ac_node.eps_hist
                np.save(r"{}\eps_mean_ch={}.npy".format(res_folder, curr_ch), np.array(eps_mean_hist))

            ####################################################################################################

            # End of enc code. Notice that I used a placeholder in order to remind ourselves that perhaps we want
            # to save the data actually used in the current step of the algorithm in separate buffers

            if self.debug:
                input_window = [pct.time for pct in ff_packets_hist]
                if len(input_window) > 0:

                    self.out_ff.nc_header = [input_window[0], input_window[-1]]
                else:
                    self.out_ff.nc_header = None
                #
                if self.fb_packets_received > 0:
                    curr_ch_state = ff_ch[-1] if isinstance(ff_ch, list) else ff_ch
                    # self.out_cur_fb.nc_header = curr_ch_state

                if ff_packets.nc_header is not None:
                    self.store_nc_enc.put(ff_packets)
                    self.store_channel_stats.put(ff_ch)

                self.out_ff.fec_type = random.choice(['FEC', 'NEW'])

                nc_enc_items = self.store_nc_enc.fifo_items()
                channel_stats_items = self.store_channel_stats.fifo_items()

                print('-----------------------    Packet (out)   ------------------------------------')
                if len(nc_enc_items) > 0:
                    print(f"[{nc_enc_items[0].nc_serial} -> {nc_enc_items[-1].nc_serial}]")
                print('-----------------------    FB (out)     ------------------------------------')
                if self.fb_packets_received > 0:
                    for curr_fb_ch in channel_stats_items:
                        print(f"{curr_fb_ch}")
                # print('-----------------------    END (out)     ------------------------------------')
                print('\n')

    def put_ff(self, packet, ch_state):
        """Sends a packet to this element."""
        self.ff_packets_received += 1

        # if self.element_id is not None:
        #     packet.perhop_time[self.element_id] = self.env.now

        self.store_ff_ch.put(ch_state)
        return self.store_ff.put(packet)

    def put_fb(self, packet, ch_state):
        """Sends a packet to this element."""
        self.fb_packets_received += 1

        # if self.element_id is not None:
        #     packet.perhop_time[self.element_id] = self.env.now

        self.store_fb_ch.put(ch_state)
        return self.store_fb.put(packet)
