"""
Implements a port with an output buffer, given an output rate and a buffer size (in either bytes
or the number of packets). This implementation uses the simple tail-drop mechanism to drop packets.
"""
import simpy
from net_sim.ns.port.fifo_store import FIFO_Store
import random  # For debug only. Erase eventually
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
        element_id: int = None,
        debug: bool = False,
        enc_default_len: float = float('inf'),
        channel_default_len: float = float('inf'),
    ):
        self.env = env
        self.out_ff = None
        self.out_fb = None
        self.element_id = element_id
        self.ff_packets_received = 0
        self.ff_packets_dropped = 0
        self.fb_packets_received = 0
        self.fb_packets_dropped = 0
        self.enc_default_len = enc_default_len
        self.channel_default_len = channel_default_len
        self.debug = debug

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
        self.store_channel_stats = FIFO_Store(env, capacity=float('inf'), memory_size=self.channel_default_len, debug=False)

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
        while True:

            ff_packets = yield self.store_ff.get()
            ff_ch = yield self.store_ff_ch.get()

            if self.fb_packets_received > 0:
                fb_packets = yield self.store_fb.get()
                fb_ch = yield self.store_fb_ch.get()
            if ff_packets.nc_header is not None:
                self.store_ff_hist.put(ff_packets)
                self.store_ff_ch_hist.put(ff_ch)
            ff_packets_hist = self.store_ff_hist.fifo_items()
            ff_ch_hist = self.store_ff_ch_hist.fifo_items()

            if self.fb_packets_received > 0:
                if fb_packets.nc_header is not None:
                    self.store_fb_hist.put(fb_packets)
                    self.store_fb_ch_hist.put(fb_ch)
                fb_packets_hist = self.store_fb_hist.fifo_items()
                fb_ch_hist = self.store_fb_ch_hist.fifo_items()

            if self.debug:
                print(str(self.env.now) + '| ' + str(self.element_id))
                print('-----------------------    FF (in)   ------------------------------------')
                for curr_ff_packet, curr_ff_ch in zip(ff_packets_hist, ff_ch_hist):
                    print(f"{curr_ff_packet} || {curr_ff_ch}")
                print('-----------------------    FB (in)     ------------------------------------')
                if self.fb_packets_received > 0:
                    for curr_fb_packet, curr_fb_ch in zip(fb_packets_hist, fb_ch_hist):
                        print(f"{curr_fb_packet} || {curr_fb_ch}")
                print('-----------------------    END (in)     ------------------------------------')
                print('\n')

            # Put the enc code here...
            input_window = [pct.time for pct in ff_packets_hist]
            if len(input_window) > 0:
                self.out_ff.nc_header = [input_window[0], input_window[-1]]
            else:
                self.out_ff.nc_header = None
            #
            if self.fb_packets_received > 0:
                curr_ch_state = ff_ch[-1] if isinstance(ff_ch, list) else ff_ch
                self.out_fb.nc_header = curr_ch_state

            if ff_packets.nc_header is not None:
                self.store_nc_enc.put(ff_packets)
                self.store_channel_stats.put(ff_ch)
            # End of enc code. Notice that I used a placeholder in order to remind ourselves that perhaps we want
            # to save the data actually used in the current step of the algorithm in separate buffers

            self.out_ff.fec_type = random.choice(['FEC', 'RLNC'])


            if self.debug:
                nc_enc_items = self.store_nc_enc.fifo_items()
                channel_stats_items = self.store_channel_stats.fifo_items()

                print('-----------------------    FF (nc)   ------------------------------------')
                if len(nc_enc_items) > 0:
                    print(f"[{nc_enc_items[0].nc_serial} -> {nc_enc_items[-1].nc_serial}]")
                print('-----------------------    FB (nc)     ------------------------------------')
                if self.fb_packets_received > 0:
                    for curr_fb_ch in channel_stats_items:
                        print(f"{curr_fb_ch}")
                print('-----------------------    END (nc)     ------------------------------------')


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
