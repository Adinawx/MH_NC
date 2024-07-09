"""
Implements a port with an output buffer, given an output rate and a buffer size (in either bytes
or the number of packets). This implementation uses the simple tail-drop mechanism to drop packets.
"""
import simpy


class Port:
    """Models an output port on a switch with a given rate and buffer size (in either bytes
    or the number of packets), using the simple tail-drop mechanism to drop packets.

    Parameters
    ----------
    env: simpy.Environment
        the simulation environment.
    rate: float
        the bit rate of the port (0 for unlimited).
    element_id: int
        the element id of this port.
    qlimit: integer (or None)
        a queue limit in bytes or packets (including the packet in service), beyond
        which all packets will be dropped.
    limit_bytes: bool
        if True, the queue limit will be based on bytes; if False, the queue limit
        will be based on packets.
    zero_downstream_buffer: bool
        if True, assume that the downstream element does not have any buffers,
        and backpressure is in effect so that all waiting packets queue up in this
        element's buffer.
    debug: bool
        If True, prints more verbose debug information.
    """

    def __init__(
            self,
            env,
            rate: float,
            qlimit: int = None,
            limit_bytes: bool = False,
            zero_downstream_buffer: bool = False,
            element_id: int = None,
            debug: bool = False,
    ):
        self.store = simpy.Store(env)
        self.store_mem = simpy.Store(env)  # MY CHANGE 26/5
        self.store_channel_ff = simpy.Store(env)  # Stores ff channel state (SNR/Erasure/..) MY CHANGE 2/6
        self.store_channel_fb = simpy.Store(env)  # Stores fb channel state (SNR/Erasure/..). NA when fb channel clean MY CHANGE 2/6
        self.store_fb = simpy.Store(env)  # MY CHANGE 3/6
        self.store_fb_mem = simpy.Store(env)  # MY CHANGE 3/6
        self.ch_type = None  # MY CHANGE 2/6
        self.rate = rate
        self.env = env
        self.out_ff = None  # MY CHANGES 4/6 (find@replace self.out -> self.out_ff
        self.out_fb = None  # MY CHANGES 4/6
        self.packets_received = 0
        self.fb_packets_received = 0
        self.packets_dropped = 0
        self.fb_packets_dropped = 0
        self.qlimit = qlimit
        self.limit_bytes = limit_bytes
        self.byte_size = 0  # the current size of the queue in bytes
        self.fb_byte_size = 0  # the current size of the queue in bytes
        self.element_id = element_id

        self.zero_downstream_buffer = zero_downstream_buffer
        if self.zero_downstream_buffer:
            self.downstream_store = simpy.Store(env)

        self.debug = debug
        self.busy = 0  # used to track if a packet is currently being sent
        self.busy_packet_size = 0

        self.action = env.process(self.run())

    def update(self, packet):
        """
        The packet has just been retrieved from this element's own buffer by a downstream
        node that has no buffers.
        """
        # There is nothing that needs to be done, just print a debug message
        if self.debug:
            print(f"Retrieved Packet {packet.packet_id} from flow {packet.flow_id}.")

    def run(self):
        """The generator function used in simulations."""
        while True:
            if self.zero_downstream_buffer:
                packet = yield self.downstream_store.get()
            else:
                packet = yield self.store.get()

            # ff_noise = yield self.store_channel_ff.get()
            # fb_noise = yield self.store_channel_fb.get()
            # fb_packet = yield self.store_fb.get()  # MY CHANGES 3/6

            self.memory_buffer = yield self.store_mem.put(packet)  # MY CHANGE 26/5
            # self.fb_memory_buffer = yield self.store_fb_mem.put(fb_packet)
            # if self.debug:  # MY CHANGE 26/5
            #     print(f"Memory queue length at port: {len(self.store_mem.items)} packets.")

            self.busy = 1
            self.busy_packet_size = packet.size

            if self.rate > 0:
                yield self.env.timeout(packet.size * 8.0 / self.rate)
                self.byte_size -= packet.size

            ### MY CHANGE 27/5 NC CODING PLACEHOLDER #####################

            print(f'In port: id: {self.element_id}, packet_id: {packet.packet_id}, channel buffer: {self.store_channel_ff.items}')  # DEBUG

            mem_items = self.store_mem.items
            nc_serials = [item.nc_serial for item in mem_items]
            self.out_ff.nc_header = [nc_serials[0], nc_serials[-1]]  # MY CHANGE 27/5
            self.out_fb.nc_header = [nc_serials[0], nc_serials[-1]]  # MY CHANGE 27/5
            ### END MY CHANGE 27/5 #################

            ## MY CHANGES - COMMENT:
            # if self.zero_downstream_buffer:
            #     self.out_ff.put(
            #         packet, upstream_update=self.update, upstream_store=self.store
            #     )
            # else:
            #     self.out_ff.put(packet)
            ## END MY CHANGES - COMMENT:
            self.busy = 0
            self.busy_packet_size = 0

    def put(self, packet):
        if packet.msg_type == 'ff':
            """Sends a packet to this element."""
            self.packets_received += 1

            if self.zero_downstream_buffer:
                # If the downstream node has no buffer, packets will be removed
                # from this buffer by the downstream node, and the byte size of the
                # buffer should be recomputed
                self.byte_size = sum(packet.size for packet in self.store.items)

            byte_count = self.byte_size + packet.size

            if self.element_id is not None:
                packet.perhop_time[self.element_id] = self.env.now

            if self.qlimit is None:
                self.byte_size = byte_count
                if self.zero_downstream_buffer:
                    self.downstream_store.put(packet)
                return self.store.put(packet)

            if self.limit_bytes and byte_count >= self.qlimit:
                self.packets_dropped += 1
                if self.debug:
                    print(
                        f"Packet dropped: flow id = {packet.flow_id} and packet id = {packet.packet_id}"
                    )
            elif not self.limit_bytes and len(self.store.items) >= self.qlimit - 1:
                self.packets_dropped += 1
                if self.debug:
                    print(
                        f"Packet dropped: flow id = {packet.flow_id}, packet id = {packet.packet_id}"
                    )
            else:
                # If the packet has not been dropped, record the queue length at this port
                if self.debug:
                    print(f"Queue length at port: {len(self.store.items)} packets.")

                self.byte_size = byte_count

                if self.zero_downstream_buffer:
                    self.downstream_store.put(packet)

                return self.store.put(packet)
        elif packet.msg_type == 'fb':
            """Sends a packet to this element."""
            self.fb_packets_received += 1

            if self.zero_downstream_buffer:  # Is needed? (4/6)
                # If the downstream node has no buffer, packets will be removed
                # from this buffer by the downstream node, and the byte size of the
                # buffer should be recomputed
                self.fb_byte_size = sum(packet.size for packet in self.store.items)

            fb_byte_count = self.fb_byte_size + packet.size

            if self.element_id is not None:
                packet.perhop_time[self.element_id] = self.env.now

            if self.qlimit is None:
                self.fb_byte_size = fb_byte_count
                if self.zero_downstream_buffer:
                    self.downstream_store.put(packet)
                return self.store_fb.put(packet)

            if self.limit_bytes and fb_byte_count >= self.qlimit:
                self.fb_packets_dropped += 1
                if self.debug:
                    print(
                        f"Packet dropped: flow id = {packet.flow_id} and packet id = {packet.packet_id}"
                    )
            elif not self.limit_bytes and len(self.store.items) >= self.qlimit - 1:
                self.fb_packets_dropped += 1
                if self.debug:
                    print(
                        f"FB Packet dropped: flow id = {packet.flow_id}, packet id = {packet.packet_id}"
                    )
            else:
                # If the packet has not been dropped, record the queue length at this port
                if self.debug:
                    print(f" FB Queue length at port: {len(self.store_fb.items)} packets.")

                self.fb_byte_size = fb_byte_count

                if self.zero_downstream_buffer:
                    self.downstream_store.put(packet)

                return self.store_fb.put(packet)

    def put_noise_ff(self, ch_output):   # MY CHANGE 2/6
        return self.store_channel_ff.put(ch_output)

    def put_noise_fb(self, ch_output):   # MY CHANGE 4/6
        return self.store_channel_fb.put(ch_output)
