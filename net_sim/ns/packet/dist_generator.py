"""
Implements a packet generator that simulates the sending of packets with a
specified inter- arrival time distribution and a packet size distribution. One
can set an initial delay and a finish time for packet generation. In addition,
one can set the source id and flow ids for the packets generated. The
DistPacketGenerator's `out` member variable is used to connect the generator to
any network element with a `put()` member function.
"""


"""
msg_types: 
ff - feedforward
fb - feedback
s_ff - generates the packets to be fed into the source node
d_fb - generates the packets to be fed into the destination node (as dummy packets)
"""
from net_sim.ns.packet.packet import Packet


class DistPacketGenerator:
    """Generates packets with a given inter-arrival time distribution.

    Parameters
    ----------
    env: simpy.Environment
        The simulation environment.
    element_id: str
        the ID of this element.
    arrival_dist: function
        A no-parameter function that returns the successive inter-arrival times
        of the packets.
    size_dist: function
        A no-parameter function that returns the successive sizes of the
        packets.
    initial_delay: number
        Starts generation after an initial delay. Defaults to 0.
    finish: number
        Stops generation at the finish time. Defaults to infinite.
    rec_flow: bool
        Are we recording the statistics of packets generated?
    """

    def __init__(
        self,
        env,
        element_id,
        arrival_dist,
        size_dist,
        initial_delay=0,
        finish=None,
        size=None,
        flow_id=0,
        rec_flow=False,
        debug=False,
        nc_header=None,  # MY CHANGES 27/5
        nc_serial=None,  # MY CHANGES 30/5
        msg_type=None,  # feefforward (ff) or feedback (fb) MY CHANGE 3/6
        fec_type=None
    ):
        self.element_id = element_id
        self.env = env
        self.arrival_dist = arrival_dist
        self.size_dist = size_dist
        self.initial_delay = initial_delay
        self.finish = float("inf") if finish is None else finish
        self.size = float("inf") if size is None else size
        self.out = None
        self.packets_sent = 0
        self.sent_size = 0
        self.action = env.process(self.run())

        self.flow_id = flow_id

        self.rec_flow = rec_flow
        self.time_rec = []
        self.size_rec = []

        self.nc_header = nc_header  # MY CHANGES 27/5
        if nc_serial is None:
            self.nc_serial = -1  # MY CHANGES 30/5
        else:
            self.nc_serial = nc_serial  # MY CHANGES 30/5
        self.msg_type = msg_type

        self.fec_type = fec_type

        self.debug = debug

    def run(self):
        """The generator function used in simulations."""
        yield self.env.timeout(self.initial_delay)

        while self.env.now < self.finish and self.sent_size < self.size:
            self.nc_serial += 1
            packet = Packet(
                self.env.now,
                self.size_dist(),
                self.packets_sent,
                src=self.element_id,
                flow_id=self.flow_id,
                nc_header=self.nc_header,
                nc_serial=self.nc_serial,
                msg_type=self.msg_type,
                fec_type=self.fec_type,
            )

            if packet.msg_type == 's_ff':
                yield self.out.put_noise_ff(None)
                self.nc_header = self.nc_serial
            elif packet.msg_type == 'd_fb':
                yield self.out.put_noise_fb(None)
                self.nc_header = self.nc_serial + 100

            self.out.put(packet)

            self.packets_sent += 1
            self.sent_size += packet.size

            if self.rec_flow:
                self.time_rec.append(packet.time)
                self.size_rec.append(packet.size)

            if self.debug:
                print(
                    f"Packet Generator {self.element_id}: sent packet {packet.packet_id}  flow_id {packet.flow_id} at time {self.env.now}. payload (type/content): {self.msg_type}/{self.nc_header}")

            # waits for the next transmission
            yield self.env.timeout(self.arrival_dist())
