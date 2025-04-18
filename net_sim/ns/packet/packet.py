"""
A very simple class that represents a packet.
"""


class Packet:
    """
    Packets in ns.py are generally created by packet generators, and will run
    through a queue at an output port.

    Key fields include: generation time, size, flow_id, packet id, source, and
    destination. We do not model upper layer protocols, i.e., packets don't
    contain a payload. The size (in bytes) field is used to determine its
    transmission time.

    We use a float to represent the size of the packet in bytes so that we can
    compare to ideal M/M/1 queues.

    Parameters
    ----------
    time: float
        the time when the packet is generated.
    size: float
        the size of the packet in bytes
    packet_id: int
        an identifier for the packet
    nc_serial: int
        If ff packet: the id of the packet after processing. The nc_header refers to this field.
        If fb packet: the packet for which the feedback refers to
    src, dst: int
        identifiers for the source and destination
    flow_id: int or str
        an integer or string that can be used to identify a flow
    fec_type: FEC type: support of two types of FEC: FEC (nc repetition code), of NC (nc code, according to the window state)
    """

    def __init__(
        self,
        time,
        size,
        packet_id,
        realtime=0,
        last_ack_time=0,
        delivered=-1,
        src="source",
        dst="destination",
        flow_id=0,
        payload=None,
        tx_in_flight=-1,
        nc_header=None,  # MY CHANGE 27/5
        nc_serial=None,  # MY CHANGES 30/5
        msg_type=None,  # feefforward (ff) or feedback (fb) MY CHANGE 3/6
        fec_type=None,  # MY CHANGE 27/
    ):
        self.time = time
        self.delivered_time = last_ack_time
        self.first_sent_time = 0
        # self.sent_time = 0
        self.size = size
        self.packet_id = packet_id
        self.realtime = realtime
        self.src = src
        self.dst = dst
        self.flow_id = flow_id
        self.payload = payload
        self.lost = 0
        self.self_lost = False
        self.tx_in_flight = tx_in_flight
        if delivered == -1:
            self.delivered = packet_id
        else:
            self.delivered = delivered
        self.out = None  # MY CHANGE 27/5
        if self.src == "s_ff":  # MY CHANGE 27/5
            self.nc_header = time  # MY CHANGE 27/5
        else:
            self.nc_header = nc_header
        self.nc_serial = nc_serial  # MY CHANGES 30/5

        self.msg_type = msg_type  # feedforward (ff) or feedback (fb) MY CHANGE 3/6
        if self.src == "d_fb" or self.src == 'fb':  # MY CHANGE 13/7
            self.fec_type = None
        elif self.src == "s_ff":  # ADINA CHANGE 15/7
            self.fec_type = "NEW"
        else:
            self.fec_type = fec_type

        self.is_app_limited = False
        self.color = None  # Used by the two-rate tri-color token bucket shaper
        self.prio = {}  # used by the Static Priority scheduler
        self.ack = None  # used by TCPPacketGenerator and TCPSink
        self.current_time = 0  # used by the Wire element
        self.perhop_time = {}  # used by Port to record per-hop arrival times

    def __repr__(self):
        # return f"id: {self.packet_id}, nc id: {self.nc_serial}, src: {self.src}, FEC type: {self.fec_type}, size: {self.size}, header: {self.nc_header}, type: {self.msg_type}"  # MY CHANGE 27/5
        return f"(Gen time: {self.time}), id: {self.packet_id}, nc id: {self.nc_serial}, src: {self.src}, FEC type: {self.fec_type}, header: {self.nc_header}, type: {self.msg_type}"  # , size: {self.size} # New - 18/8

