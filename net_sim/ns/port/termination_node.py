from functools import partial
import random
from random import expovariate
import os
import sys
import simpy
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.port.airinterface import AirInterface
from ns.port.port import Port
from ns.port.buffer_manager import BufferManeger
from ns.port.nc_enc import NCEncoder


class Termination_node:
    def __init__(self, env, node_type, arrival, pct_size, pct_debug=False, sink_debug=False):
        self.env = env

        self.node_type = node_type

        self.pct_arrival = arrival
        self.pct_size = pct_size

        if self.node_type == 'source':
            self.pct_element_id = 's_ff'
            self.pct_msg_type = 's_ff'
            self.pct_flow_id = 100
            self.sink_element_id = 'source_sink'
        elif self.node_type == 'destination':
            self.pct_element_id = 'd_fb'
            self.pct_msg_type = 'd_fb'
            self.pct_flow_id = 200
            self.sink_element_id = 'sdest_sink'

        self.pct_gen = DistPacketGenerator(env,
                                           element_id=self.pct_element_id,
                                           arrival_dist=self.pct_arrival,
                                           size_dist=self.pct_size,
                                           msg_type=self.pct_msg_type,
                                           flow_id=self.pct_flow_id,
                                           nc_header=None,
                                           debug=pct_debug
                                           )
        self.sink = PacketSink(env,
                               rec_flow_ids=False,
                               element_id='source_sink',
                               debug=sink_debug
                               )
