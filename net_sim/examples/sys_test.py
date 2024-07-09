
"""
A basic example that connects two packet generators to a network wire with
a propagation delay distribution, and then to a packet sink.
"""

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

def constArrival():
    return 1   # time interval

def constPacketSize():
    return 0.0  # bytes, Proportional to the processing time of packets in a node

def constDelay():
    return 3.0  # Delay [steps]


def noise_param(noise_type, *args):
    if noise_type == 'delay_only':
        noise_dict = {
            'type': noise_type,
            'debug': False
        }
    elif noise_type == 'Gaussian':  # Example for args: args = [0,1]
        noise_dict = {
            'type': noise_type,
            'mean': args[0],
            'variance': args[1],
            'debug': False
        }
    elif noise_type == 'erasure':  # Example for args: args = [0.1]
        noise_dict = {
            'type': noise_type,
            'p_e': args[0],
            'debug': False
        }
    elif noise_type == 'from_mat':  # Example for args: args = 'C:\\Users\\tmp.mat'
        noise_dict = {
            'type': noise_type,
            'path': 'C:\\Users\\shaigi\\Desktop\\deepNP\\SINR\\SINR_Mats\\scenario_fast\\sinr_mats_test\\SINR(111).mat',
            'debug': False
        }
    else:
        print(["Wrong input to noise_param. Num of input params:" + str(len(args))])
        return None

    return noise_dict


def curr_loc() -> str:
    file_name = os.path.basename(__file__)
    return f"File: {file_name}, Line: {sys._getframe().f_lineno}"

if __name__ == '__main__':
    env = simpy.Environment()

    # -------------
    # Components: |
    # -------------

    # Source
    pg_s = DistPacketGenerator(env, "s_ff", constArrival, constPacketSize, msg_type='s_ff', flow_id=100, nc_header=None, debug=False)
    source_sink = PacketSink(env, rec_flow_ids=False, element_id='source_sink', debug=False)
    # Node 0
    ff_buffer_0 = BufferManeger(env, rate=0.0, qlimit=None, element_id='buff_ff_node0', debug=True)
    ff_pct_0 = DistPacketGenerator(env, "ff_node0", constArrival, constPacketSize, msg_type='ff', flow_id=0, debug=False)
    # en_enc_0 = NCEncoder(env, element_id='enc_node0', debug=False)
    fb_buffer_0 = BufferManeger(env, rate=0.0, qlimit=None, element_id='buff_fb_node0', debug=False)
    fb_pct_0 = DistPacketGenerator(env, "fb_node0", constArrival, constPacketSize, msg_type='fb', flow_id=10, debug=False)
    # Channel 0
    ff_ch_0 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('from_mat'), wire_id=0, debug=False)
    fb_ch_0 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('delay_only'), wire_id=10, debug=False)
    # Node 1 (Destination)
    ff_buffer_1 = BufferManeger(env, rate=0.0, qlimit=None, element_id='buff_ff_node1',  debug=False)
    ff_pct_1 = DistPacketGenerator(env, "ff_node1", constArrival, constPacketSize,  msg_type='ff', flow_id=1, debug=False)
    fb_buffer_1 = BufferManeger(env, rate=0.0, qlimit=None, element_id='buff_fb_node1',  debug=False)
    fb_pct_1 = DistPacketGenerator(env, "ff_node1", constArrival, constPacketSize,  msg_type='fb', flow_id=11, debug=False)
    # Sinks
    dest_sink = PacketSink(env, rec_flow_ids=False, element_id='dest_sink', debug=False)
    pg_d = DistPacketGenerator(env, "d_fb", constArrival, constPacketSize, msg_type='d_fb', flow_id=200, nc_header=None, debug=False)

    # -------------
    # Connections |
    # -------------

    # Feedforward path
    pg_s.out = ff_buffer_0
    ff_buffer_0.out_ff = ff_pct_0
    ff_pct_0.out = ff_ch_0
    ff_ch_0.out = ff_buffer_1
    ff_buffer_1.out_ff = ff_pct_1  # MY CHANGES 29/5
    ff_pct_1.out = dest_sink

    # Feedback path
    pg_d.out = fb_buffer_1
    fb_buffer_1.out_ff = fb_pct_1
    fb_pct_1.out = fb_ch_0
    fb_ch_0.out = fb_buffer_0
    fb_buffer_0.out_ff = fb_pct_0  # MY CHANGES 29/5
    fb_pct_0.out = source_sink

    env.run(until=15)

    # TODO: Drop the _ff from everywhere
    # TODO: put_noise_fb is the same as put_noise_ff. Fix this (erase one?)



