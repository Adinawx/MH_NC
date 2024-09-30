"""
A basic example that connects two packet generators to a network wire with
a propagation dec_timea distribution, and then to a packet sink.
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
from ns.port.nc_node import NC_node
from ns.port.termination_node import Termination_node


def constArrival():
    return 1  # time interval


def constPacketSize():
    return 0.0  # bytes, Proportional to the processing time of packets in a node


def constDelay():
    return 2.0  # Delay [steps]


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


def get_node_default_params():
    ff_buffer = {'capacity': float('inf'), 'memory_size': float('inf'), 'debug': False}
    ff_pct = {'arrival_dist': constArrival, 'size_dist': constPacketSize, 'debug': False}
    en_enc = {'enc_default_len': float('inf'), 'channel_default_len': 5, 'debug': False}
    fb_buffer = {'capacity': 5, 'memory_size': 5, 'debug': False}
    fb_pct = {'arrival_dist': constArrival, 'size_dist': constPacketSize, 'debug': False}

    node_default_params = {
        'ff_buffer': ff_buffer,
        'ff_pct': ff_pct,
        'en_enc': en_enc,
        'fb_buffer': fb_buffer,
        'fb_pct': fb_pct,
    }

    return node_default_params


if __name__ == '__main__':
    env = simpy.Environment()
    timesteps = 15

    # Get node default params (note: all debug flags are set to False)
    default_params = get_node_default_params()

    # choose what to display from debug:
    default_params['en_enc']['debug'] = True

    # Topology
    num_of_nodes = 2
    # -------------
    # Components: |
    # -------------
    # Nodes
    nc_nodes = []
    for curr_node in range(num_of_nodes):
        nc_nodes.append(NC_node(env, ind=curr_node, **default_params))

    # Channels
    ff_channels = []
    fb_channels = []
    for curr_ch in range(num_of_nodes-1):
        ff_channels.append(AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('erasure', [0]), wire_id=curr_ch, debug=False))
        fb_channels.append(AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('delay_only'), wire_id=curr_ch, debug=False))

    # Terminations
    source_term = Termination_node(env, node_type='source', arrival=constArrival, pct_size=constPacketSize, pct_debug=False, sink_debug=False)
    dest_term = Termination_node(env, node_type='destination', arrival=constArrival, pct_size=constPacketSize, pct_debug=False, sink_debug=False)

    # # Source
    # source_term = Termination_node(env, node_type='source', arrival=constArrival, pct_size=constPacketSize, pct_debug=False, sink_debug=False)
    # # Node 0
    # nc_node0 = NC_node(env, ind=0, **default_params)
    # # Channel 0
    # ff_ch_0 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('from_mat'), wire_id=0, debug=False)
    # fb_ch_0 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('delay_only'), wire_id=10, debug=False)
    # # Node 1
    # nc_node1 = NC_node(env, ind=1, **default_params)
    # # Sinks
    # dest_term = Termination_node(env, node_type='destination', arrival=constArrival, pct_size=constPacketSize, pct_debug=False, sink_debug=False)

    # -------------
    # Connections |
    # -------------
    # Feedforward path
    # source_term.pct_gen.out = nc_node0.ff_in
    # nc_node0.ff_out.out = ff_ch_0
    # ff_ch_0.out = nc_node1.ff_in
    # nc_node1.ff_out.out = dest_term.sink
    #
    # # Feedback path
    # dest_term.pct_gen.out = nc_node1.fb_in
    # nc_node1.fb_out.out = fb_ch_0
    # fb_ch_0.out = nc_node0.fb_in
    # nc_node0.fb_out.out = source_term.sink

    source_term.pct_gen.out = nc_nodes[0].ff_in
    nc_nodes[0].ff_out.out = ff_channels[0]
    ff_channels[0].out = nc_nodes[1].ff_in
    nc_nodes[1].ff_out.out = dest_term.sink

    # Feedback path
    dest_term.pct_gen.out = nc_nodes[1].fb_in
    nc_nodes[1].fb_out.out = fb_channels[0]
    fb_channels[0].out = nc_nodes[0].fb_in
    nc_nodes[0].fb_out.out = source_term.sink

    env.run(until=timesteps)

    # TODO: Add decoder (that deletes the packet from the buffer)
    # TODO: change _mem to _hist (from the word "history" in the store nem as in the buffer class)
    # TODO: Talk with Adina: I think it is better to have a "local" track of the history inside the nc_enc module. this means that the history is recorded twice: once in the nc_enc and the other in the bufffer_manager. What to do about it?
    # TODO: Header of the FB msg should contain ACK/NACK and the id of the packet it refers to
    # TODO: put_noise_fb is the same as put_noise_ff. Fix this (erase one?)
    # TODO: Define a panda df that stores traffic (in/out packets) in a way that enables easy analysis
    # TODO Add element_id field to airInterface
    # TODO define FIFO_Store class based on the script in test_store.py. This implements a FIFO memery structure with finite memry dropouts (in the example the memory changes in runtime. Use this as well - define a function that does this)
