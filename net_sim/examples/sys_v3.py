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
from net_sim.ns.packet.dist_generator import DistPacketGenerator
from net_sim.ns.packet.sink import PacketSink
from net_sim.ns.port.airinterface import AirInterface
from net_sim.ns.port.port import Port
from net_sim.ns.port.buffer_manager import BufferManeger
from net_sim.ns.port.nc_enc import NCEncoder
from net_sim.ns.port.nc_node import NC_node
from net_sim.ns.port.termination_node import Termination_node


def constArrival():
    return 1  # time interval


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

    # Topology
    num_of_nodes = 3  # Multi-hop, single path network
    # -------------
    # Instances: |
    # -------------
    # Nodes
    nc_nodes = []
    for curr_node in range(num_of_nodes):
        # Get node default params (note: all debug flags are set to False)
        curr_node_params = get_node_default_params()
        # choose what to display from debug:
        if curr_node == 2:
            curr_node_params['en_enc']['debug'] = True
        nc_nodes.append(NC_node(env, ind=curr_node, **curr_node_params))

    # Channels
    ff_channels = []
    fb_channels = []
    for curr_ch in range(num_of_nodes-1):
        ff_channels.append(AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('from_mat'), wire_id=curr_ch, debug=False))
        fb_channels.append(AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('delay_only'), wire_id=curr_ch, debug=False))

    # Terminations
    source_term = Termination_node(env, node_type='source', arrival=constArrival, pct_size=constPacketSize, pct_debug=False, sink_debug=False)
    dest_term = Termination_node(env, node_type='destination', arrival=constArrival, pct_size=constPacketSize, pct_debug=False, sink_debug=False)

    # -------------
    # Connections |
    # -------------
    # Feedforward path
    source_term.pct_gen.out = nc_nodes[0].ff_in
    for curr_link in range(num_of_nodes-1):
        nc_nodes[curr_link].ff_out.out = ff_channels[curr_link]
        ff_channels[curr_link].out = nc_nodes[curr_link+1].ff_in
    nc_nodes[curr_link+1].ff_out.out = dest_term.sink

    # Feedback path
    dest_term.pct_gen.out = nc_nodes[num_of_nodes-1].fb_in
    for curr_link in range(num_of_nodes - 1,0, -1):
        nc_nodes[curr_link].fb_out.out = fb_channels[curr_link-1]
        fb_channels[curr_link-1].out = nc_nodes[curr_link-1].fb_in
    nc_nodes[0].fb_out.out = source_term.sink

    env.run(until=timesteps)

    # TODO: Add decoder (that deletes the packet from the buffer)
    # TODO: change _mem to _hist (from the word "history" in the store nem as in the buffer class)
    # TODO: Talk with Adina: I think it is better to have a "local" track of the history inside the nc_enc module. this means that the history is recorded twice: once in the nc_enc and the other in the bufffer_manager. What to do about it?
    # TODO: Header of the FB msg should contain ACK/NACK and the id of the packet it refers to. Print the extra field
    # TODO: put_noise_fb is the same as put_noise_ff. Fix this (erase one?)
    # TODO: Define a panda df that stores traffic (in/out packets) in a way that enables easy analysis
    # TODO Add element_id field to airInterface
    # TODO define FIFO_Store class based on the script in test_store.py. This implements a FIFO memery structure with finite memry dropouts (in the example the memory changes in runtime. Use this as well - define a function that does this)
