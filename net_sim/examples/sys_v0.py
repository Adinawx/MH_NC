
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

    # Source
    pg_s = DistPacketGenerator(env, "s_ff", constArrival, constPacketSize, msg_type='s_ff', flow_id=100, nc_header=None, debug=False)
    source_sink = PacketSink(env, rec_flow_ids=False, debug=False)
    # Node 0
    port_0 = Port(env, rate=0.0, qlimit=None, element_id='port_node0', debug=True)
    ff_pct_0 = DistPacketGenerator(env, "ff_node0", constArrival, constPacketSize, msg_type='ff', flow_id=0)
    fb_pct_0 = DistPacketGenerator(env, "fb_node0", constArrival, constPacketSize, msg_type='fb', flow_id=1)
    # Channel 0
    ff_ch_0 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('from_mat'), wire_id=0, debug=True)
    fb_ch_0 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('delay_only'), wire_id=1, debug=True)
    # Node 1 (Destination)
    port_1 = Port(env, rate=0.0, qlimit=None, element_id='port_node1',  debug=True)
    ff_pct_1 = DistPacketGenerator(env, "ff_node1", constArrival, constPacketSize,  msg_type='ff', flow_id=2)
    fb_pct_1 = DistPacketGenerator(env, "fb_node1", constArrival, constPacketSize, msg_type='fb', flow_id=3)
    # Sinks
    pg_d = DistPacketGenerator(env, "fb_node0", constArrival, constPacketSize, msg_type='d_fb', flow_id=200)
    dest_sink = PacketSink(env, rec_flow_ids=False, debug=True)

    # Connections:
    # ------------
    # Feedforward path
    pg_s.out = port_0
    port_0.out_ff = ff_pct_0
    ff_pct_0.out = ff_ch_0
    ff_ch_0.out = port_1
    port_1.out_ff = ff_pct_1  # MY CHANGES 29/5
    ff_pct_1.out = dest_sink

    # Feedback path
    pg_d.out = port_1
    port_1.out_fb = fb_pct_1
    fb_pct_1.out = fb_ch_0
    fb_ch_0.out = port_0
    port_0.out_fb = fb_pct_0
    fb_pct_0.out = source_sink

    env.run(until=15)

    # TODO: Add channel_type. Of course, the channel should not depend on packet_type...
    # TODO: Think about what to do with the sink nodes
    # TODO: Build a dummy block and test the ability to pass buffer content
    # TODO: Split to two ports (buffer manager)


    # pg2 = DistPacketGenerator(env, "s_fb", constArrival, packet_size, flow_id=1)
    # wire2 = Wire(env, constDelay, wire_id=2, debug=True)
    # pg2.out = wire2
    # wire2.out = ps

    # ch = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('erasure', [0.1]), wire_id=0, debug=False)
    # ch = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('from_mat'), wire_id=0, debug=False)
    # wire_0 = Wire(env, constDelay, wire_id=0, debug=False)  # Undo to fix bug 2/6
    # wire_1 = Wire(env, constDelay, wire_id=1, debug=True)  # Undo to fix bug 2/6



    # Channel 1
    # ff_ch_1 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('from_mat'), wire_id=2, debug=True)
    # fb_ch_1 = AirInterface(env, delay_dist=constDelay, noise_dict=noise_param('delay_only'), wire_id=3, debug=True)


    # print(
    #     "Flow 1 packet delays: "
    #     + ", ".join(["{:.2f}".format(x) for x in dest_sink.waits["flow_1"]])
    # )
    # print(
    #     "Flow 2 packet delays: "
    #     + ", ".join(["{:.2f}".format(x) for x in dest_sink.waits["flow_2"]])
    # )
    #
    # print(
    #     "Packet arrival times in flow 1: "
    #     + ", ".join(["{:.2f}".format(x) for x in dest_sink.arrivals["flow_1"]])
    # )
    #
    # print(
    #     "Packet arrival times in flow 2: "
    #     + ", ".join(["{:.2f}".format(x) for x in dest_sink.arrivals["flow_2"]])
    # )
