#
# """
# A basic example that connects two packet generators to a network wire with
# a propagation dec_timea distribution, and then to a packet sink.
# """
#
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


def combine_dicts(dict1, dict2):
    """
    combine dict1 and dict2, giving priority to dict2
    """
    combined_dict = dict1.copy()
    combined_dict.update(dict2)
    return combined_dict
#
class NC_node:
    def __init__(self, env, cfg, ind, **kwargs):
        self.env = env
        self.cfg = cfg
        self.ind = ind
        self.node_type = None

        # Init default values (for the class):
        self.ff_buffer_defaults = {'rate': 0.0, 'qlimit': None, 'element_id': 'buff_ff_node'+str(ind)}
        self.ff_pct_defaults = {'element_id': 'ff_node' + str(ind), 'msg_type': 'ff', 'flow_id': 0}
        self.nc_enc_defaults = {'element_id': 'enc_node'+str(ind)}
        self.fb_buffer_defaults = {'rate': 0.0, 'qlimit': None, 'element_id': 'buff_fb_node' + str(ind)}
        self.fb_pct_defaults = {'element_id': 'fb_node' + str(ind), 'msg_type': 'fb', 'flow_id': 10}

        # Get inputs values:
        self.ff_buffer_kwargs = kwargs.pop('ff_buffer', {})
        self.ff_pct_kwargs = kwargs.pop('ff_pct', {})
        self.nc_enc_kwargs = kwargs.pop('en_enc', {})
        self.fb_buffer_kwargs = kwargs.pop('fb_buffer', {})
        self.fb_pct_kwargs = kwargs.pop('fb_pct', {})

        # Combine dictionaries (giving priority to the input params over the default ones:
        self.ff_buffer_params = combine_dicts(self.ff_buffer_defaults, self.ff_buffer_kwargs)
        self.ff_pct_params = combine_dicts(self.ff_pct_defaults, self.ff_pct_kwargs)
        self.nc_enc_params = combine_dicts(self.nc_enc_defaults, self.nc_enc_kwargs)
        self.fb_buffer_params = combine_dicts(self.fb_buffer_defaults, self.fb_buffer_kwargs)
        self.fb_pct_params = combine_dicts(self.fb_pct_defaults, self.fb_pct_kwargs)

        # Initialize buffers, packet generators, and encoders
        self.ff_buffer = BufferManeger(env, **self.ff_buffer_params)
        self.ff_pct = DistPacketGenerator(env,  **self.ff_pct_params)
        self.en_enc = NCEncoder(env, cfg, **self.nc_enc_params)
        self.fb_buffer = BufferManeger(env, **self.fb_buffer_params)
        self.fb_pct = DistPacketGenerator(env,  **self.fb_pct_params)

        # -------------
        # Connections |
        # -------------
        # Define in and out of the module (for convenience):
        self.ff_in = self.ff_buffer
        self.ff_out = self.ff_pct
        self.fb_in = self.fb_buffer
        self.fb_out = self.fb_pct

        # Feedforward connections:
        self.ff_buffer.out = self.en_enc
        self.en_enc.out_ff = self.ff_pct

        # Feedback connections:
        self.fb_buffer.out = self.en_enc
        self.en_enc.out_fb = self.fb_pct

    def update_type(self, node_type):
        """
        Update the type of the node. To be used during runtime
        """
        self.node_type = node_type

        if hasattr(self.ff_buffer, 'node_type'):
            self.ff_buffer.node_type = node_type

        if hasattr(self.ff_pct, 'node_type'):
            self.ff_pct_param.node_type = node_type

        if hasattr(self.en_enc, 'node_type'):
            self.en_enc.node_type = node_type

        if hasattr(self.fb_buffer, 'node_type'):
            self.fb_buffer.node_type = node_type

        if hasattr(self.fb_pct, 'node_type'):
            self.fb_pct.node_type = node_type
