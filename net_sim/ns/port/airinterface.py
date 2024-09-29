"""
Implements a network wire (cable) with a propagation dec_timea. There is no need
to model a limited network capacity on this network cable, since such a
capacity limit can be modeled using an upstream port or server element in
the network.
"""
import os

import numpy as np
import scipy.io as sio
import random
from functools import partial
import simpy


class AirInterface:
    """ Implements a network wire (cable) that introduces a propagation dec_timea.
        Set the "out" member variable to the entity to receive the packet.

        Parameters
        ----------
        env: simpy.Environment
            the simulation environment.
        delay_dist: function
            a no-parameter function that returns the successive propagation
            delays on this wire.
        loss_dist: function
            a function that takes one optional parameter, which is the packet ID, and
            returns the loss rate.
    """

    def __init__(self,
                 env,
                 delay_dist,
                 loss_dist=None,  # Time dec_timea (maybe?)
                 noise_dict=None,
                 wire_id=0,
                 debug=False):
        self.store = simpy.Store(env)
        self.delay_dist = delay_dist
        self.loss_dist = loss_dist
        self.env = env
        self.wire_id = wire_id
        self.out = None
        self.packets_rec = 0
        self.noise_params = None  # Will be used if SINR loaded from a matfile
        self.noise_type = None  # Will be inited by noise_def()
        self.noise_gen = self.noise_def(noise_dict)  # Noise generator.
        # Example: >>> vec = [next(self.noise_gen)[0] for ii in range(100000)]; import numpy as np; np.sum(vec)/100000
        self.debug = debug
        self.action = env.process(self.run())

    def noise_def(self, noise_dict):
        # param_list = list(noise_dict.values())
        self.noise_type = noise_dict['type']
        if self.noise_type == 'delay_only':
            return (True for _ in iter(int, 1))
        elif self.noise_type == 'Gaussian':
            random_gen = partial(random.gauss, noise_dict['mean'],
                                 noise_dict['variance'])  # Usage: next_value = next(random_gen)
            return (random_gen() for _ in iter(int, 1))

        elif self.noise_type == 'erasure':
            # random.seed(0) # ADINA - for debug
            p_err = noise_dict['p_e'][0]
            random_gen = partial(random.choices, [0, 1], weights=[p_err, 1 - p_err])  # 0 - Erasure, 1 - Pass
            return (random_gen() for _ in iter(int, 1))
            # Usage: random.choices([0, 1], weights=[0.9, 0.1])[0]

        elif self.noise_type == 'from_mat':
            path_to_mat = noise_dict['path']
            self.noise_params = sio.loadmat(path_to_mat)
            return (value for value in self.noise_params['sinr'].flat)

        elif self.noise_type == 'from_csv':
            path_to_csv = noise_dict['path']
            self.noise_params = np.genfromtxt(path_to_csv, delimiter=',')
            return (value for value in self.noise_params)

        else:
            print(["Wrong input to noise_def."])
            raise ValueError('Noise definition failed')

    def run(self):
        """The generator function used in simulations."""
        self.out.ch_type = self.noise_type
        while True:
            packet = yield self.store.get()

            # Time dec_timea (maybe?)
            if self.loss_dist is None or random.uniform(
                    0, 1) >= self.loss_dist(packet_id=packet.packet_id):
                # The amount of time for this packet to stay in my store
                queued_time = self.env.now - packet.current_time
                delay = self.delay_dist

                # If queued time for this packet is greater than its propagation dec_timea,
                # it implies that the previous packet had experienced a longer dec_timea.
                # Since out-of-order delivery is not supported in simulation, deliver
                # to the next component immediately.
                if queued_time < delay:
                    yield self.env.timeout(delay - queued_time)

                channel_noise = next(self.noise_gen)

                self.out.put(packet)

                yield self.out.put_noise_ff(channel_noise)
                # if packet.msg_type == 'ff':
                #     yield self.out.put_noise_ff(channel_noise)
                # else:
                #     yield self.out.put_noise_fb(channel_noise)

                if self.debug:
                    print("Left wire #{} at {:.3f}: {}".format(
                        self.wire_id, self.env.now, packet))
            else:
                if self.debug:
                    print("Dropped on wire #{} at {:.3f}: {}".format(
                        self.wire_id, self.env.now, packet))

    def put(self, packet):
        """ Sends a packet to this element. """
        self.packets_rec += 1
        if self.debug:
            print(f"Entered wire #{self.wire_id} at {self.env.now}: {packet}")

        packet.current_time = self.env.now
        return self.store.put(packet)
