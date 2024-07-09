import os
import sys
import simpy
from ns.packet.dist_generator import DistPacketGenerator
from ns.packet.sink import PacketSink
from ns.port.airinterface import AirInterface
from ns.port.port import Port
from ns.port.buffer_manager import BufferManeger
from ns.port.nc_enc import NCEncoder

class FIFO_Store:
    def __init__(self, env):
        self.env = env
        self.buff_size = 5
        self.store = simpy.Store(env, capacity=self.buff_size+3)
        self.action = env.process(self.run())


    def run(self):
        while True:

            store_items = self.store.items
            if len(store_items) >= self.buff_size:
                buff_get = self.store.get()
                buff = self.store.put(self.env.now)
                print([self.env.now, buff_get.value, store_items])
            else:
                buff = self.store.put(self.env.now)
                print([self.env.now, store_items])

            if self.env.now == 10:
                self.buff_size += 1

            yield self.env.timeout(1)



if __name__ == '__main__':
    env = simpy.Environment()
    tmp = FIFO_Store(env)
    env.run(until=15)
