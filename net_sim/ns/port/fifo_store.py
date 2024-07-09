
import simpy


class FIFO_Store():
    """
    store_type:
        (*) FIFO - when buffer full -> throw old and in with the new
        (*) dropout - when buffer full -> throw new (dropout event)
    """
    def __init__(self, env, store_type='FIFO', capacity=None, memory_size=None, debug=False):
        self.env = env
        self.store_type = store_type
        self.curr_memory = memory_size
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=self.capacity)
        self.debug = debug
        self.items = self.store.items

    def put(self, value):
        store_items = self.store.items
        if len(store_items) >= self.curr_memory:
            if self.store_type == 'FIFO':  # The default store implementation is "dropout"
                buff_get = self.store.get()
            store_return_val = self.store.put(value)
            if self.debug:
                print([self.env.now, buff_get.value, store_items])
        else:
            store_return_val = self.store.put(value)
            if self.debug:
                print([self.env.now, store_items])

        return store_return_val

    def get(self):
        if len(self.store.items) >= 0:
            return self.store.get()
        else:
            return None  # Note: this is not an event (will not activate yield calls..)

    def modify_memory_size(self, new_size):
        """
        Modify the memory size of the store. May be done at any time during runtime
        """
        self.curr_memory = new_size

    def fifo_items(self):
        return self.store.items

    def __len__(self):
        return len(self.store.items)

    def delete_item(self, item, del_indx=False):
        if not del_indx:
            self.store.items.remove(item)
        else:
            assert isinstance(item, int), "Item must be an integer"
            del self.store.items[item]
