import numpy as np
from ns.port.fifo_store import FIFO_Store
from acrlnc_node.ac_node import ACRLNC_Node

class BlankSpacesNode(ACRLNC_Node):
    def __init__(self):
        super().__init__()

