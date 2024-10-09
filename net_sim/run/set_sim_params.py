import os
import sys


def constArrival():
    return 1  # time interval


def constPacketSize():
    return 0.0  # bytes, Proportional to the processing time of packets in a node


def constDelay():
    return 2.0  # Delay [steps] # rtt/2


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
    elif noise_type == 'from_csv':  # Example for args: args = 'C:\\Users\\tmp.mat'
        noise_dict = {
            'type': noise_type,
            'eps': args[0],
            'path': args[1],
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

