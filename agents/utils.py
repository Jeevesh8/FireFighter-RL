from functools import reduce
import numpy as np

def hash_bool_array(arr):
    """Hashes a boolean ndarray.
    Flattens and converts to a packed boolean tuple.

    Returns:
        A tuple: (packed numpy array, shape of original array)
    """
    return tuple(np.packbits(np.reshape(np.copy(arr), (-1)))), arr.shape

def revert_hash(hash):
    """Computes the array given the hash.
    Inverse of the function ``hash_bool_array``.

    Returns:
        np.ndarray, whose hash is the one provided as argument.
    """
    total_size = reduce(lambda x,y: x*y, hash[1])
    return np.reshape( np.unpackbits(np.array(np.copy(hash[0])))[:total_size], hash[1])

class numpy_dict(dict):
    """Implements a dictionary with numpy boolean arrays as its keys.
    This dictionary implementation is corresponding to the agents and environments,
    where the state is a pair of boolean arrays.
    """
    def __init__(self, *args, **kwargs):
        super(bool).__init__(*args, **kwargs)

    def __getitem__(self, burned_n_defended):
        burned, defended = burned_n_defended
        return super().__getitem__((hash_bool_array(burned), hash_bool_array(defended)))
    
    def __setitem__(self, burned_n_defended, value):
        burned, defended = burned_n_defended
        return super().__setitem__((hash_bool_array(burned), hash_bool_array(defended)), value)
