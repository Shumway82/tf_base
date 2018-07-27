"""
Copyright (C) Silvio Jurk - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential.

@author: Silvio Jurk
"""

from io import StringIO, BytesIO
from math import log2
import os

import numpy as np


def compute_size(bitstream, max_value=None):
    """ This function compute the size of a bitstream in Bytes.

    A bitstream can be a list, a numpy array, or even a file.
    When examining the bitstream, tuples of less than 4 integers are ignored.

    A `max_value` can be provided to make correct estimations in case
    you don't use the full range of available integers. This way,
    the theoretical size is reduced.

    It is used in the benchmark class to compute the lenght of your butstream (in bytes).

    If you don't like it you can override it in the constructor of the benchmark.
    See the `custom_compute_size` for that.
    """

    # Keys would be types and values would be functions.
    if isinstance(bitstream, tuple):
        # is it the same type?
        if len(bitstream) <= 4 and all(isinstance(x, int) for x in bitstream):
            return 0
        else:
            return sum(compute_size(x, max_value) for x in bitstream)
    elif isinstance(bitstream, np.ndarray):
        if max_value is None:
            return bitstream.nbytes
        else:
            if np.all(bitstream < max_value):
                return bitstream.size * log2(max_value) / 8
            else:
                print("There was an array not conforme to the max_value")
                return bitstream.nbytes
    elif isinstance(bitstream, list):
        # We check if it contains other objects or not.
        if isinstance(bitstream[0], int):
            if max_value is None:
                raise TypeError('When using lists in the bitstream, '
                                'please provide max_value_bitstream.')
            else:
                assert all(x < max_value for x in bitstream)
                return len(bitstream) * log2(max_value) / 8
        else:
            return sum(compute_size(x, max_value) for x in bitstream)
    elif isinstance(bitstream, np.dtype):
        return 0
    elif isinstance(bitstream, str):
        return os.path.getsize(bitstream)
    elif isinstance(bitstream, StringIO) or isinstance(bitstream, BytesIO):
        return bitstream.getbuffer().nbytes
    elif isinstance(bitstream, bytes):
        return len(bitstream)
    else:
        raise KeyError("Bitstream type {} not supported.".format(type(bitstream)))
