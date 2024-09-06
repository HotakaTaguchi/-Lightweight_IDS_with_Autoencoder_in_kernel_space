
"""
Module interfacing between the C files and python
"""
import ctypes
import ctypes.util
import os
import sys
import time

import numpy as np


def load_c_lib(library):
    """
    Load C shared library
    :param library: 
    :return:
    """
    try:
        c_lib = ctypes.CDLL(f"{os.path.dirname(os.path.abspath(__file__))}/{library}")
    except OSError as e:
        print(f"Unable to load the requested C library: {e}")
        sys.exit()
    return c_lib

def ensure_contiguous(array):
    """
    Ensure that array is contiguous
    :param array:
    :return:
    """
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array


def run_mlp(x, c_lib):
    """
    Call 'run_mlp' function from C in Python
    :param x:
    :param c_lib:
    :return:
    """
    M = len(x[0])
    #print(x, N, M)
    x = x.flatten()
    x = ensure_contiguous(x.numpy())
    x = x.astype(np.int64)
    class_indices = ensure_contiguous(np.zeros(M, dtype=np.int64))

    c_long_p = ctypes.POINTER(ctypes.c_longlong)
    #c_int_p = ctypes.POINTER(ctypes.c_int)
    #c_uint_p = ctypes.POINTER(ctypes.c_uint)

    c_run_mlp = c_lib.run_mlp
    c_run_mlp.argtypes = (c_long_p, c_long_p )
    c_run_mlp.restype = None

    _x = x.ctypes.data_as(c_long_p)
    _class_indices = class_indices.ctypes.data_as(c_long_p )

    start = time.perf_counter()
    c_run_mlp(_x, _class_indices)
    end = time.perf_counter()

    pred = np.ctypeslib.as_array(class_indices, M)
    x = np.ctypeslib.as_array(x, M)

    elapsed = end - start
    
    return x, pred, elapsed

