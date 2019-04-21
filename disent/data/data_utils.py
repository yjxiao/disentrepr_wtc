import contextlib
import numpy as np


def batch_sampler(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:(i+batch_size)]


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
