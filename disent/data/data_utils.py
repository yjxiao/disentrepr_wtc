import contextlib
import numpy as np


def batch_sampler(indices, batch_size, num_batches=None):
    for i in range(0, len(indices), batch_size):
        if num_batches is not None and i >= num_batches:
            break
        else:
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
