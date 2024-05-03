import os
from typing import Dict

import numpy as np
import jax.numpy as jnp
from jax import Array as JArray
from numpy.typing import ArrayLike as NPArray

_ROOT = os.path.dirname(os.path.abspath(__file__))
_data_cache: Dict[str, NPArray] = {}


# TODO: unsure if it should be used at all.
def read_from_cache(key: str) -> NPArray:
    """
    Reads the cache from the file system.

    The cache is stored in the same directory as this file.
    """
    if key not in _data_cache:
        _data_cache[key] = np.load(os.path.join(_ROOT, "{}.npy".format(key)))["arr_0"]
    return _data_cache[key]


def jax_read_from_cache(key: str) -> JArray:
    return jnp.load(os.path.join(_ROOT, "{}.npy".format(key)))
