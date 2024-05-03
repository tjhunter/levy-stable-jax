"""
The current implementation only supports 1d interpolation.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.typing import Array as JArray


def interp_linear(points: JArray, grid: JArray, lower: JArray, upper: JArray) -> JArray:
    """
    Interpolation of points on a multi-dimensional grid using catmul-rom algorithm.
    This code is compatible with differentiation primitives of Jax.

    points: has shape [n,m] where n is the number of points and m is the len of the
     shape of the grid
    grid: has shape Y
    lower, upper: have shape (m,) where m is the len of the shape of the grid
    """
    grid_shape = jnp.array(jnp.shape(grid))
    points_ = (points - lower) * ((grid_shape - 1) / (upper - lower))
    res = jsp.ndimage.map_coordinates(grid, points_.T, order=1, mode="nearest")  # type: ignore
    # The +- infinity values trigger NaN in the jax interpolation code.
    res = jnp.where(jnp.isnan(res), -jnp.inf, res)
    return res
