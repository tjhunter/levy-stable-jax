"""
The current implementation only supports 1d interpolation.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax._src.typing import Array as JArray

_THRESH = 1e30


def interp_linear(points: JArray, grid: JArray, lower: JArray, upper: JArray) -> JArray:
    """
    Interpolation of points on a multi-dimensional grid.
    This code is compatible with differentiation primitives of Jax.

    points: has shape [n,m] where n is the number of points and m is the len of the
     shape of the grid
    grid: has shape Y
    lower, upper: have shape (m,) where m is the len of the shape of the grid
    """
    grid_shape = jnp.array(jnp.shape(grid))
    points_ = (points - lower) * ((grid_shape - 1) / (upper - lower))
    # The interpolation seems to have NaN issues at the boundary of the grid.
    # TODO: ensure that all the interpolation happens strictly within the boundaries.
    # The mode is supposed to take into account, but it is flaky currently.
    # jax.debug.print(
    #     "interp_linear: nan in points_: {a} nan in grid: {b} inf in grid: {c}
    # all lower: {d} >=0: {e}",
    #     a=jnp.any(jnp.isnan(points_)),
    #     b=jnp.any(jnp.isnan(grid)),
    #     c=jnp.any(jnp.isinf(grid)),
    #     d=jnp.all(points_ <= (grid_shape - 1)),
    #     e=jnp.all(points_ >= 1),
    # )
    # jax.debug.print("interp_linear {x}", x=points_)
    # Infinite values cause issues with the calculation of the gradient.
    # Clipping everything below a threshold.
    grid_ = jnp.clip(grid, a_min=-_THRESH)
    res: JArray = jsp.ndimage.map_coordinates(grid_, points_.T, order=1, mode="constant")  # type: ignore
    # The +- infinity values trigger NaN in the jax interpolation code.
    res = jnp.where(jnp.isnan(res), -jnp.inf, res)
    return res
