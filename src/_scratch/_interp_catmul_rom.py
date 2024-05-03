# This code is slightly adapted from PyLevy
# TODO: put proper attribution
# TODO: remove for the time being? This code is not used.
# Using directly a linear intepolation seems enough.
import jax.numpy as jnp
from jax._src.typing import Array as JArray

_IDX_TYPE = jnp.int32


def interp_catmul_nd(
    points: JArray, grid: JArray, lower: JArray, upper: JArray
) -> JArray:
    """
    Interpolation of points on a multi-dimensional grid using catmul-rom algorithm.
    This code is compatible with differentiation primitives of Jax.

    points: has shape [n,m] where n is the number of points and m is the len of the shape of the grid
    grid: has shape Y
    lower, upper: have shape (m,) where m is the len of the shape of the grid
    """
    num_points = len(points)
    grid_shape = jnp.array(jnp.shape(grid))
    dims = len(grid_shape)
    dtype = points.dtype
    assert points.shape == (num_points, dims), (points.shape, dims)
    assert lower.shape == (dims,), (lower.shape, grid.shape)
    assert upper.shape == (dims,), (upper.shape, grid.shape)
    # TODO assert lower < upper
    # Centered points
    points_ = (points - lower) * ((grid_shape - 1) / (upper - lower))
    floors = jnp.floor(points_).astype(_IDX_TYPE)
    offsets = points_ - floors
    offsets2 = offsets * offsets
    offsets3 = offsets2 * offsets
    weighters = [
        -0.5 * offsets3 + offsets2 - 0.5 * offsets,
        1.5 * offsets3 - 2.5 * offsets2 + 1.0,
        -1.5 * offsets3 + 2 * offsets2 + 0.5 * offsets,
        0.5 * offsets3 - 0.5 * offsets2,
    ]

    ravel_grid = jnp.ravel(grid)
    result = jnp.zeros((num_points,), dtype=dtype)
    for i in range(1 << (dims * 2)):
        weights = jnp.ones((num_points,), dtype=dtype)
        ravel_offset = 0
        for j in range(dims):
            n = (i >> (j * 2)) % 4
            ravel_offset = ravel_offset * grid_shape[j] + jnp.maximum(  # type: ignore
                0, jnp.minimum(grid_shape[j] - 1, floors[:, j] + (n - 1))
            )
            weights = weights * weighters[n][:, j]

        result = result + weights * jnp.take(ravel_grid, ravel_offset)
    return result


def interp_catmul_1d(xs: JArray, tab_xs: Array, tab_ys: Array) -> JArray:
    """
    Interpolate the values of a function at the points `xs` using the Catmull-Rom algorithm.

    Values are expected to be 1D arrays.

    This is probably not as fast as `jax.numpy.interpolate.interp1d`
      but it is more accurate, as of jax==0.4.19
    """
    j_tab_ys = jnp.asarray(tab_ys)
    n = tab_xs.shape[0]
    x_norm: JArray = (n - 1) * (
        (xs - np.min(tab_xs)) / (np.max(tab_xs) - np.min(tab_xs))
    )
    x_norm_fl: JArray = jnp.floor(x_norm)
    idxs = x_norm_fl.astype(jnp.int32)
    offsets = x_norm - x_norm_fl
    p_m1 = j_tab_ys[idxs - 1]
    p_0 = j_tab_ys[idxs]
    p_1 = j_tab_ys[idxs + 1]
    p_2 = j_tab_ys[idxs + 2]

    t = offsets
    y_interp = (
        t * ((2 - t) * t - 1) * p_m1
        + (t * t * (3 * t - 5) + 2) * p_0
        + t * ((4 - 3 * t) * t + 1) * p_1
        + (t - 1) * t * t * p_2
    ) / 2
    return y_interp
