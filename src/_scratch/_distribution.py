# mypy: ignore-errors
import jax.numpy as jnp
import jax
import numpy.typing as npt
from typing import List, Tuple
from . import _pylevy_copy as pylevy
from ._interp_catmul_rom import interp_catmul_nd


def _pdf_interp_n0(x, alpha, beta, loc, scale):
    """
    The PDF of the stable-Levy distribution, as parametrized in Nolan's 0 notation.

    All the parameters must be jax arrays, all of the same length.
    """
    print("x", x)
    # TODO: parameters checks.
    xs_cent = (jnp.asarray(x) - jnp.asarray(loc)) / scale
    points = jnp.stack([jnp.arctan(xs_cent), alpha, beta], axis=1)
    assert points.shape == (len(xs_cent), 3)

    tab_pdf = pylevy._read_from_cache("pdf")
    res = interp_catmul_nd(points, tab_pdf, pylevy._lower, pylevy._upper)
    res = res / scale
    return res


def _canonicalize_input(l: List[npt.ArrayLike]) -> Tuple[List[jax.Array], bool]:
    """
    Given a set of various sorts of inputs, returns a list of 1d arrays that all have the
     same lengths.
    """

    def _can_elt(t) -> jax.Array:
        arr = jnp.asarray(t)
        if arr.ndim == 0:
            arr = arr[jnp.newaxis, :]
        if arr.ndim > 1:
            raise Exception(
                f"Input should be scalar or 1d array, one array has dim {arr.ndim}"
            )
        return arr

    arrs = [_can_elt(t) for t in l]
    sizes = set(len(arr) for arr in arrs)
    if len(sizes) < 1:
        return [], False
    if len(sizes) > 2:
        raise Exception(f"Size mismatch: {[arr.shape for arr in arrs]}")
    if sizes == {1}:
        # TODO: missing the case with size-1 1D arrays
        return arrs, True
    max_len = max(sizes)

    def _check_size(arr: jax.Array) -> jax.Array:
        if arr.size == max_len:
            return arr
        if arr.size == 1:
            return jnp.tile(arr, max_len)
        # TODO: error here
        assert False, f"Size mismatch: {arr.size} vs {max_len}"

    return [_check_size(arr) for arr in arrs], False
