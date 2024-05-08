from typing import Tuple, Union, Sequence
from scipy.stats import levy_stable as sp_levy_stable  # type: ignore
from contextlib import contextmanager
from jax import Array as JArray
import jax.numpy as jnp

from ._typing import Params, Param


@contextmanager
def set_stable(p: Param):
    """
    Manages the parametrization of scipy's levy_stable distribution.

    Since the parametrization of scipy is at the module level, this function
    provides a way to temporarily change the parametrization of the distribution.

    Example:

    ```python
    from scipy.stats import levy_stable as sp_levy_stable
    with set_stable(Params.N0):
        # Will be parametrized as N0 / scipy's S0 instead of default S1.
        _ = sp_levy_stable.rvs(alpha=1.5, beta=0.0, size=10)
    ```

    TODO: turn into doctest, there is an issue with indentation
    """
    curr = sp_levy_stable.parameterization
    if p == Params.N0:
        sp_levy_stable.parameterization = "S0"
    elif p == Params.N1:
        sp_levy_stable.parameterization = "S1"
    else:
        # TODO: proper error
        raise ValueError(f"Invalid parametrization {p}")
    try:
        yield
    finally:
        sp_levy_stable.parameterization = curr


INPUT = JArray | float


def param_convert(
    alpha: INPUT,
    beta: INPUT,
    loc: INPUT,
    scale: INPUT,
    param_from: Param,
    param_to: Param,
) -> Tuple[JArray | float, JArray | float]:
    """
    Shifts the loc and scale of the alpha-stable distribution from
    one parametrization to another.

    Args:
        alpha: The stability parameter of the stable distribution (0-2.0].
        beta: The skewness parameter of the stable distribution.
        loc: The location parameter of the stable distribution.
        scale: The scale parameter of the stable distribution.
        param_from: The initial parametrization of the stable distribution.
        param_to: The requested parametrization

    Returns:
        (loc, scale) in the requested parametrization.
    """

    # For now, we only need to shift the location. The conversion of the scale
    # will only be needed for N2/N3 parametrization.
    def _phi():
        return jnp.tan(jnp.pi * alpha / 2)

    if param_from == param_to:
        return (loc, scale)
    # All N0 -> Nx
    if param_from == Params.N0:
        if param_to == Params.N1:
            # N0 -> N1
            loc_to = jnp.where(
                alpha == 1,
                loc - beta * 2 / jnp.pi * _phi(),
                loc - beta * scale * _phi(),
            )
        return (loc_to, scale)
    # All Nx -> N0
    elif param_from == Params.N1 and param_to == Params.N0:
        # N1 -> N0
        loc_to = jnp.where(
            alpha == 1, loc + beta * 2 / jnp.pi * _phi(), loc + beta * scale * _phi()
        )
        return (loc_to, scale)
    else:
        # For other combinations (which do not exist right now),
        # convert first to N0 and then from N0 to the requested parametrization.
        loc_n0, scale_n0 = param_convert(alpha, beta, loc, scale, param_from, Params.N0)
        return param_convert(alpha, beta, loc_n0, scale_n0, Params.N0, param_to)


def shift_scale(
    alpha: INPUT,
    beta: INPUT,
    loc: INPUT,
    scale: INPUT,
    a: INPUT,
    b: INPUT,
    param: Param,
) -> Tuple[JArray | float, JArray | float, JArray]:
    """
    Given an alpha-stable distribution X with parameters alpha, beta, loc, scale,
    returns the distribution Y = a * X + b, in the same parametrization.
    Since the factor alpha is the same, it returns a tuple of (beta, loc, scale)

    Args:
        alpha: The stability parameter of the stable distribution (0-2.0].
        beta: The skewness parameter of the stable distribution.
        loc: The location parameter of the stable distribution.
        scale: The scale parameter of the stable distribution.
        a: The scaling factor.
        b: The shift factor.
        param: The parametrization of the stable distribution.

    Example:
    ```python
    import jax.numpy as jnp
    >>> shift_scale(2.0, 0.0, 0.0, 1.0, a=2.0, b=1.0, param=Params.N1) # doctest: +ELLIPSIS
    (Array(0., dtype=...), Array(1., dtype=...), Array(2., dtype=...))

    ```

    """

    beta2 = jnp.sign(a) * beta
    scale2 = jnp.abs(a) * scale
    if param == Params.N0:
        return (beta2, loc * a + b, scale2)
    elif param == Params.N1:
        # This code will not support a == 0, but then we have other degenerate distribution problems
        # anyway...
        loc2 = jnp.where(
            alpha == 1,
            a * loc + b - (2 / jnp.pi) * (a * jnp.log(jnp.abs(a))) * (beta * scale),
            a * loc + b,
        )
        return (beta2, loc2, scale2)


def sum(
    alpha: INPUT,
    beta: INPUT,
    loc: INPUT = 0.0,
    scale: INPUT = 1.0,
    param: Param = Params.N0,
    axis: Union[int, Sequence[int], None] = None,
) -> Tuple[JArray, JArray, JArray]:
    """
    Computes the sum of independent stable random distributions.

    Args:
        alpha: The stability parameter of the stable distribution (0-2.0]. It is fixed
            for all distributions.
        beta: The skewness parameter of the stable distribution.
        loc: The location parameter of the stable distribution (exact definition
            depending on the choice of pametrization)
        scale: The scale parameter of the stable distribution.
        param: The parametrization of the stable distribution.

    Returns: A tuple of (beta_sum, loc_sum, scale_sum)

    Example: the sum of two centered Gaussian distributions is a centered Gaussian distribution with
    a standard deviation of sqrt(2).
    ```python
    import jax.numpy as jnp
    >>> sum(2.0, 0.0, 0.0, jnp.asarray([1.0, 1.0]), Params.N1) # doctest: +ELLIPSIS
    (Array(0., dtype=float64), Array(0., dtype=float64), Array(1.4142135...,...))

    ```

    Example: dot product.
    Say that x1 ~ S(2, 0, 1, 1) and x2 ~ S(2, 0, 1.1, 1) are independent.
    We want to compute the distribution z = A . x + b where
    x = [x1, x2], A is a 3 x 2 matrix, b is a 3-vector, and . is the matrix-vector product.

    This can be written by a summing a scale operation:

    ```python
    >>> locs = jnp.asarray([1.0, 1.1])
    >>> scales = 2.0
    >>> A = jnp.asarray([[1,2],[3,4],[5,6]])
    >>> b = jnp.asarray([[2,2,2]]).T
    >>> (beta1, loc1, scale1) = shift_scale(2.0, 0.0,locs, scales, A,b,"N1")
    >>> print(beta1)
    [[0. 0.]
     [0. 0.]
     [0. 0.]]
    >>> print(loc1)
    [[3.  4.2]
     [5.  6.4]
     [7.  8.6]]
    >>> print(scale1)
    [[ 2.  4.]
     [ 6.  8.]
     [10. 12.]]
    >>> sum(2.0, beta1, loc1, scale1, param="N1", axis=0) # doctest: +ELLIPSIS
    (Array([0., 0.], ...), Array([15. , 19.2], ...), Array([11.83215957, 14.96662955], ...))

    ```

    """
    # Adopting the notation of Nolan 2020.
    gamma = scale
    delta = loc
    phi = jnp.tan(jnp.pi * alpha / 2)
    gamma_alpha = jnp.sum(gamma**alpha, axis=axis)
    gamma_sum = gamma_alpha ** (1.0 / alpha)
    # TODO: this is not robust to gamma == 0
    glg = gamma * jnp.log(gamma)
    beta_sum = jnp.sum(beta * gamma**alpha, axis=axis) / gamma_alpha
    # Proposition 1.3 in Nolan 2020.
    if param == Params.N0:
        s = jnp.sum(delta, axis=axis)
        s1 = jnp.sum(beta * gamma, axis=axis)
        s1_a1 = jnp.sum(beta * glg, axis=axis)
        delta_sum = s + phi * (beta_sum * gamma - s1)
        delta_sum_a1 = s + 2.0 / jnp.pi * (
            beta_sum * gamma_sum * jnp.log(gamma_sum) - s1_a1
        )
        delta_sum = jnp.where(alpha == 1, delta_sum_a1, delta_sum)
    elif param == Params.N1:
        delta_sum = jnp.sum(delta, axis=axis)
    return (beta_sum, delta_sum, gamma_sum)
