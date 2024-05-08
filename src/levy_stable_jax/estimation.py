"""
Different algorithms to estimate the parameters of a stable distribution.

These algorithms should be all considered as experimental and not part 
of the standard API: their interface is not likely to change but 
they may not be as numerically stable.
"""

from typing import Optional

import jax.numpy as jnp
from jax import Array as JArray
import jaxopt  # type: ignore
import scipy.stats._levy_stable  # type: ignore

from .distribution import logpdf, Params, Param
from ._utils import param_convert


def fit_quantiles(samples: JArray, param: Param) -> JArray:
    """
    A rough approximation of the distribution parameters based on quantiles.

    For now, it is just wrapping the scipy code.

    Examples:

    ```py
    >>> samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 30.0])
    >>> fit_quantiles(samples, Params.N0)  # doctest: +ELLIPSIS
    Array([0.76..., 0.89..., 3.07..., 0.64...], dtype=float64)

    ```

    The first value is alpha, then beta, then the location parameter and then
     the scale parameter.

    """
    # TODO: it would be nice to have pure jax code, but the scipy code
    # depends on the scipy.interpolate module, which is not going to be
    # implemented in jax.
    assert samples.ndim == 1, samples
    assert samples.size > 4, samples.size
    if param == Params.N1:
        res = scipy.stats._levy_stable._fitstart_S1(samples)
    else:
        res = scipy.stats._levy_stable._fitstart_S0(samples)
    return jnp.array(res)


def fit_ll(
    samples: JArray,
    param: Param,
    weights: Optional[JArray] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
):
    """
    Maximum likelihood evaluation for univariate Lévy-Stable distributions.

    Args:
        param: the parametrization of the returned distribution
        samples: a 1-d array with the observed samples.
        weights: optional, the weights on each of the samples.
        alpha: optional, the value of alpha to be used in the optimization.
        beta: optional, the value of beta to be used in the optimization.

    Examples:

    ```py
    >>> samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> fit_quantiles(samples, Params.N0) # doctest: +ELLIPSIS
    Array([2.        , 0.        , 3.        , 1.048...], dtype=float64)

    ```

    The first value is alpha, then beta, then the location parameter and then
     the scale parameter.
    """
    assert samples.ndim == 1, samples
    assert samples.size > 4, samples.size

    def _unpack(theta):
        idx = 0
        if alpha is None:
            alpha_ = theta[idx]
            idx += 1
        else:
            alpha_ = alpha
        if beta is None:
            beta_ = theta[idx]
            idx += 1
        else:
            beta_ = beta
        loc_ = theta[idx]
        scale_ = theta[idx + 1]
        return (alpha_, beta_, loc_, scale_)

    def _pack(alpha_, beta_, loc_, scale_):
        vals = [
            [alpha_] if alpha is None else [],
            [beta_] if beta is None else [],
            [loc_],
            [scale_],
        ]
        return jnp.array([x for val in vals for x in val])

    def _log_likelikhood_n0(theta):
        (alpha_, beta_, loc_, scale_) = _unpack(theta)
        logs = logpdf(
            samples, alpha=alpha_, beta=beta_, loc=loc_, scale=scale_, param=Params.N0
        )
        if weights is None:
            res = -jnp.mean(logs)
        else:
            res = -jnp.mean(logs * weights)
        return res

    solver = jaxopt.ScipyBoundedMinimize(
        fun=_log_likelikhood_n0, method="l-bfgs-b", tol=1e-6
    )
    # Get a crude start
    # TODO: this is not using the weights
    # All the work is done in the N0 parametrization.
    start = fit_quantiles(samples, Params.N0)
    theta_start = _pack(*start)

    # Build the bounds for the solver:
    def _bounds():
        lower = []
        upper = []
        idx = 0
        if alpha is None:
            lower.append(1.1)
            upper.append(2.0)
            idx += 1
        if beta is None:
            lower.append(-1.0)
            upper.append(1.0)
            idx += 1
        # Loc
        lower.append(-1e4)
        upper.append(1e4)
        # Scale
        lower.append(1e-3)
        upper.append(1e2)
        return (jnp.array(lower), jnp.array(upper))

    theta_bounds = _bounds()
    sres = solver.run(theta_start, bounds=theta_bounds)
    (alpha0, beta0, loc0, scale0) = _unpack(sres.params)
    # Convert to the requested parametrization
    (loc_x, scale_x) = param_convert(alpha0, beta0, loc0, scale0, Params.N0, param)
    return jnp.array([alpha0, beta0, loc_x, scale_x])
