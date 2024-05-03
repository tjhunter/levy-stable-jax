# mypy: ignore-errors
# TODO: fix the estimation module
"""
Different algorithms to estimate the parameters of a stable distribution.

These algorithms should be all considered as experimental and not part 
of the standard API: their interface is not likely to change but 
they may not be as numerically stable.
"""

import logging
from typing import Tuple, Optional

import jax.numpy as jnp
from jax import Array as JArray
from scipy.stats import levy_stable as sp_levy_stable  # type: ignore
import jaxopt  # type: ignore
import scipy.stats._levy_stable

from .distribution import logpdf, Params, Param

_logger = logging.getLogger(__name__)


def from_percentiles(
    dis: Distribution,
    lower_quant: JArray,
    lower_val: JArray,
    upper_quant: JArray,
    upper_val: JArray,
    param: Param,
) -> Tuple[JArray, JArray]:
    """
    Returns the loc and scale that match the given quantiles.

    This is a relatively crude estimate.
    """
    # assert lower_val < upper_val
    # assert lower_quant < upper_quant
    assert param == Params.N1
    low_q_vals = sp_levy_stable.ppf(lower_quant, alpha=dis.alpha, beta=dis.beta)
    up_q_vals = sp_levy_stable.ppf(upper_quant, alpha=dis.alpha, beta=dis.beta)
    scales = (upper_val - lower_val) / (up_q_vals - low_q_vals + 1e-8)
    scales = jnp.nan_to_num(scales, nan=0.0)  # Deal with dirac distributions.
    locs = lower_val - low_q_vals * scales
    return (locs, scales)


def ecdf(samples: JArray) -> Tuple[JArray, JArray, JArray]:
    """
    The empirical cumulative distribution function.

    Returns (x, cdf, counts):
    - x: the sorted unique values of the samples (deduplicated)
    - cdf: the empirical cumulative distribution function at each point
    - counts: the number of times each value appears in the samples.

    TODO: move this to jax.
    """
    assert samples.ndim == 1, samples
    assert samples.size > 4, samples.size
    # This is based on the scipy implementation:
    # https://github.com/scipy/scipy/blob/main/scipy/stats/_survival.py#L255
    sample = jnp.sort(samples)
    x, counts = jnp.unique(samples, return_counts=True)
    events = jnp.cumsum(counts)
    n = sample.size
    cdf = events / n
    return (x, cdf, counts)


def best_quantiles(
    dis: Distribution, samples: JArray, param: Param
) -> Tuple[JArray, JArray]:
    """
    Simple approximation based on quantiles.

    Returns the loc and scale that match the given quantiles.
    This is meant to be a relatively simple and robust estimate used as a
    starting initialization point for more complex algorithms, or if the
    precision is not too important.
    """
    assert param == Params.N1, param
    assert samples.ndim == 1, samples
    assert samples.size > 4, samples.size
    (vals, cdf, _) = ecdf(samples)
    cutoff_min = 0.2
    cutoff_max = 0.8
    quantile_low = jnp.min(cdf[cdf >= cutoff_min])
    x_low = jnp.min(vals[cdf >= cutoff_min])
    quantile_hi = jnp.max(cdf[cdf <= cutoff_max])
    x_hi = jnp.max(vals[cdf <= cutoff_max])
    return from_percentiles(dis, quantile_low, x_low, quantile_hi, x_hi, param)


def max_ll(
    dis: Distribution, samples: JArray, param: Param, weights: Optional[JArray] = None
):
    """
    Maximum likelihood evaluation for univariat Lévy-Stable distributions.

    Currently tuned for positive distributions that do not have large deviations.

    d: a distribution object
    samples: a 1-d array with the observed samples.
    weights: optional, the weights on each of the samples.
    """
    assert param == Params.N1, param
    assert samples.ndim == 1, samples
    assert samples.size > 4, samples.size

    def _log_likelikhood(theta):
        loc = theta[0]
        scale = theta[1]
        if weights is None:
            res = -jnp.mean(dis.logpdf(samples, loc=loc, scale=scale))
        else:
            res = -jnp.mean(dis.logpdf(samples, loc=loc, scale=scale) * weights)
        return res

    solver = jaxopt.ScipyBoundedMinimize(fun=_log_likelikhood, method="l-bfgs-b")
    # Get a crude start
    (start_loc, start_scale) = best_quantiles(dis, samples, param)
    start = jnp.asarray([start_loc, start_scale])
    # The bounds are for our domain: loc >= 0, scale in [epsi, 100]
    res = solver.run(start, bounds=(jnp.asarray([0.0, 1e-3]), jnp.asarray([1e4, 1e2])))
    return res.params


def _ks_init_n1(
    dis: Distribution, param: Param, vals: JArray, cdf: JArray
) -> Tuple[JArray, JArray]:
    """
    Provides an init value based on quantiles.

    Assumes that there is at least one quantile <= 0.3 and one quantile >= 0.6
    for each distribution.
    """
    (num_dis, _) = cdf.shape
    low_q = 0.3
    low_idxs = jnp.sum(cdf <= low_q, axis=1)
    low_vals = vals[jnp.arange(num_dis), low_idxs]
    low_quants = cdf[jnp.arange(num_dis), low_idxs]

    hi_q = 0.6
    hi_idxs = jnp.sum(cdf <= hi_q, axis=1)
    hi_vals = vals[jnp.arange(num_dis), hi_idxs]
    hi_quants = cdf[jnp.arange(num_dis), hi_idxs]
    return from_percentiles(dis, low_quants, low_vals, hi_quants, hi_vals, param)


def ks(
    dis: Distribution, param: Param, vals: JArray, cdf: JArray
) -> Tuple[jaxopt.OptStep, JArray, JArray]:
    """
    Returns the loc and scale that match the given empirical cumulative distribution.

    Minimizes the distance of the distribution to the empirical distribution
    using the Kolmogorov-Smirnov distance.

    It is vectorized: if 2d arrays are passed, it will return a 1d array,
    corresponding to estimating each of the rows in the values and cdf matrices.
    """
    assert param == Params.N1
    vals_ = jnp.atleast_2d(vals)
    cdf_ = jnp.atleast_2d(cdf)
    num_dis = vals_.shape[0]
    num_samples = vals_.shape[1]

    def unpack(theta):
        return (theta[:num_dis], theta[num_dis:])

    def pack(locs, scales):
        return jnp.concatenate([locs, scales])

    def _ks_distance(theta):
        (loc, scale) = unpack(theta)
        loc_b = jnp.tile(loc[:, jnp.newaxis], (1, num_samples))
        scale_b = jnp.tile(scale[:, jnp.newaxis], (1, num_samples))
        curr_cdf = dis.cdf(vals_, param=param, loc=loc_b, scale=scale_b)
        ks_vals = jnp.max(jnp.abs(curr_cdf - cdf_), axis=1)
        opt_val = jnp.mean(ks_vals)
        # jax.debug.print(
        #     "Opt value: {ks_vals} {loc} {scale}", ks_vals=ks_vals, loc=loc, scale=scale
        # )
        return opt_val

    assert cdf.shape[0] == num_dis
    # This is crude as an estimation, but it should at least put
    # the initial point in the right space.
    # TODO: better init, but needs to adapt code for batch version.
    start_loc, start_scale = _ks_init_n1(dis, param, vals_, cdf_)
    _logger.info(f"Start point: {(start_loc, start_scale)}")
    start = jnp.concatenate([start_loc, start_scale])
    solver = jaxopt.ScipyBoundedMinimize(fun=_ks_distance, method="l-bfgs-b", tol=1e-8)
    loc_min_bounds = pack(-1e4 * jnp.ones((num_dis,)), 1e-4 * jnp.ones((num_dis,)))
    loc_max_bounds = pack(1e4 * jnp.ones((num_dis,)), 1e2 * jnp.ones((num_dis,)))
    res = solver.run(start, bounds=(loc_min_bounds, loc_max_bounds))
    (res_loc_, res_scale_) = unpack(res.params)
    res_loc = res_loc_.squeeze()
    res_scale = res_scale_.squeeze()
    # TODO: reshape the values if 1d
    return (res, res_loc, res_scale)


# def ks(
#     dis: Distribution, param: Param, vals: JArray, cdf: JArray
# ) -> Tuple[jaxopt.OptStep, JArray, JArray]:
#     """
#     Returns the loc and scale that match the given empirical cumulative distribution.

#     Minimizes the distance of the distribution to the empirical distribution
#     using the Kolmogorov-Smirnov distance.
#     """
#     assert param == Params.N1

#     def _ks_distance(theta):
#         loc = theta[0]
#         scale = theta[1]
#         curr_cdf = dis.cdf(vals, param=param, loc=loc, scale=scale)
#         opt_val = jnp.max(jnp.abs(curr_cdf - cdf))
#         # jax.debug.print(
#         #     "Opt value: {opt_val} {loc} {scale}", opt_val=opt_val, loc=loc, scale=scale
#         # )
#         return opt_val

#     # This is crude as an estimation, but it should at least put
#     # the initial point in the right space.
#     (start_loc, start_scale) = best_quantiles(dis, vals, param)
#     _logger.info(f"Start point: {(start_loc, start_scale)}")
#     start = jnp.asarray([start_loc, start_scale])
#     solver = jaxopt.ScipyBoundedMinimize(fun=_ks_distance, method="l-bfgs-b")
#     res = solver.run(start, bounds=(jnp.asarray([0.0, 1e-3]), jnp.asarray([1e4, 1e2])))
#     return (res, res.params[0], res.params[1])
