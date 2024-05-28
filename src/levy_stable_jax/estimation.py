"""
Different algorithms to estimate the parameters of a stable distribution.

These algorithms should be all considered as experimental and not part 
of the standard API: their interface is not likely to change but 
they may not be as numerically stable.
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import Array as JArray, jit
from scipy.interpolate import RectBivariateSpline
import jax
import jaxopt  # type: ignore

from .distribution import logpdf, Params, Param, cdf as lsj_cdf
from ._utils import param_convert

def fit_quantiles(
    samples: JArray | None, param: Param, percentiles: JArray | None = None
) -> JArray:
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
    if percentiles is None:
        assert samples.ndim == 1, samples
        assert samples.size > 4, samples.size
        p05 = np.percentile(samples, 5)
        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)
        p25 = np.percentile(samples, 25)
        p75 = np.percentile(samples, 75)
    else:
        (p05, p25, p50, p75, p95) = percentiles

    if param == Params.N1:
        res = _fitstart_S1(p05, p25, p50, p75, p95)
    else:
        res = _fitstart_S0(p05, p25, p50, p75, p95)
    return jnp.array(res)


def fit_ks(
    x: JArray,
    cdf: JArray,
    param: Param,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
):
    """
    Fits the closest Levy-stable distribution to cumulative distribution points.
    The distance is the Kolmogorov-Smirnov distance.

    Because the Levy-stable distribution is sensitive to extreme
    values, this can provide a better matching distribution.

    Args:
        x: the points at which the CDF is evaluated.
        cdf: the values of the CDF at the given point

    The first value is alpha, then beta, then the location parameter and then
     the scale parameter.
    """

    min_x = jnp.min(x)
    max_x = jnp.max(x)
    delta = max_x - min_x
    n = jnp.size(x)

    def _metric(cdf_, target_):
        return jnp.max(jnp.abs(cdf_-target_))
        # return jnp.mean(jnp.abs(cdf_ - target_))

    def _ks_distance(alpha_, beta_, loc_, scale_):
        curr_cdf = lsj_cdf(
            x, loc=loc_, scale=scale_, alpha=alpha_, beta=beta_, param=Params.N0
        )
        low_x = jnp.linspace(min_x - delta, min_x, n)
        low_cdf = lsj_cdf(low_x, alpha=alpha_, beta=beta_, loc=loc_, scale=scale_, param=Params.N0)
        high_x = jnp.linspace(max_x, max_x + delta, n)
        high_cdf = lsj_cdf(high_x, alpha=alpha_, beta=beta_, loc=loc_, scale=scale_, param=Params.N0)
        # Triple the size of the interval so that the values before and after
        # the segment are also evaluated.
        # It is assumed that these values are meant to be tail values
        opt_val = _metric(curr_cdf, cdf) + 1.0 * _metric(low_cdf, 0) + 1.0 * _metric(high_cdf, 1)
        # jax.debug.print(
        #     "Opt value: {opt_val} {alpha_} {beta_} {loc_} {scale_}",
        #     opt_val=opt_val,
        #     alpha_=alpha_,
        #     beta_=beta_,
        #     loc_=loc_,
        #     scale_=scale_,
        # )
        return opt_val

    # Get a crude start
    # Reuse the existing samples to get a coarse approximation of the parameters.
    def _quantile(q):
        idx = min(np.sum(cdf<=q), len(x) - 1)
        return x[idx]
    quants = [_quantile(q) for q in [0.05, 0.25, 0.5, 0.75, 0.95]]
    # All the work is done in the N0 parametrization.
    start = fit_quantiles(None, Params.N0, percentiles=quants)
    print("start", start)
    res = _fit_objective_n0(alpha, beta, _ks_distance, param, start)
    print("res", res)
    return res


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
    >>> fit_ll(samples, Params.N1) # doctest: +ELLIPSIS
    Array([2.        , 0.        , 3.        , 1.048...], dtype=float64)

    ```

    The first value is alpha, then beta, then the location parameter and then
     the scale parameter.
    """
    assert samples.ndim == 1, samples
    assert samples.size > 4, samples.size

    def _log_likelikhood_n0(alpha_, beta_, loc_, scale_):
        # (alpha_, beta_, loc_, scale_) = _unpack(theta)
        logs = logpdf(
            samples, alpha=alpha_, beta=beta_, loc=loc_, scale=scale_, param=Params.N0
        )
        if weights is None:
            res = -jnp.mean(logs)
        else:
            res = -jnp.mean(logs * weights)
        jax.debug.print(
            "Opt value: {res} {alpha_} {beta_} {loc_} {scale_}",
            res=res,
            alpha_=alpha_,
            beta_=beta_,
            loc_=loc_,
            scale_=scale_,
        )
        return res

    # Get a crude start
    # TODO: this is not using the weights
    # All the work is done in the N0 parametrization.
    start = fit_quantiles(samples, Params.N0)
    print("start", start)
    return _fit_objective_n0(alpha, beta, _log_likelikhood_n0, param, start)


def _fit_objective_n0(alpha_cons, beta_cons, obj, param, start):
    """
    Minimizes an objective, using the N0 param.
    Returns the requested param.

    start must be in the N0 param.
    """

    def _unpack(theta):
        idx = 0
        if alpha_cons is None:
            alpha_ = theta[idx]
            idx += 1
        else:
            alpha_ = alpha_cons
        if beta_cons is None:
            beta_ = theta[idx]
            idx += 1
        else:
            beta_ = beta_cons
        loc_ = theta[idx]
        scale_ = theta[idx + 1]
        return (alpha_, beta_, loc_, scale_)

    def _pack(alpha_, beta_, loc_, scale_):
        vals = [
            [alpha_] if alpha_cons is None else [],
            [beta_] if beta_cons is None else [],
            [loc_],
            [scale_],
        ]
        return jnp.array([x for val in vals for x in val])

    def obj_fun(theta):
        (alpha_, beta_, loc_, scale_) = _unpack(theta)
        return obj(alpha_, beta_, loc_, scale_)

    solver = jaxopt.ScipyBoundedMinimize(
        fun=obj_fun, method="l-bfgs-b", tol=1e-8, jit=False
    )
    theta_start = _pack(*start)

    # Build the bounds for the solver:
    def _bounds():
        lower = []
        upper = []
        idx = 0
        if alpha_cons is None:
            lower.append(1.1)
            upper.append(2.0)
            idx += 1
        if beta_cons is None:
            # TODO: explain. this is important for numerical stability for the time being.
            lower.append(-0.9)
            upper.append(0.9)
            idx += 1
        # Loc
        lower.append(-1e10)
        upper.append(1e10)
        # Scale
        lower.append(1e-15)
        upper.append(1e15)
        return (jnp.array(lower), jnp.array(upper))

    theta_bounds = _bounds()
    sres = solver.run(theta_start, bounds=theta_bounds)
    (alpha0, beta0, loc0, scale0) = _unpack(sres.params)
    # TODO: check the number of steps. When beta=+-1, the surface to optimize goes
    # very rapidly to infinite values, which is something the optimizer will struggle
    # with.
    # Convert to the requested parametrization
    (loc_x, scale_x) = param_convert(alpha0, beta0, loc0, scale0, Params.N0, param)
    return jnp.array([alpha0, beta0, loc_x, scale_x])


def _fitstart_S0(p05, p25, p50, p75, p95):
    alpha, beta, delta1, gamma = _fitstart_S1(p05, p25, p50, p75, p95)

    # Formulas for mapping parameters in S1 parameterization to
    # those in S0 parameterization can be found in [NO]. Note that
    # only delta changes.
    if alpha != 1:
        delta0 = delta1 + beta * gamma * np.tan(np.pi * alpha / 2.0)
    else:
        delta0 = delta1 + 2 * beta * gamma * np.log(gamma) / np.pi

    return alpha, beta, delta0, gamma


def _fitstart_S1(p05, p25, p50, p75, p95):
    # We follow McCullock 1986 method - Simple Consistent Estimators
    # of Stable Distribution Parameters

    # fmt: off
    # Table III and IV
    nu_alpha_range = [2.439, 2.5, 2.6, 2.7, 2.8, 3, 3.2, 3.5, 4,
                      5, 6, 8, 10, 15, 25]
    nu_beta_range = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]

    # table III - alpha = psi_1(nu_alpha, nu_beta)
    alpha_table = np.array([
        [2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
        [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
        [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
        [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
        [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
        [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
        [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
        [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
        [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
        [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
        [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
        [0.896, 0.892, 0.884, 0.883, 0.855, 0.823, 0.769],
        [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
        [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.597],
        [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]]).T
    # transpose because interpolation with `RectBivariateSpline` is with
    # `nu_beta` as `x` and `nu_alpha` as `y`

    # table IV - beta = psi_2(nu_alpha, nu_beta)
    beta_table = np.array([
        [0, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
        [0, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
        [0, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
        [0, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
        [0, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
        [0, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
        [0, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
        [0, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
        [0, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
        [0, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
        [0, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
        [0, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
        [0, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
        [0, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
        [0, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]]).T

    # Table V and VII
    # These are ordered with decreasing `alpha_range`; so we will need to
    # reverse them as required by RectBivariateSpline.
    alpha_range = [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1,
                   1, 0.9, 0.8, 0.7, 0.6, 0.5][::-1]
    beta_range = [0, 0.25, 0.5, 0.75, 1]

    # Table V - nu_c = psi_3(alpha, beta)
    nu_c_table = np.array([
        [1.908, 1.908, 1.908, 1.908, 1.908],
        [1.914, 1.915, 1.916, 1.918, 1.921],
        [1.921, 1.922, 1.927, 1.936, 1.947],
        [1.927, 1.930, 1.943, 1.961, 1.987],
        [1.933, 1.940, 1.962, 1.997, 2.043],
        [1.939, 1.952, 1.988, 2.045, 2.116],
        [1.946, 1.967, 2.022, 2.106, 2.211],
        [1.955, 1.984, 2.067, 2.188, 2.333],
        [1.965, 2.007, 2.125, 2.294, 2.491],
        [1.980, 2.040, 2.205, 2.435, 2.696],
        [2.000, 2.085, 2.311, 2.624, 2.973],
        [2.040, 2.149, 2.461, 2.886, 3.356],
        [2.098, 2.244, 2.676, 3.265, 3.912],
        [2.189, 2.392, 3.004, 3.844, 4.775],
        [2.337, 2.634, 3.542, 4.808, 6.247],
        [2.588, 3.073, 4.534, 6.636, 9.144]])[::-1].T
    # transpose because interpolation with `RectBivariateSpline` is with
    # `beta` as `x` and `alpha` as `y`

    # Table VII - nu_zeta = psi_5(alpha, beta)
    nu_zeta_table = np.array([
        [0, 0.000, 0.000, 0.000, 0.000],
        [0, -0.017, -0.032, -0.049, -0.064],
        [0, -0.030, -0.061, -0.092, -0.123],
        [0, -0.043, -0.088, -0.132, -0.179],
        [0, -0.056, -0.111, -0.170, -0.232],
        [0, -0.066, -0.134, -0.206, -0.283],
        [0, -0.075, -0.154, -0.241, -0.335],
        [0, -0.084, -0.173, -0.276, -0.390],
        [0, -0.090, -0.192, -0.310, -0.447],
        [0, -0.095, -0.208, -0.346, -0.508],
        [0, -0.098, -0.223, -0.380, -0.576],
        [0, -0.099, -0.237, -0.424, -0.652],
        [0, -0.096, -0.250, -0.469, -0.742],
        [0, -0.089, -0.262, -0.520, -0.853],
        [0, -0.078, -0.272, -0.581, -0.997],
        [0, -0.061, -0.279, -0.659, -1.198]])[::-1].T
    # fmt: on

    psi_1 = RectBivariateSpline(
        nu_beta_range, nu_alpha_range, alpha_table, kx=1, ky=1, s=0
    )

    def psi_1_1(nu_beta, nu_alpha):
        return psi_1(nu_beta, nu_alpha) if nu_beta > 0 else psi_1(-nu_beta, nu_alpha)

    psi_2 = RectBivariateSpline(
        nu_beta_range, nu_alpha_range, beta_table, kx=1, ky=1, s=0
    )

    def psi_2_1(nu_beta, nu_alpha):
        return psi_2(nu_beta, nu_alpha) if nu_beta > 0 else -psi_2(-nu_beta, nu_alpha)

    phi_3 = RectBivariateSpline(beta_range, alpha_range, nu_c_table, kx=1, ky=1, s=0)

    def phi_3_1(beta, alpha):
        return phi_3(beta, alpha) if beta > 0 else phi_3(-beta, alpha)

    phi_5 = RectBivariateSpline(beta_range, alpha_range, nu_zeta_table, kx=1, ky=1, s=0)

    def phi_5_1(beta, alpha):
        return phi_5(beta, alpha) if beta > 0 else -phi_5(-beta, alpha)

    nu_alpha = (p95 - p05) / (p75 - p25)
    nu_beta = (p95 + p05 - 2 * p50) / (p95 - p05)

    if nu_alpha >= 2.439:
        eps = np.finfo(float).eps
        alpha = np.clip(psi_1_1(nu_beta, nu_alpha)[0, 0], eps, 2.0)
        beta = np.clip(psi_2_1(nu_beta, nu_alpha)[0, 0], -1.0, 1.0)
    else:
        alpha = 2.0
        beta = np.sign(nu_beta)
    c = (p75 - p25) / phi_3_1(beta, alpha)[0, 0]
    zeta = p50 + c * phi_5_1(beta, alpha)[0, 0]
    delta = zeta - beta * c * np.tan(np.pi * alpha / 2.0) if alpha != 1.0 else zeta

    return (alpha, beta, delta, c)
