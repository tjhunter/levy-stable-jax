"""
Implementation of the stable distribution using Jax.

All the formulas are based on Nolan (2020):

Univariate stable distribution by John Nolan, 2020.

"""

from __future__ import annotations  # for the | syntax.

from math import pi as PI
from typing import Tuple, List, Sequence

import jax
import jax.numpy as jnp
from jax import Array as JArray
import jax.scipy.special as jax_special
from jax import lax
import numpy as np
import numpy.typing as npt

from ._interp import interp_linear
from ._typing import Param, Params
from ._cache import jax_read_from_cache

Shape = Sequence[int]  # TODO: unsure how to obtain from jax.

NUM_X_POINTS = 200  # 200
NUM_ALPHA_POINTS = 80  # 76
NUM_BETA_POINTS = 101  # 101

# Some bounds and parameters
ALPHA_MIN: float = 1.1
ALPHA_MAX: float = 2.0
BETA_MIN: float = -1.0  # TODO !! this is used in the interp, use elsewhere too
BETA_MAX: float = 1.0

# An epsilon value that is required in some calculations
EPSI: float = 1e-10

# The cutoff value for the tabulation of the PDF.
# Even if the PPF may indicate that higher values are required,
# Nolan (2020) shows empirically that the relative error against the power tail is < 1e-6.
# See Fig 3.14 in Nolan (2020).
# In short: for abs(x) >= TABX_X_CUTOFF, the relative error is less than 1e-6.
TAB_X_CUTOFF: float = 25

# Beyond this value, the distribution should be simply considered as a gaussian for the purpose of
# approximating the tails.
# This could be refined, but then it causes other issues:
#
ALPHA_GAUSSIAN_CUTOFF = 2.0 - 1e-7

# For beta <= BETA_EXP_CUTOFF and alpha < ALPHA_GAUSSIAN_CUTOFF, the distribution
# is considered to be in the exponential regime
BETA_EXP_CUTOFF = -1.0 + 1e-10

X_TAIL_CUTOFF = 1.0

BETA_EXP_REGIME = -(1.0 - 2e-1)
ALPHA_GAUSSIAN_REGIME = 2 - 1e-1


def _log_tail_gaussian(x: JArray) -> JArray:
    """
    Distribution in the gaussian regime (alpha = 2)
    """
    # When alpha = 2, the Levy-stable distribution is equal to a
    # Gaussian distribution with scale = 2
    return -0.25 * jnp.square(x) - jnp.log(2 * PI) / 2 - np.log(2.0) / 2


INPUT_TYPE = npt.ArrayLike | JArray


def pdf(
    x: INPUT_TYPE,
    alpha: INPUT_TYPE,
    beta: INPUT_TYPE,
    loc: INPUT_TYPE = 0.0,
    scale: INPUT_TYPE = 1.0,
    param: Param = Params.N0,
) -> JArray | float:
    """
    The probability density function for the Levy-Stable distribution.

    Parameters:

    - x: the values at which to evaluate the PDF
    - alpha: the alpha parameter of the distribution
    - beta: the beta parameter of the distribution
    - loc: the location parameter of the distribution
        (default 0.0, meaning depends on the parametrization)
    - scale: the scale parameter of the distribution
        (default 1.0, meaning depends on the parametrization)
    - param: the parametrization of the distribution

    Returns:
    - the value of the PDF at the given point(s)

    Examples:

    Evaluation of the unit Levy-stable distribution at 0.0, with alpha=1.5, beta=0.0.
    ```py
    >>> pdf(0.0, 1.5, 0.0) # doctest: +ELLIPSIS
    0.28568...
    >>> pdf(0.0, 1.5, 0.0,param=Params.N0) # doctest: +ELLIPSIS
    0.28568...

    ```
    """
    # TODO: implement the other parametrizations
    (alpha_, beta_, x_, loc_, scale_), is_scalar = _canonicalize_input(
        [alpha, beta, x, loc, scale]
    )

    assert param == Params.N0, param
    res = _pdf_n0(x_, alpha_, beta_, loc_, scale_)
    if is_scalar:
        res = res.item()
    return res


def logpdf(
    x: INPUT_TYPE,
    alpha: INPUT_TYPE,
    beta: INPUT_TYPE,
    loc: INPUT_TYPE = 0.0,
    scale: INPUT_TYPE = 1.0,
    param: Param = Params.N0,
) -> JArray | float:
    """
    The probability density function for the Levy-Stable distribution.

    Parameters:

    - x: the values at which to evaluate the PDF
    - alpha: the alpha parameter of the distribution
    - beta: the beta parameter of the distribution
    - loc: the location parameter of the distribution
        (default 0.0, meaning depends on the parametrization)
    - scale: the scale parameter of the distribution
        (default 1.0, meaning depends on the parametrization)
    - param: the parametrization of the distribution (see Params)

    Examples:

    ```py
    >>> logpdf(0.0, 1.5, 0.0, 1.0) # doctest: +ELLIPSIS
    -1.603550...

    ```

    """
    (alpha_, beta_, x_, loc_, scale_), is_scalar = _canonicalize_input(
        [alpha, beta, x, loc, scale]
    )

    # TODO: implement the other parametrizations
    assert param == Params.N0, param
    res = _logpdf_n0(x_, alpha_, beta_, loc_, scale_)
    if is_scalar:
        res = res.item()
    return res


def rvs(
    alpha: INPUT_TYPE,
    beta: INPUT_TYPE,
    prng: jax.random.PRNGKey,
    loc: INPUT_TYPE = 0.0,
    scale: INPUT_TYPE = 1.0,
    shape: Shape = (),
    param: Param = Params.N0,
) -> JArray:
    """
    Generate random samples from the Levy-Stable distribution.

    Args:
        alpha: the alpha parameter of the distribution
        beta: the beta parameter of the distribution
        prng: the pseudo-random number generator key
        loc: the location parameter of the distribution
        scale: the scale parameter of the distribution
        shape: the shape of the output array
        param: the parametrization of the distribution

    Returns:
    - the generated samples

    Examples:

    ```py
    >>> import jax
    >>> prng = jax.random.PRNGKey(1)
    >>> rvs(alpha=1.5, beta=0.0, loc=0.0, scale=1.0, param=Params.N1,
    ...  shape=(10,), prng=prng) # doctest: +ELLIPSIS
    Array([-0.750..., -0.495..., ...], ...)

    ```
    """
    # The sampling algorithm is for N1 parametrization.
    unit_n1 = _sample_unit_n1(alpha, beta, prng, shape)
    return _values_n1_to_param(alpha, beta, unit_n1, loc, scale, param)


def _values_n1_to_param(
    alpha: INPUT_TYPE,
    beta: INPUT_TYPE,
    values: INPUT_TYPE,
    loc: INPUT_TYPE,
    scale: INPUT_TYPE,
    to_param: Param,
) -> Tuple[JArray, JArray]:
    """
    Takes values in the unit N1 parametrization and shifts them to the requested parametrization.

    Valid for all alpha.
    """
    # TODO: merge with utils.scale?
    if to_param == Params.N0:
        return jnp.where(
            jnp.abs(alpha - 1) < 1e-10,
            loc + scale * values,
            loc + scale * (values - beta * jnp.tan(PI * alpha / 2)),
        )
    elif to_param == Params.N1:
        # TODO: have a separate function to define x log(x) around x == 1 with Taylor expansion
        return jnp.where(
            jnp.abs(alpha - 1) < 1e-10,
            loc + scale * (values + beta * 2 / PI * scale * jnp.log(scale)),
            loc + scale * values,
        )


def _pdf_n0(
    x: JArray, alpha: JArray, beta: JArray, loc: JArray, scale: JArray
) -> JArray:
    """
    The PDF of the stable-Levy distribution, as parametrized in Nolan's 0 notation.
    """
    return jnp.exp(_logpdf_n0(x, alpha, beta, loc, scale))


def _logpdf_n0(
    x: JArray, alpha: JArray, beta: JArray, loc: JArray, scale: JArray
) -> JArray:
    """
    The logdPDF of the stable-Levy distribution, as parametrized in Nolan's 0 notation.
    """

    # Perform a unit transform
    scale_p = _clip_pos(scale)
    return -jnp.log(scale_p) + _logpdf_unit((x - loc) / scale_p, alpha, beta)


def _log_c_alpha_jax(alpha: JArray) -> JArray:
    return lax.lgamma(alpha) + jnp.log(jnp.sin(alpha * PI / 2)) - jnp.log(PI)


def _n0_to_n1_unit(x: JArray, alpha: JArray, beta: JArray) -> JArray:
    """
    Shifts the x unit vector so that the vector x is properly parametrized for the N1 notation
    with the same alpha and beta

    TODO: better explain
    """
    return x + beta * jnp.tan(PI * alpha / 2)


def _log_tail_exp_pos_n1(x: JArray, alpha: JArray) -> JArray:
    """the exponential tail regime (when beta=-1, alpha < 2) for positive values (x > 0)

    Note this is in N1 parametrization, following Nolan (2020).
    """
    # Notation from Prop 3.1 Nolan (2020)
    c1 = (
        1
        / jnp.sqrt(2 * PI * jnp.abs(1 - alpha))
        * jnp.power(alpha / jnp.abs(jnp.cos(PI * alpha / 2)), 1 / (2 - 2 * alpha))
    )
    log_c1 = jnp.log(c1)
    c2 = jnp.abs(1 - alpha) * jnp.power(
        (jnp.power(alpha, alpha) / jnp.abs(jnp.cos(PI * alpha / 2))),
        1.0 / (1.0 - alpha),
    )
    x_checked = jnp.maximum(x, EPSI)
    return (
        log_c1
        + ((2 - alpha) / (2 * alpha - 2)) * jnp.log(x_checked)
        - c2 * jnp.power(x_checked, alpha / (alpha - 1.0))
    )


def _log_tail_power_pos(x: JArray, alpha: JArray, beta: JArray) -> JArray:
    """
    The tail in power law regime: beta > -1, alpha < 2, x > 0
    """
    return (
        jnp.log(alpha)
        + jnp.log(1 + beta)
        + _log_c_alpha_jax(alpha)
        - (alpha + 1) * jnp.log(_clip_pos(x))
    )


def _logpdf_unit(x: JArray, alpha: JArray, beta: JArray) -> JArray:
    """ """
    cutoff_min = -TAB_X_CUTOFF
    cutoff_max = TAB_X_CUTOFF

    def return_interp(x_, alpha_, beta_):
        # For all values in the interior, use interpolation.
        # For everything else, use the tail approximation function.

        # tail values and interpolation values are used the following:
        # - tail values are always used for |x| > TAB_X_CUTOFF
        # - for some special cases (beta ~ +- 1, alpha ~ 2) either the
        #    tail or the interpolation values are used.
        #    The special cases are hardcoded, the value is picked based
        #    on a heuristic.

        # TODO: fix documentation
        # Special case: x > 5 & |beta| ~ 1 should always be evaluated with
        # Special cases:
        #  - x < -5 & beta > BETA_INTERP_CUTOFF  : evaluate with power approximation
        #  - x > 5 & beta ~ -1: evaluate with tail approximation:
        #          the behaviour of the tail diverges quickly from power law
        #          to exponential law, and the interpolation code cannot deal
        #          with large fluctuations at the boundary
        # the tail approximation. scipy <= 1.12 and pylevy are known
        # in this case to return wrong values for the logpdf (and they
        # are used by the interpolation code)
        # TODO: open a bug against scipy
        cutoff_mask = (cutoff_min < x_) & (x_ < cutoff_max)
        interp_vals = _logpdf_interp_unit(
            jnp.clip(x_, cutoff_min, cutoff_max), alpha_, beta_
        )
        tail_vals = check_tail(x_, alpha_, beta_)

        # The interpolation is not reliable for extreme values of beta, and
        # it falls off too quickly.
        # Heuristic: use the maximum of tail and interpolation in the cases:
        interp_mask = (
            (alpha_ > ALPHA_GAUSSIAN_REGIME)
            & (jnp.abs(tail_vals - interp_vals) > 1e-1)
            & (interp_vals < -15)
        ) | jnp.isinf(interp_vals)

        # blend = jax_special.logsumexp(
        #     jnp.stack([tail_vals, interp_vals], axis=1),
        #     axis=1,
        # )
        blend = jnp.maximum(interp_vals, tail_vals)

        return jnp.where(
            cutoff_mask,
            jnp.where(
                interp_mask,
                # Heuristic to  blend tail and interp
                blend,
                interp_vals,
            ),
            tail_vals,
        )

    def check_tail(x_, alpha_, beta_) -> JArray:
        # This function should just apply to tail values.

        # Flip all x to positive and switch the sign of beta.
        # This uses the symmetry properties of the Levy-stable distribution.
        beta_ = jnp.where(x_ < 0, -beta_, beta_)
        x_ = jnp.maximum(
            jnp.abs(x_), X_TAIL_CUTOFF
        )  # guard on x, only consider points at tails

        return jnp.where(
            beta_ <= BETA_EXP_CUTOFF,
            _log_tail_exp_pos_n1(_n0_to_n1_unit(x_, alpha_, beta_), alpha_),
            check_tail_inner_pos(x_, alpha_, jnp.maximum(beta_, BETA_EXP_CUTOFF)),
        )

    def check_tail_inner_pos(x_, alpha_, beta_) -> JArray:
        # Builds tail in the standard case, and applies corrections if the tail is close to
        # boundaries.
        # Assumes that x > 1, beta < 1, alpha < 2

        zeros = jnp.zeros_like(x_)
        ninfs = jnp.ones_like(x_) * (-np.inf)

        pow_tail = _log_tail_power_pos(x_, alpha_, beta_)
        exp_tail = _log_tail_exp_pos_n1(_n0_to_n1_unit(x_, alpha_, beta_), alpha_)
        gau_tail = _log_tail_gaussian(x_)

        # BETA_EXP_REGIME is negative
        assert BETA_EXP_REGIME < 0

        # The base is pow_tail
        # If beta < BETA_EXP_REGIME, blend exp_tail but not gau_tail
        # If alpha > ALPHA_GAU_REGIME and beta > BETA_EXP_REGIME, blend gau_tail
        # The tricky part: the gaussian tail is concentrated around 0 and falls quickly.
        # With beta ~ -1, alpha ~ 2, it is unclear which parts are gaussian and which
        # parts are exponential
        exp_tail_mask = jnp.where(beta_ <= BETA_EXP_REGIME, zeros, ninfs)
        gau_tail_mask = jnp.where(
            (alpha_ >= ALPHA_GAUSSIAN_REGIME) & (beta_ >= BETA_EXP_REGIME), zeros, ninfs
        )

        res = jax_special.logsumexp(
            jnp.stack(
                [pow_tail, gau_tail + gau_tail_mask, exp_tail + exp_tail_mask], axis=1
            ),
            axis=1,
        )
        assert res.shape == x_.shape, res.shape
        return res

    # Gaussian case out of the way. No interpolation happens then.
    return jnp.where(
        alpha >= ALPHA_GAUSSIAN_CUTOFF,
        _log_tail_gaussian(x),
        return_interp(x, jnp.minimum(alpha, ALPHA_GAUSSIAN_CUTOFF), beta),
    )


def _clip_pos(x: JArray) -> JArray:
    return jnp.maximum(x, EPSI)


def _logpdf_interp_unit(x: JArray, alpha: JArray, beta: JArray) -> JArray:
    """
    The log-PDF of the stable-Levy distribution, as parametrized in Nolan's 0 notation.

    This function is backed by a table and only performs interpolation on a given grid.
    It will fail for values outside of the grid.
    """
    # TODO: cleanup
    # Flip all x to positive and switch the sign of beta.
    # This uses the symmetry properties of the Levy-stable distribution.
    beta_ = beta  # jnp.where(x < 0, -beta, beta)
    x_ = x  # jnp.abs(x)

    points = jnp.stack([x_, alpha, beta_], axis=1)
    assert points.shape == (len(x_), 3)
    tab_logpdf = jax_read_from_cache("logpdf")
    lower = jnp.array([-TAB_X_CUTOFF, ALPHA_MIN, BETA_MIN])
    upper = jnp.array([TAB_X_CUTOFF, ALPHA_MAX, BETA_MAX])
    res = interp_linear(points, tab_logpdf, lower, upper)
    return res


def _canonicalize_input(arrays: List[npt.ArrayLike]) -> Tuple[List[JArray], bool]:
    """
    Given a set of various sorts of inputs, returns a list of 1d arrays that all have the
     same lengths.
    """
    # TODO: use primitives in numpy to achieve the same result

    def _can_elt(t) -> JArray:
        arr = jnp.asarray(t)
        if arr.ndim == 0:
            arr = arr[jnp.newaxis]
        if arr.ndim > 1:
            raise Exception(
                f"Input should be scalar or 1d array, one array has dim {arr.ndim}"
            )
        return arr

    arrs = [_can_elt(t) for t in arrays]
    sizes = set(len(arr) for arr in arrs)
    is_scalar = all([isinstance(t, (int, float)) for t in arrays])
    if len(sizes) < 1:
        return [], is_scalar
    if len(sizes) > 2:
        raise Exception(f"Size mismatch: {[arr.shape for arr in arrs]}")
    if sizes == {1}:
        # TODO: missing the case with size-1 1D arrays
        return arrs, is_scalar
    max_len = max(sizes)

    def _check_size(arr: JArray) -> JArray:
        if arr.size == max_len:
            return arr
        if arr.size == 1:
            return jnp.tile(arr, max_len)
        raise AssertionError(f"Size mismatch: {arr.shape} vs {max_len}")

    return [_check_size(arr) for arr in arrs], False


def _sample_unit_n1(
    alpha: JArray, beta: JArray, prng: jax.random.PRNGKey, shape: Shape
) -> JArray:
    """
    Sample from the unit Levy-stable distribution in the N1 parametrization.

    This implementation only support alpha != 1, it will diverge or return NaNs for alpha == 1.

    Algorithm based on Nolan (2020), Theorem 1.3

    Based on the scipy code:
    https://github.com/scipy/scipy/blob/v1.13.0/scipy/stats/_levy_stable/__init__.py#L422
    """
    k1, k2 = jax.random.split(prng, 2)
    TH = (jax.random.uniform(k1, shape=shape) - 0.5) * PI
    W = jax.random.exponential(k2, shape=shape)
    aTH = alpha * TH
    cosTH = jnp.cos(TH)
    tanTH = jnp.tan(TH)

    # Not implementing the special cases for alpha = 1 or beta=0
    val0 = beta * np.tan(np.pi * alpha / 2)
    th0 = jnp.arctan(val0) / alpha
    val3 = W / (cosTH / jnp.tan(alpha * (th0 + TH)) + jnp.sin(TH))
    res3 = val3 * (
        (
            jnp.cos(aTH)
            + jnp.sin(aTH) * tanTH
            - val0 * (jnp.sin(aTH) - jnp.cos(aTH) * tanTH)
        )
        / W
    ) ** (1.0 / alpha)
    return res3
