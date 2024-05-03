# mypy: ignore-errors
from numpy import log
from scipy.special import loggamma  # type: ignore
from scipy.stats import levy_stable as sp_levy_stable  # type: ignore
import logging

from numpy import ndarray as Array
import jax.scipy as jsp

_logger = logging.getLogger(__name__)


# class Distribution:
#     """
#     A stable-Lévy distribution in which the alpha and beta values are frozen.

#     This distribution has fixed alpha and beta parameters,
#     and will take gradient values with respect to likelihood values.
#     """

#     alpha: float
#     beta: float

#     def __init__(self, alpha: float, beta: float) -> None:
#         self.alpha = alpha
#         self.beta = beta

#     def cdf(
#         self,
#         x: JArray,
#         param: Param = Params.N1,
#         loc: Optional[JArray] = None,
#         scale: Optional[JArray] = None,
#     ) -> JArray:
#         """
#         The CDF of the distribution.

#         param: the type of parametrization chosen.
#         loc: the location parameter, corresponding to delta in Nolan (2020)
#         scale: the scale parameter, corresponding to gamma in Nolan (2020)

#         """
#         assert param == Params.N1, "No other parametrization allowed for now"
#         gamma: JArray = scale if scale is not None else jnp.array([1.0])
#         delta: JArray = loc if loc is not None else jnp.array([0.0])
#         # Prevent the gradient search from coming too close to zero and creating infinite values
#         gamma_pos = jnp.maximum(gamma, EPSI)
#         return self._cdf_unit((x - delta) / gamma_pos)

#     # def ppf(
#     #     self,
#     #     quantiles: JArray,
#     #     param: Param = Params.N1,
#     #     loc: Optional[JArray] = None,
#     #     scale: Optional[JArray] = None,
#     # ) -> JArray:
#     #     """
#     #     The CDF of the distribution.

#     #     param: the type of parametrization chosen.
#     #     loc: the location parameter, corresponding to delta in Nolan (2020)
#     #     scale: the scale parameter, corresponding to gamma in Nolan (2020)

#     #     """
#     #     assert False, "TODO: implement ppf"

#     def logpdf(
#         self,
#         x: JArray,
#         param: Param = Params.N1,
#         loc: Optional[JArray] = None,
#         scale: Optional[JArray] = None,
#     ) -> JArray:
#         """
#         The PDF in log scale.

#         param: the type of parametrization chosen.
#         loc: the location parameter, corresponding to delta in Nolan (2020)
#         scale: the scale parameter, corresponding to gamma in Nolan (2020)
#         """
#         assert param == Params.N1, "No other parametrization allowed for now"
#         gamma: JArray = scale if scale is not None else jnp.array([1.0])
#         delta: JArray = loc if loc is not None else jnp.array([0.0])
#         # Prevent the gradient search from coming too close to zero and creating infinite values
#         gamma_pos = jnp.maximum(gamma, EPSI)
#         return -jnp.log(gamma_pos) + self._logpdf_unit((x - delta) / gamma_pos)

#     def pdf(
#         self,
#         x: JArray,
#         param: Param = Params.N1,
#         loc: Optional[JArray] = None,
#         scale: Optional[JArray] = None,
#     ) -> JArray:
#         """
#         The PDF in log scale.

#         param: the type of parametrization chosen.
#         loc: the location parameter, corresponding to delta in Nolan (2020)
#         scale: the scale parameter, corresponding to gamma in Nolan (2020)
#         """
#         return jnp.exp(self.logpdf(x, param, loc, scale))

#     @cached_property
#     def _tabulation(self) -> Tuple[float, float, Array, Array]:
#         _logger.info("Creating tabulation")
#         res = _generate_tab_logpdf(self.alpha, self.beta, ppf_cutoff=TAB_CUTOFF)
#         _logger.info("Tabulation created")
#         return res

#     def _logpdf_unit(self, x: JArray) -> JArray:
#         (tab_lower, tab_upper, tab_x, tab_logpdf) = self._tabulation
#         # There is a small kink at the cutoff value because the
#         # heavy tail function is only valid in the limit.
#         # This code performs a small blending between the tabulation
#         # and the tail functions, if necessary.
#         # TODO: this is introducing artifacts.
#         rs = jnp.clip((x - tab_lower) / (tab_upper - tab_lower), 0, 1)
#         rate = 10
#         low_mask = jnp.maximum(0.0, 1 - rate * rs)
#         up_mask = jnp.maximum(0.0, rate * rs - (rate - 1))
#         full_mask = jnp.maximum(low_mask, up_mask)
#         pos = _log_tail_pos(self.alpha, self.beta, x)
#         neg = _log_tail_neg(self.alpha, self.beta, x)
#         print(_interp_catmul_1d(x, tab_x, tab_logpdf).shape, "catmul")
#         print("low_mask", low_mask.shape)
#         print("up_mask", up_mask.shape)
#         print("full_mask", full_mask.shape)
#         return jnp.where(
#             x >= tab_upper,
#             pos,
#             jnp.where(
#                 x <= tab_lower,
#                 neg,
#                 _interp_catmul_1d(x, tab_x, tab_logpdf),
#                 #   * (1 - full_mask)
#                 # + low_mask * neg
#                 # + up_mask * pos,
#             ),
#         )

#     @cached_property
#     def _tabulation_cdf(self) -> Tuple[float, float, Array, Array]:
#         _logger.info("Creating tabulation for CDF")
#         qs = np.linspace(0.0, 1.0, TAB_CDF_NUM_POINTS)
#         xs = sp_levy_stable.ppf(qs, alpha=self.alpha, beta=self.beta)
#         # TODO: truncate the tail for the time being. It is not so much of a problem as for the PDF.
#         xs[0] = xs[1] - 1.0
#         xs[-1] = xs[-2] + 1.0
#         _logger.info("Tabulation for CDF created")
#         # The cutoff values are strictly within the point bounds.
#         return (xs[1], xs[-2], xs, qs)

#     def _cdf_unit(self, x: JArray) -> JArray:
#         (min_cutoff, max_cutoff, xs, qs) = self._tabulation_cdf
#         # TODO: results are clipped to zero for the time being, use a proper tail
#         #
#         return jnp.where(
#             x >= max_cutoff,
#             jnp.ones_like(x),
#             jnp.where(
#                 x <= min_cutoff,
#                 jnp.zeros_like(x),
#                 jnp.interp(
#                     x,
#                     jnp.asarray(xs),
#                     jnp.asarray(qs),
#                 ),
#             ),
#         )


# def levy_stable(alpha: float, beta: float) -> Distribution:
#     """
#     Creates a stable distribution with the given alpha and beta parameters.

#     The alpha value is currently restricted to the range of (1.1, 2.0).
#     This is a range of value that is most interesting for physical emissions processes,
#     and keeps the mean finite.

#     beta is currently restricted to the range of (0.0, 1.0).
#     This the most interesting range for physical emissions processes (bias
#     towards positive values).
#     """
#     alpha = float(alpha)
#     beta = float(beta)
#     assert ALPHA_MIN <= alpha <= ALPHA_MAX
#     assert BETA_MIN <= beta <= BETA_MAX
#     return Distribution(alpha, beta)


def _log_tail_pareto_neg_n1(alpha: float, beta: float, x: JArray) -> JArray:
    """
    The log of the pareto tail of the stable distribution.

    This assumes the N1 parametrization. It internally converts to the N0 parametrization.
    """
    # Use the symmetry of the stable distribution.
    return _log_tail_pareto_pos_n1(alpha, -beta, -x)


def _log_tail_pos_standard(alpha: float, beta: float, x: JArray) -> JArray:
    """The positive tail regime outside of boundaries"""
    return (
        log(alpha)
        + log(1 + beta)
        + _log_c_alpha(alpha)
        - (alpha + 1) * jnp.log(jnp.maximum(x, EPSI))
    )


def _log_tail_neg(alpha: float, beta: float, x: JArray) -> JArray:
    """
    The log of the tail of the stable distribution (negative side)
    """
    # Using symmetry
    return _log_tail_pos(alpha, -beta, -x)


def _log_tail_pos_old(alpha: float, beta: float, x: JArray) -> JArray:
    assert beta > -1, beta
    # The max(EPSI) is here to prevent some underflows in the gradent.
    if alpha >= ALPHA_GAUSSIAN_CUTOFF:
        return -0.25 * jnp.square(x)
    return (
        log(alpha)
        + log(1 + beta)
        + _log_c_alpha(alpha)
        - (alpha + 1) * jnp.log(jnp.maximum(x, EPSI))
    )


def _log_tail_neg_old(alpha: float, beta: float, x: JArray) -> JArray:
    if alpha >= ALPHA_GAUSSIAN_CUTOFF:
        return -0.25 * jnp.square(x)
    if beta < BETA_1_CUTOFF:
        return (
            np.log(alpha)
            + np.log(1 - beta)
            + _log_c_alpha(alpha)
            - (alpha + 1) * jnp.log(jnp.maximum(-x, EPSI))
        )
    else:
        # Notation from Prop 3.1 Nolan (2020)
        c1 = (
            1
            / np.sqrt(2 * PI * abs(1 - alpha))
            * np.power(alpha / np.abs(np.cos(PI * alpha / 2)), 1 / (2 - 2 * alpha))
        )
        log_c1 = np.log(c1)
        c2 = np.abs(1 - alpha) * np.power(
            (np.power(alpha, alpha) / np.abs(np.cos(PI * alpha / 2))),
            1.0 / (1.0 - alpha),
        )
        x_checked = jnp.maximum(-x, EPSI)
        return (
            log_c1
            + ((2 - alpha) / (2 * alpha - 2)) * jnp.log(x_checked)
            - c2 * jnp.power(x_checked, alpha / (alpha - 1.0))
        )


def _generate_tab_logpdf(
    alpha: float, beta: float, ppf_cutoff: float
) -> Tuple[float, float, Array, Array]:
    tab_lower = sp_levy_stable.ppf(ppf_cutoff, alpha, beta)
    # Two cases:
    # - we are close to the gaussian regime -> disregard the heavy tail
    # - We are in a very heavy tail regime, the PPF is not meaningful
    #   and the heavy tail regime starts much earlier
    if tab_lower < -TAB_X_CUTOFF or alpha >= ALPHA_GAUSSIAN_CUTOFF:
        tab_lower = -TAB_X_CUTOFF
    tab_upper = sp_levy_stable.ppf(1 - ppf_cutoff, alpha, beta)
    if tab_upper > TAB_X_CUTOFF or alpha >= ALPHA_GAUSSIAN_CUTOFF:
        tab_upper = TAB_X_CUTOFF
    _logger.info("tab_lower: %s tab_upper: %s", tab_lower, tab_upper)
    tab_x = np.linspace(tab_lower, tab_upper, TAB_NUM_POINTS)
    rv = sp_levy_stable(alpha, beta)
    tab_logpdf = rv.logpdf(tab_x)
    return (tab_lower, tab_upper, tab_x, tab_logpdf)


def _log_tail_pos(alpha: float, beta: float, x: JArray) -> JArray:
    """
    The log of the tail of the stable distribution (positive side)
    """
    # This blends the tails together in different regimes.
    # This code is here to ensure that the tails are correct, but
    # while respecting most of the distribution's probability mass.
    # For example, in the case alpha=1.99, the tail is still a power tail,
    # but the distribution is mostly gaussian until x=10.
    # This is handled by taking the potential heavy tail and
    # blending it through a softmax function.

    # The cases to consider are:
    # - pure pareto regime: alpha = (1.1, 1.9) and beta = (-0.9, 0.9)
    # - gaussian regime: alpha = (1.9, 2.0) and beta = (-0.9, 0.9)
    # - positive heavy tail regime: alpha = (1.1, 1.9) and beta = (0.9, 1.0)
    # - negative heavy tail regime: alpha = (1.1, 1.9) and beta = (-1.0, -0.9)
    alpha_2_cut = 1.8
    beta_1n_cut = -0.8

    def softmax(y1, y2):
        return jsp.special.logsumexp(a=jnp.stack([y1, y2]), axis=0)

    # Special cases in which there is no heavy tail:
    if alpha >= 2.0 - EPSI:
        return _log_gaussian(x)
    if beta <= -1.0 + EPSI:
        # Minus sign for the positive side by reflection.
        return _log_tail_beta1_neg(alpha, -x)

    # The heavy tail regime that may require blending.
    heavy_tail = _log_tail_pareto_pos_n1(alpha, beta, x)
    special_tail = None
    if alpha >= alpha_2_cut:
        # It does not matter too much for the value of beta, as it converges
        # towards the gaussian regime.
        special_tail = _log_gaussian(x)
    elif beta <= beta_1n_cut:
        # Minus sign for the positive side by reflection.
        special_tail = _log_tail_beta1_neg(alpha, -x)

    if special_tail is not None:
        return softmax(heavy_tail, special_tail)
    else:
        return heavy_tail


def _log_tail_beta1_neg(alpha: float, x: JArray) -> JArray:
    """
    The log of the beta=1 tail of the stable distribution.
    (negative side)
    """
    # Notation from Prop 3.1 Nolan (2020)
    c1 = (
        1
        / np.sqrt(2 * PI * abs(1 - alpha))
        * np.power(alpha / np.abs(np.cos(PI * alpha / 2)), 1 / (2 - 2 * alpha))
    )
    log_c1 = np.log(c1)
    c2 = np.abs(1 - alpha) * np.power(
        (np.power(alpha, alpha) / np.abs(np.cos(PI * alpha / 2))),
        1.0 / (1.0 - alpha),
    )
    x_checked = jnp.maximum(-x, EPSI)
    return (
        log_c1
        + ((2 - alpha) / (2 * alpha - 2)) * jnp.log(x_checked)
        - c2 * jnp.power(x_checked, alpha / (alpha - 1.0))
    )


def _log_gaussian(x: JArray) -> JArray:
    """
    The gaussian regime of the stable distribution.
    """
    return -0.5 * jnp.log(2 * np.pi) - 0.25 * jnp.square(x)


def _log_tail_pareto_pos_n1(alpha: float, beta: float, x: JArray) -> JArray:
    """
    The log of the pareto tail of the stable distribution.

    This assumes the N1 parametrization. It internally converts to the N0 parametrization.
    """
    u = beta * np.tan(np.pi * alpha / 2.0)
    # Guard against gradient issues
    x0 = jnp.maximum(x - u, EPSI)
    # See (3.43) in Nolan (2020)
    return log(alpha) + log(1 + beta) + _log_c_alpha(alpha) - (alpha + 1) * jnp.log(x0)


def _log_c_alpha(alpha: float) -> float:
    """
    The log of the c_alpha constant.
    """
    return loggamma(alpha) + log(np.sin(alpha * PI / 2)) - np.log(PI)


def _pdf_interp_unit(x, alpha, beta):
    """
    The PDF of the stable-Levy distribution, as parametrized in Nolan's 0 notation.

    This function is backed by a table and only performs interpolation on a given grid.
    It will fail for values outside of the grid.
    """
    # TODO: parameters checks.
    if _USE_SCIPY_DATA_SOURCE:
        return jnp.exp(_logpdf_interp_unit(x, alpha, beta))
    points = jnp.stack([jnp.arctan(x), alpha, beta], axis=1)
    assert points.shape == (len(x), 3)

    tab_pdf = read_from_cache("pdf")
    res = interp_catmul_nd(points, tab_pdf, pylevy._lower, pylevy._upper)
    return res


def dot(
    alpha: float,
    weights: JArray | NPArray | float,
    param: Param,
    locs: JArray | NPArray,
    scales: JArray | NPArray,
) -> Tuple[JArray, JArray]:
    """
    Performs a dot product operation on multiple stable distributions at once.

    Returns a pair of (loc, scale) in the given parametrization.
    """
    assert param == Params.N1
    ws: JArray
    if isinstance(weights, JArray):
        ws = weights
    else:
        raise NotImplementedError
    # TODO: check ws >= 0
    # TODO: check rows > 0
    gammas: JArray
    if isinstance(scales, JArray):
        gammas = scales
    else:
        raise NotImplementedError
    # TODO: check > 0
    deltas: JArray
    if isinstance(locs, JArray):
        deltas = locs
    else:
        raise NotImplementedError

    return _dot_n1(alpha, ws, gammas=gammas, deltas=deltas, extra_loc=None)


def sum(
    alpha: float,
    param: Param,
    locs: JArray | NPArray,
    scales: JArray | NPArray,
    axis=None,
) -> Tuple[JArray, JArray]:
    """
    Peforms a sum of Levy-stable distributions, as speficied by the axis.

    The coefficients (location and scale) are expected to be in an tensor,
    and the sum will be done along the speficied axis.

    Returns a pair of (loc, scale) after the reduction along the given axis.
    """
    assert param == Params.N1
    gammas: JArray
    if isinstance(scales, JArray):
        gammas = scales
    else:
        raise NotImplementedError
    # TODO: check > 0
    deltas: JArray
    if isinstance(locs, JArray):
        deltas = locs
    else:
        raise NotImplementedError

    return _sum_n1(alpha, gammas=gammas, deltas=deltas, axis=axis)


def _sum_n1(alpha: float, gammas: JArray, deltas: JArray, axis=None):
    gamma_alpha = jnp.power(gammas, alpha)
    gamma_res = jnp.power(jnp.sum(gamma_alpha, axis=axis), 1.0 / alpha)
    delta_res = jnp.sum(deltas, axis=axis)
    return (delta_res, gamma_res)


def _dot_n1(
    alpha: float,
    weights: JArray,
    gammas: JArray,
    deltas: JArray,
    extra_loc: Optional[JArray],
) -> Tuple[JArray, JArray]:
    """
    Performs a dot product of multiple stable distributions.
    """
    gamma_alpha = jnp.power(gammas, alpha)
    weights_alpha = jnp.power(weights, alpha)
    gammas_res = jnp.power(jnp.dot(weights_alpha, gamma_alpha), 1.0 / alpha)
    delta_res = jnp.dot(weights, deltas)
    if extra_loc is not None:
        delta_res = delta_res + extra_loc
    return (delta_res, gammas_res)
