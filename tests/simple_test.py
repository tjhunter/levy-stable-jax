from jaxopt._src import test_util
from scipy.stats import levy_stable as sp_levy_stable  # type: ignore
import levy_stable_jax as lsj
import levy_stable_jax._utils
from levy_stable_jax import Params
from hypothesis import (
    given,
    strategies as st,
    settings,
    Verbosity,
    Phase,
)
import jax.numpy as jnp
import numpy as np

_ATOL_LOGP = 0.33
_RTOL_LOGP = 7e-2
_ATOL_PDF = 2e-3
_RTOL_PDF = 0.7
# The distribution has many corner cases, it pays off to have
# a high number of examples.
_MAX_EXAMPLES = 100

# The full range of values allowed for alpha
_alpha_st = st.floats(min_value=1.1, max_value=2.0)
# The full range of values allowed for beta
_beta_st = st.floats(min_value=-1.0, max_value=1.0)
_locs_st = st.floats(min_value=-100.0, max_value=100.0)
_scales_st = st.floats(min_value=1e-2, max_value=1e2)
_param_st = st.sampled_from([Params.N0])  # TODO: add N1

# Hypothesis tries (rightfully) to generate examples close to the boundaries.
_alpha_restricted_st = st.floats(min_value=1.2, max_value=2.0 - 1e-6)
# TODO: allow beta >= -1.0
# Scipy has an incorrect implementation for |beta| == 1, disregard these values for the time being.
_beta_sp_st = st.floats(min_value=-1 + 1e-5, max_value=1 - 1e-5)

simple_settings = settings(
    max_examples=_MAX_EXAMPLES,
    verbosity=Verbosity.verbose,
    phases=[Phase.generate],
    derandomize=True,
    deadline=8000,  # High value so that the github runner does not timeout.
)


# Some values in the interpolation range have low accuracies for specific values of alpha and beta.
# For now, the tests are considering these as passing, if it is just one value.
def _compare_logpdf(alpha, beta, param, xs, loc=None, scale=None):
    with levy_stable_jax._utils.set_stable(param):
        if loc is None or scale is None:
            exp = sp_levy_stable.logpdf(xs, alpha, beta)
        else:
            exp = sp_levy_stable.logpdf(xs, alpha, beta, loc=loc, scale=scale)
    if loc is None or scale is None:
        ys = lsj.logpdf(xs, alpha, beta, param=param)
    else:
        ys = lsj.logpdf(xs, alpha, beta, param=param, loc=loc, scale=scale)
    np.testing.assert_allclose(
        ys,
        exp,
        atol=_ATOL_LOGP,
        rtol=0,
    )
    np.testing.assert_allclose(
        ys,
        exp,
        atol=0,
        rtol=_RTOL_LOGP,
    )


def _compare_pdf(alpha, beta, param, xs):
    with levy_stable_jax._utils.set_stable(param):
        exp = sp_levy_stable.pdf(xs, alpha, beta)
    ys = lsj.pdf(xs, alpha, beta, param=param)
    np.testing.assert_allclose(
        ys,
        exp,
        atol=_ATOL_PDF,
        rtol=0,
    )
    np.testing.assert_allclose(
        ys,
        exp,
        atol=0,
        rtol=_RTOL_PDF,
    )


class PdfTest(test_util.JaxoptTestCase):
    xs = jnp.linspace(-40, 40, 60)

    @settings(simple_settings)
    @given(_alpha_restricted_st, _beta_sp_st, _param_st)
    def test_logpdf_range_unit(self, alpha, beta, param):
        _compare_logpdf(alpha, beta, param, self.xs)

    @settings(simple_settings)
    @given(_beta_sp_st, _param_st)
    def test_logpdf_range_gaussian(self, beta, param):
        _compare_logpdf(2.0, beta, param, self.xs)

    @settings(simple_settings)
    @given(_alpha_restricted_st, _beta_sp_st, _param_st)
    def test_pdf_range(self, alpha, beta, param):
        _compare_pdf(alpha, beta, param, self.xs)

    @settings(simple_settings)
    @given(_beta_sp_st, _param_st)
    def test_pdf_range_gaussian(self, beta, param):
        _compare_pdf(2.0, beta, param, self.xs)

    @settings(simple_settings)
    @given(_alpha_restricted_st, _beta_sp_st, _param_st, _locs_st, _scales_st)
    def test_logpdf_range(self, alpha, beta, param, loc, scale):
        _compare_logpdf(alpha, beta, param, self.xs, loc=loc, scale=scale)


class SanityTest(test_util.JaxoptTestCase):
    xs = jnp.linspace(-100, 100, 1000)

    @settings(simple_settings)
    @given(_alpha_st, _beta_st, _locs_st, _scales_st, _param_st)
    def test_logpdf_real(self, alpha, beta, loc, scale, param):
        ys = lsj.logpdf(self.xs, alpha, beta, param=param, loc=loc, scale=scale)
        assert np.all(np.isreal(ys))
        assert np.all(~np.isnan(ys))
        assert np.all(np.isfinite(ys))

    @settings(simple_settings)
    @given(_alpha_st, _beta_st, _param_st)
    def test_logpdf_real_unit(self, alpha, beta, param):
        ys = lsj.logpdf(self.xs, alpha, beta, param=param)
        assert np.all(np.isreal(ys))
        assert np.all(~np.isnan(ys))
        assert np.all(np.isfinite(ys))

    def test_logpdf_real_1(self):
        (alpha, beta) = (1.6, 1)
        ys = lsj.logpdf(self.xs, alpha, beta, param=Params.N0)
        assert np.all(~np.isnan(ys))
        assert np.all(np.isfinite(ys))
