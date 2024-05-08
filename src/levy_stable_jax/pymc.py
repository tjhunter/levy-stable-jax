"""
Monte Carlo sampling of Levy-stable distributions using PyMC.

As far as users are concerned, the only relevant function should be `LevyStableN0`.

A full example notebook is available here:
https://github.com/tjhunter/levy-stable-jax/blob/main/notebooks/pymc_levy.ipynb

"""

# The source of most of this code are the following tutorials:
# https://www.pymc-labs.com/blog-posts/jax-functions-in-pymc-3-quick-examples/
# https://www.pymc.io/projects/examples/en/2022.12.0/howto/custom_distribution.html


import jax.numpy as jnp
import numpy as np
import jax
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
import pymc as pm  # type: ignore

from .distribution import logpdf as levy_stable_logpdf
from .distribution import Params

__all__ = ["LevyStableN0"]


def LevyStableN0(name, alpha, beta, loc, scale, observed=None):
    """
    A Lévy-stable distribution. The parametrization follows the "0" notation
    from Nolan (2022). It is also known as the "S0" parametrization in scipy.

    Args:
        alpha: the stability parameter. Must be in (1.1, 2].
        beta: the skewness parameter. Must be in [-1, 1].
        loc: the location parameter (delta in Nolan's notation).
        scale: the scale parameter (gamma in Nolan's notation).

    This distribution is explicitly implemented and parametrized for the N0
    parametrization, because the N1 parametrization is not continuous
    for alpha close to 1, and is not recommended in general for numerical work.
    Use the `shift_scale` function if you wish to convert to other parametrizations.

    **Practical tips**

    It is currently only implemented for alpha in (1.1 - 2.0]. Smaller values
    will get trimmed.

    In the context of MCMC, it is highly recommended to keep beta in (-0.8, 0.8).
    The Lévy-stable distribution changes abruptly around values of beta close to
    1 or -1. As a result, the sampler will reject many values for small changes
    around these values. The symptoms will be a high number of divergences and slow
    progress. It is better to trim the range of accepted values of beta and inspect
    if the posterior distribution is highly skewed towards the extremes.

    You cannot really infer a good value for |beta| > 0.8, unless you have an
    extremely large amount of points.
    In addition, all other parameters being equal, the log-likelihood as a function of beta
    is very flat around |beta| ~ (0.8-0.99) and alpha > 1.5: a very large number
    of observations would be necessary to observed a rare event in the non-heavy tail that
    would allow to distinguish between these values.

    Because of the rather complex and non-convex shape of the log-likelihood,
    the sampler may struggle to initially reach a region of high probability.
    I have found that tune=6000 really helps further convergence.

    Since all the code is implemented in JAX, a JAX-based sampler will be much faster.
    It is recommended to use `pymc.sampling_jax.sample_numpyro_nuts`

    TODO: turn into a proper distribution.

    TODO: add a sampler.


    """
    return pm.CustomDist(
        name, alpha, beta, loc, scale, logp=_levy_stable_logp_op, observed=observed
    )


def _logp_sum_n0(x, alpha, beta, loc, scale):
    # TODO: it should already return a scalar, but currently returning a vector
    return jnp.sum(levy_stable_logpdf(x, alpha, beta, loc, scale, Params.N0))


_jit_logp_n0 = jax.jit(_logp_sum_n0)
_jit_logp_n0_grad = jax.jit(jax.grad(_logp_sum_n0, argnums=list(range(5))))


class LevyStableLogpOp(Op):
    def make_node(self, x, alpha, beta, loc, scale):
        # Create a PyTensor node specifying the number and type of inputs and outputs

        # We convert the input into a PyTensor tensor variable
        inputs = [pt.as_tensor_variable(z) for z in (x, alpha, beta, loc, scale)]
        # Output has the same type and shape as `x`
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        # Evaluate the Op result for a specific numerical input

        # The inputs are always wrapped in a list
        (x, alpha, beta, loc, scale) = inputs
        result = _jit_logp_n0(x, alpha, beta, loc, scale)
        # The results should be assigned inplace to the nested list
        # of outputs provided by PyTensor. If you have multiple
        # outputs and results, you should assign each at outputs[i][0]
        outputs[0][0] = np.asarray(result, dtype="float64")

    def grad(self, inputs, output_gradients):
        # Create a PyTensor expression of the gradient
        (x, alpha, beta, loc, scale) = inputs
        (gz,) = output_gradients
        g = _levy_stable_logp_grad_op(x, alpha, beta, loc, scale)
        # print("LevyStableLogpOp", "grad", output_gradients, g)
        (g_x, g_alpha, g_beta, g_loc, g_scale) = g
        return [gz * g_x, gz * g_alpha, gz * g_beta, gz * g_loc, gz * g_scale]


class LevyStableLogpGradOp(Op):
    def make_node(self, x, alpha, beta, loc, scale):
        # Make sure the two inputs are tensor variables
        inputs = [pt.as_tensor_variable(z) for z in (x, alpha, beta, loc, scale)]
        # Output has the shape type and shape as the first input
        outputs = [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        (x, alpha, beta, loc, scale) = inputs
        g = _jit_logp_n0_grad(x, alpha, beta, loc, scale)
        # print("LevyStableLogpGradOp perform", "g", g)
        (g_x, g_alpha, g_beta, g_loc, g_scale) = g
        outputs[0][0] = np.asarray(g_x, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(g_alpha, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(g_beta, dtype=node.outputs[2].dtype)
        outputs[3][0] = np.asarray(g_loc, dtype=node.outputs[3].dtype)
        outputs[4][0] = np.asarray(g_scale, dtype=node.outputs[4].dtype)


_levy_stable_logp_op = LevyStableLogpOp()
_levy_stable_logp_grad_op = LevyStableLogpGradOp()


@jax_funcify.register(LevyStableLogpOp)
def _levy_stable_logp_dispatch(op, **kwargs):
    return _logp_sum_n0
