import numpy as np
from scipy.stats import levy_stable as sp_levy_stable  # type: ignore
import scipy.interpolate  # type: ignore
import logging
import os

from .distribution import (
    TAB_X_CUTOFF,
    ALPHA_MIN,
    ALPHA_MAX,
    BETA_MIN,
    BETA_MAX,
    NUM_X_POINTS,
    NUM_ALPHA_POINTS,
    NUM_BETA_POINTS,
)
from ._typing import Params
from ._utils import set_stable

_logger = logging.getLogger(__name__)

_THRESH_D2 = 1.0
_KINK_DELTA = 1e-1


def make_data_files():
    """
    Generates the lookup tables.

    This is based on the pylevy's code,
    but it uses instead the scipy implementation
    of the Levy distribution.
    - Some of the values are inaccurate and lead to oscillatory behaviour
    - it generates the log-pdf rather than the pdf itself
    - it generates floats rather than doubles
    """
    # TODO: consider a further transform to tan(x) as in pylevy
    # This concentrates the points to the area of interest (the center of the distribution)
    xs = np.linspace(-TAB_X_CUTOFF, TAB_X_CUTOFF, NUM_X_POINTS)
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, NUM_ALPHA_POINTS)
    betas = np.linspace(BETA_MIN, BETA_MAX, NUM_BETA_POINTS)
    shape = (NUM_X_POINTS, NUM_ALPHA_POINTS, NUM_BETA_POINTS)
    dx = 2 * TAB_X_CUTOFF / NUM_X_POINTS
    _logger.info(f"Generating log-pdf shape: {shape}")
    # Calculations are done in the N0 domain for numerical stability.
    logpdf = np.zeros(shape, np.float32)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            with set_stable(Params.N0):
                logpdf[:, i, j] = _gen_clean_vals(alpha, beta, xs, dx)
                _logger.info(
                    f"Calculating alpha={alpha} beta={beta} {sp_levy_stable.parameterization}"
                )
    ROOT = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT, "logpdf.npy")
    _logger.info(f"Saving logpdf to {path}")
    np.save(path, logpdf)


def _is_kink(z):
    before = np.array(z)
    before[:-1] = z[1:]
    after = np.array(z)
    after[1:] = z[:-1]
    epsi = _KINK_DELTA
    return ((z > before + epsi) & (z > after + epsi)) | (
        (z < before - epsi) & (z < after - epsi)
    )


def _gen_clean_vals(alpha, beta, xs, dx):
    """
    Generates the values for the logpdf using scipy's code, and perform various
    checks.

    The code from scipy has a number of bugs which this function corrects.
    - for beta close to 1/-1, the values are incorrect in the tail.
    - for some values close to the mode, one value may be jumping far.
    """
    ys = sp_levy_stable.logpdf(xs, alpha=alpha, beta=beta)
    ys1 = np.gradient(ys) / dx
    ys2 = np.gradient(ys1) / dx
    # Some values around the mode are incorrect:
    # They tend to jump quite far from the values around them.
    # A particularly bad example is sp_levy_stable.logpdf(xs, 1.1,-0.14) for N0.
    # TODO: open a bug against scipy.
    jump_mask = _is_kink(ys)
    if np.sum(jump_mask) > 1:
        _logger.warning(
            f"Jumping values for alpha={alpha} beta={beta}: {np.sum(jump_mask)}"
        )
    f = scipy.interpolate.interp1d(xs[~jump_mask], ys[~jump_mask], bounds_error=False)
    y_corr = np.where(jump_mask, f(xs), ys)
    # For beta close to 1/-1, the values are incorrect in the tail.
    if np.abs(beta) > 0.9:
        # It is noticeable because there is an abrupt kink in the logpdf.
        # The second derivative is large at this place, while it should be decreasing.
        # TODO: open a bug against scipy.
        # A good heuristic is to remove values that have a large second derivative.
        ys2 = np.gradient(np.gradient(y_corr)) / (dx * dx)
        inval_mask = ys2 > _THRESH_D2
        if np.any(inval_mask):
            idx = min(np.arange(NUM_X_POINTS)[inval_mask])
            if beta < 0:
                inval_mask[idx:] = True
            else:
                inval_mask[:idx] = True
            # Settting the value to -inf to differentiate from NaNs
            # that may be coming from elsewhere, such as gradient.
            y_corr[inval_mask] = -np.inf
    assert np.all(~np.isnan(y_corr)), (alpha, beta, xs, y_corr)
    return y_corr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    make_data_files()
