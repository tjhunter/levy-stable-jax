# Introduction

Fast and robust implementation of the univariate [Lévy alpha-stable family of distribution](https://en.wikipedia.org/wiki/Stable_distribution) for 
the [JAX numerical framework](https://jax.readthedocs.io/en/latest/).

The Lévy-stable distribution is also known as the alpha-stable distribution or simply the stable distribution. It is a generalization of the Gaussian 
distribution for skewed, possibly very large events. It is also the only class of distribution that is close under both scaling and summation.

This distribution is particularly suitable when a) large values could be 
reasonably expected (more than twice the standard deviation for example), b)
for mathematical convenience (very similar to the Gaussian distribution).

Despite all these advantages, the Lévy-stable distribution has shown limited uptake in the scientific world, one reason being that it is very
challenging to achieve a robust and fast implementation.

This implementation of the Lévy-stable distribution is written entirely 
with the Jax framework, and it is highly optimized for speed and for robustness, at the expense of some accuracy.

- measured as 50x faster for maximum likelihood estimation compared to scipy's reference implementation for limited loss in accuracy

- highly robust: returns correct values and gradients for all ranges of 
alpha in (1.-2] and beta in [-1,1]. Nearly all existing implementations 
fail on various corner cases and return wrong or infinite values for some
ranges of parameters.

- implemented in pure Jax code: all functions are suitable for vectorization, including using a GPU or a TPU.

## Installation

This package is written in pure Jax code, with the reliance on numpy and 
scipy for constant special operations. It depends explicitly on 
`jax`, `numpy` and `scipy`.

For estimation, the `jaxopt` package is also required (not listed as a dependency)

For MCMC inference, the following packages are needed: `pymc>=5`, `numpyro`.

## Notations and parametrizations

Stable distributions can be parametrized in many ways. This package follows
the notations of Nolan (2020) internally, and it provides 2 main parametrizations, listed in the `Params` object.

- `N0` (also known as `S0`): preferred for numerical work
- `N1` (also known as `S1`): the most intuitive

## Deviations from scipy

This package uses tabulated values from scipy's levy_stable package. 
Since there are no easy formulas for calculating the density function 
of stable distributions, interpolation provides fast and reasonably
accurate values for the core of the distribution. For the tails, it 
directly implements known formulas.

This package deviates from `scipy` in the case beta = 1 and beta = -1,
for which the scipy implementation has been found to not respect well-established tail formulas. 