# Levy-stable-jax

Implementation of the Lévy-stable distributions for the Jax framework.

The Lévy-stable distribution is a generalization of the Normal, Cauchy and Lévy distributions. It enjoys many appealing properties, but it has no 
known closed-form solutions.

This package provides an implementation using the Jax framework. With this package, you can use this distribution with all AI frameworks based on Jax. In particular, you can use Lévy-stable distributions in Bayesian inference 
frameworks such as `numpyro` or `PyMC`. 

This implementation is optimized for speed and ease of use with Jax.
 It uses the scipy implementation as a reference.


It also includes various experimental methods for fitting Lévy-stable
distributions. These should be considered experimental, both in API and in
functionality.

## Current limitations

- values of alpha are only supported in the range (1.01 - 2). This code may work
  for lower values of alpha but it has not been tested.

- only one parametrization (Nolan's N0 notation) is implemented. It is the more stable parametrization for numerical computations.

## Commands (development only)


Setting up the test environment

```
pip install .[dev]
make lint
make test
poetry build
poetry publish

```