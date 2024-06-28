# Ported from SSM stats
# [https://github.com/lindermanlab/ssm/blob/master/ssm/stats.py]

import autograd.numpy as np
from autograd.scipy.special import logsumexp

def flatten_to_dim(X, d):
    """
    Flatten an array of dimension k + d into an array of dimension 1 + d.

    Example:
        X = npr.rand(10, 5, 2, 2)
        flatten_to_dim(X, 4).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 3).shape # (10, 5, 2, 2)
        flatten_to_dim(X, 2).shape # (50, 2, 2)
        flatten_to_dim(X, 1).shape # (100, 2)

    Parameters
    ----------
    X : array_like
        The array to be flattened.  Must be at least d dimensional

    d : int (> 0)
        The number of dimensions to retain.  All leading dimensions are flattened.

    Returns
    -------
    flat_X : array_like
        The input X flattened into an array dimension d (if X.ndim == d)
        or d+1 (if X.ndim > d)
    """
    assert X.ndim >= d
    assert d > 0
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def diagonal_gaussian_logpdf(data, mus, sigmasqs, mask=None):
    """
    Compute the log probability density of a Gaussian distribution with
    a diagonal covariance.  This will broadcast as long as data, mus,
    sigmas have the same (or at least compatible) leading dimensions.

    Parameters
    ----------
    data : array_like (..., D)
        The points at which to evaluate the log density

    mus : array_like (..., D)
        The mean(s) of the Gaussian distribution(s)

    sigmasqs : array_like (..., D)
        The diagonal variances(s) of the Gaussian distribution(s)

    mask : array_like (..., D) bool
        Optional mask indicating which entries in the data are observed

    Returns
    -------
    lps : array_like (...,)
        Log probabilities under the diagonal Gaussian distribution(s).
    """
    # Check inputs
    D = data.shape[-1]
    assert mus.shape[-1] == D
    assert sigmasqs.shape[-1] == D

    # Check mask
    mask = mask if mask is not None else np.ones_like(data, dtype=bool)
    assert mask.shape == data.shape

    normalizer = -0.5 * np.log(2 * np.pi * sigmasqs)
    return np.sum((normalizer - 0.5 * (data - mus)**2 / sigmasqs) * mask, axis=-1)