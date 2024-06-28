# Ported from SSM messages
# [https://github.com/lindermanlab/ssm/blob/master/ssm/messages.py]

import os
import numba
import pickle
import numpy as np
import numpy.random as npr
import scipy.special as scsp
from functools import partial


from autograd.tracer import getval
from autograd.extend import primitive, defvjp
from viarhmm.utils import LOG_EPS, DIV_EPS

PATH = os.path.join(os.getcwd(), 'models', 'best_model', 'state_transitions')

to_c = lambda arr: np.copy(getval(arr), 'C') if not arr.flags['C_CONTIGUOUS'] else getval(arr)

@numba.jit(nopython=True, cache=True)
def logsumexp(x):
    """
    Compute the log of the sum of exponentials of x in a numerically stable way.

    Parameters
    ----------
    x : ndarray
        Array of values.

    Returns
    -------
    logsumexp : float
        The log of the sum of exponentials of the input array.
    """
    N = x.shape[0]

    # find the max
    m = -np.inf
    for i in range(N):
        m = max(m, x[i])

    # sum the exponentials
    out = 0
    for i in range(N):
        out += np.exp(x[i] - m)

    return m + np.log(out)


@numba.jit(nopython=True, cache=True)
def dlse(a, out):
    K = a.shape[0]
    lse = logsumexp(a)
    for k in range(K):
        out[k] = np.exp(a[k] - lse)


@numba.jit(nopython=True, cache=True)
def forward_pass(pi0, Ps, log_likes, alphas):
    """
    Forward pass for the HMM.

    Parameters
    ----------
    pi0 : ndarray
        Initial state distribution
    Ps : ndarray
        Transition matrix
    log_likes : ndarray
        Log likelihoods
    alphas : ndarray
        Storage for the forward messages
    
    Returns
    -------
    log_likelihood : float
        The log likelihood of the data under the model.
    """

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert alphas.shape[0] == T
    assert alphas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    alphas[0] = np.log(pi0) + log_likes[0]
    for t in range(T-1):
        m = np.max(alphas[t])
        alphas[t+1] = np.log(np.dot(np.exp(alphas[t] - m), Ps[t * hetero])) + m + log_likes[t+1]
    return logsumexp(alphas[T-1])


@numba.jit(nopython=True, cache=True)
def hmm_filter(pi0, Ps, ll):
    """
    Filter the HMM.

    Parameters
    ----------
    pi0 : ndarray
        Initial state distribution
    Ps : ndarray
        Transition matrix
    ll : ndarray
        Log likelihoods

    Returns
    -------
    pz_tt : ndarray
        Filtered state probabilities
    """
    T, K = ll.shape

    # Forward pass gets the predicted state at time t given
    # observations up to and including those from time t
    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)

    # Check if using heterogenous transition matrices
    hetero = (Ps.shape[0] == T-1)

    # Predict forward with the transition matrix
    pz_tt = np.empty((T-1, K))
    pz_tp1t = np.empty((T-1, K))
    for t in range(T-1):
        m = np.max(alphas[t])
        pz_tt[t] = np.exp(alphas[t] - m)
        pz_tt[t] /= np.sum(pz_tt[t])
        pz_tp1t[t] = pz_tt[t].dot(Ps[hetero*t])

    # Include the initial state distribution
    # Numba's version of vstack requires all arrays passed to vstack
    # to have the same number of dimensions.
    pi0 = np.expand_dims(pi0, axis=0)
    pz_tp1t = np.vstack((pi0, pz_tp1t))

    # Numba implementation of np.sum does not allow axis keyword arg,
    # and does not support np.allclose, so we loop over the time range
    # to verify that each sums to 1.
    for t in range(T):
        assert np.abs(np.sum(pz_tp1t[t]) - 1.0) < 1e-8

    return pz_tp1t


@numba.jit(nopython=True, cache=True)
def backward_pass(Ps, log_likes, betas):
    """
    Backward pass of HMM.

    Parameters
    ----------
    Ps : ndarray
        Transition matrix
    log_likes : ndarray
        Log likelihoods
    betas : ndarray
        Storage for the backward messages

    Returns
    -------
        None
    """

    T = log_likes.shape[0]  # number of time steps
    K = log_likes.shape[1]  # number of discrete states

    assert Ps.shape[0] == T-1 or Ps.shape[0] == 1
    assert Ps.shape[1] == K
    assert Ps.shape[2] == K
    assert betas.shape[0] == T
    assert betas.shape[1] == K

    # Check if we have heterogeneous transition matrices.
    # If not, save memory by passing in log_Ps of shape (1, K, K)
    hetero = (Ps.shape[0] == T-1)
    tmp = np.zeros(K)

    # Initialize the last output
    betas[T-1] = 0
    for t in range(T-2,-1,-1):
        tmp = log_likes[t+1] + betas[t+1]
        m = np.max(tmp)
        betas[t] = np.log(np.dot(Ps[t * hetero], np.exp(tmp - m))) + m


# @numba.jit(nopython=True, cache=True)
def _compute_stationary_expected_joints(alphas, betas, lls, log_P, E_zzp1):
    """
    Helper function to compute summary statistics, summing over time.
    NOTE: Can rewrite this in nicer form with Numba.

    Parameters
    ----------
    alphas : ndarray
        Forward messages
    betas : ndarray
        Backward messages
    lls : ndarray
        Log likelihoods
    log_P : ndarray
        Log transition matrix
    E_zzp1 : ndarray
        Storage for the expected joint probabilities

    Returns
    -------
        None
    """
    T = alphas.shape[0]
    K = alphas.shape[1]
    assert betas.shape[0] == T and betas.shape[1] == K
    assert lls.shape[0] == T and lls.shape[1] == K
    assert log_P.shape[0] == K and log_P.shape[1] == K
    assert E_zzp1.shape[0] == K and E_zzp1.shape[1] == K

    tmp = np.zeros((K, K))
    # joints = np.zeros((K, K, T-1))

    # Compute the sum over time axis of the expected joints
    for t in range(T-1):
        maxv = -np.inf
        for i in range(K):
            for j in range(K):
                # Compute expectations in this batch
                tmp[i, j] = alphas[t,i] + betas[t+1,j] + lls[t+1,j] + log_P[i, j]
                # joints[i, j, t] = alphas[t,i] + betas[t+1,j] + lls[t+1,j] + log_P[i, j]

                if tmp[i, j] > maxv:
                    maxv = tmp[i, j]

        # safe exponentiate
        tmpsum = 0.0
        for i in range(K):
            for j in range(K):
                tmp[i, j] = np.exp(tmp[i, j] - maxv)
                tmpsum += tmp[i, j]

        # Add to expected joints
        for i in range(K):
            for j in range(K):
                E_zzp1[i, j] += tmp[i, j] / (tmpsum + DIV_EPS)

    ## uncomment the code below to save transition matrix at each iteration
    # if tag is not None:
    #     joints = joints.reshape(K*K, T-1)
    #     joints_logsumexp = np.zeros(joints.shape)
    #     for t in range(T-1):
    #         joints_logsumexp[:, t] = logsumexp(joints[:, t])
    #     joints = joints - joints_logsumexp
    #     joints = np.exp(joints)
    #     with open(os.path.join(PATH, tag + '.pkl'), 'wb') as f:
    #         pickle.dump(joints, f)
    #     f.close()


def hmm_expected_states(pi0, Ps, ll):
    """
    Compute the expected states and expected state transitions

    Parameters
    ----------
    pi0 : ndarray
        Initial state distribution
    Ps : ndarray
        Transition matrix
    ll : ndarray
        Log likelihoods

    Returns
    -------
    expected_states : ndarray
        Expected states
    """
    T, K = ll.shape

    alphas = np.zeros((T, K))
    forward_pass(pi0, Ps, ll, alphas)
    normalizer = logsumexp(alphas[-1])

    betas = np.zeros((T, K))
    backward_pass(Ps, ll, betas)

    # Compute E[z_t] for t = 1, ..., T
    expected_states = alphas + betas
    expected_states -= scsp.logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)

    # Compute the log transition matrices.
    # Suppress log(0) warnings as they are expected.
    with np.errstate(divide="ignore"):
        log_Ps = np.log(Ps)


    # Compute E[z_t, z_{t+1}] for t = 1, ..., T-1
    # Note that this is an array of size T*K*K, which can be quite large.
    # To be a bit more frugal with memory, first check if the given log_Ps
    # are TxKxK.  If so, instantiate the full expected joints as well, since
    # we will need them for the M-step.  However, if log_Ps is 1xKxK then we
    # know that the transition matrix is stationary, and all we need for the
    # M-step is the sum of the expected joints.
    stationary = (Ps.shape[0] == 1)
    if not stationary:
        expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
        expected_joints -= expected_joints.max((1,2))[:,None, None]
        expected_joints = np.exp(expected_joints)
        expected_joints /= expected_joints.sum((1,2))[:,None,None]

    else:
        # Compute the sum over time axis of the expected joints
        expected_joints = np.zeros((K, K))
        _compute_stationary_expected_joints(alphas, betas, ll, log_Ps[0], expected_joints)
        expected_joints = expected_joints[None, :, :]

    return expected_states, expected_joints, normalizer


@numba.jit(nopython=True, cache=True)
def _viterbi(pi0, Ps, ll):
    """
    This is modified from pyhsmm.internals.hmm_state by Matthew Johnson.

    Parameters
    ----------
    pi0 : ndarray
        Initial state distribution
    Ps : ndarray
        Transition matrix
    ll : ndarray
        Log likelihoods

    Returns
    -------
    z : ndarray
        Most likely state sequence
    """
    T, K = ll.shape

    # Check if the transition matrices are stationary or
    # time-varying (hetero)
    hetero = (Ps.shape[0] == T-1)
    if not hetero:
        assert Ps.shape[0] == 1

    # Pass max-sum messages backward
    scores = np.zeros((T, K))
    args = np.zeros((T, K))
    for t in range(T-2,-1,-1):
        vals = np.log(Ps[t * hetero] + LOG_EPS) + scores[t+1] + ll[t+1]
        for k in range(K):
            args[t+1, k] = np.argmax(vals[k])
            scores[t, k] = np.max(vals[k])

    # Now maximize forwards
    z = np.zeros(T)
    z[0] = (scores[0] + np.log(pi0 + LOG_EPS) + ll[0]).argmax()
    for t in range(1, T):
        z[t] = args[t, int(z[t-1])]

    return z


def viterbi(pi0, Ps, ll):
    """
    Find the most likely state sequence
    """
    return _viterbi(pi0, Ps, ll).astype(int)


@primitive
def hmm_normalizer(pi0, Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))

    # Make sure everything is C contiguous
    pi0 = to_c(pi0)
    Ps = to_c(Ps)
    ll = to_c(ll)

    forward_pass(pi0, Ps, ll, alphas)
    return logsumexp(alphas[-1])


