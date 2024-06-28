# Ported from SSM initial state distributions
# [https://github.com/lindermanlab/ssm/blob/master/ssm/transitions.py]
# Modified to fit the VIARHMM class

from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import hessian

from viarhmm.optimizers import adam, bfgs, lbfgs, rmsprop, sgd
from viarhmm.utils import ensure_args_are_lists, LOG_EPS

class Transitions(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, random_state=None):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        return 0

    def log_transition_matrices(self, data, input, mask, tag):
        raise NotImplementedError

    def transition_matrices(self, data, input, mask, tag):
        return np.exp(self.log_transition_matrices(data, input, mask, tag))

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="lbfgs", num_iters=1000, **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to BFGS.
        """
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[optimizer]

        # Maximize the expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # Normalize and negate for minimization
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            optimizer(_objective, self.params, num_iters=num_iters,
                      state=optimizer_state, full_output=True, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        warn("Analytical Hessian is not implemented for this transition class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        obj = lambda x, E_zzp1: np.sum(E_zzp1 * self.log_transition_matrices(x, input, mask, tag))
        hess = hessian(obj)
        terms = np.array([-1 * hess(x[None,:], Ezzp1) for x, Ezzp1 in zip(data, expected_joints)])
        return terms

class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M=0, random_state=None):
        super(StationaryTransitions, self).__init__(K, D, M=M)
        if random_state is None:
            Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        else:
            rs = npr.RandomState(random_state)
            Ps = .95 * np.eye(K) + .05 * rs.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps + LOG_EPS)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K = self.K
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-32
        P = np.nan_to_num(P / P.sum(axis=-1, keepdims=True))

        # Set rows that are all zero to uniform
        P = np.where(P.sum(axis=-1, keepdims=True) == 0, 1.0 / K, P)
        if ('sink' in kwargs) and (kwargs['sink']):
            tmp = np.zeros(K)
            tmp[K-1] = 1
            P[K-1, :] = tmp
        log_P = np.log(P)
        self.log_Ps = log_P - logsumexp(log_P, axis=-1, keepdims=True)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))


class VIStationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M, random_state=None):
        super(VIStationaryTransitions, self).__init__(K, D, M=M)
        if random_state is None:
            Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        else:
            rs = npr.RandomState(random_state)
            Ps = .95 * np.eye(K) + .05 * rs.rand(K, K)
        
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return self.log_Ps

    @params.setter
    def params(self, value):
        self.log_Ps = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, datas, inputs, masks, tags, **kwargs):

        K, M = self.K ,self.M
        N = len(datas)

        # convert probabilities from log to simplex
        tilda_log_pis = kwargs['tilda_log_pis']
        tilda_pis = {}
        for data, tag in zip(datas, tags):
            tilda_pis[tag] = np.exp(tilda_log_pis[tag] - logsumexp(tilda_log_pis[tag], axis=0))
            assert np.rint(sum(tilda_pis[tag].sum(axis=0))) == (data.shape[0])

        # m-step
        Ps = np.zeros((K, K))
        for data, tag in zip(datas, tags):
            T = data.shape[0]
            for t in np.arange(M, T):
                Ps += np.dot(tilda_pis[tag][:, [t]], tilda_pis[tag][:, [t-1]].T)

        # check inf values
        # P = np.nan_to_num(P / P.sum(axis=-1, keepdims=True))

        # Set rows that are all zero to uniform
        # P = np.where(P.sum(axis=-1, keepdims=True) == 0, 1.0 / K, P)
        Ps /= Ps.sum(axis=1, keepdims=True)
        log_P = np.log(Ps + LOG_EPS)
        self.log_Ps = (log_P - logsumexp(log_P, axis=1, keepdims=True)).copy()