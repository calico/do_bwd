# Ported from SSM initial state distributions
# [https://github.com/lindermanlab/ssm/blob/master/ssm/init_state_distns.py]
# Modified to fit the VIARHMM class

from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp

from viarhmm.utils import ensure_args_are_lists
from viarhmm.utils import LOG_EPS

class InitialStateDistribution(object):
    def __init__(self, K, D, M=0, source=False):
        self.K, self.D, self.M = K, D, M
        self.log_pi0 = -np.log(K) * np.ones(K)
        if source:
            print("Enabled source state.")
            self.log_pi0 = np.log(LOG_EPS) * np.ones(K)
            self.log_pi0[0] = np.log(1 - LOG_EPS*(K-1))

    @property
    def params(self):
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        self.log_pi0 = value[0]

    @property
    def initial_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    @property
    def log_initial_state_distn(self):
        return self.log_pi0 - logsumexp(self.log_pi0)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_pi0 = self.log_pi0[perm]

    def log_prior(self):
        return 0

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        pi0 = sum([Ez[0] for Ez, _, _ in expectations]) + 1e-8
        self.log_pi0 = np.log(pi0 / pi0.sum())

class VIInitialStateDistribution(InitialStateDistribution):
    """
    Standard Hidden Markov Model with fixed initial distribution.
    """
    def __init__(self, K, D, M):
        super(VIInitialStateDistribution, self).__init__(K, D, M=M)
        self.log_pi0 = -np.log(K) * np.ones((K, 1))

    @property
    def params(self):
        return self.log_pi0

    @params.setter
    def params(self, value):
        self.log_pi0 = value

    @property
    def initial_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    @property
    def log_initial_state_distn(self):
        return self.log_pi0 - logsumexp(self.log_pi0)

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_pi0 = self.log_pi0[perm]

    def log_prior(self):
        return 0

    def m_step(self, datas, inputs, masks, tags, **kwargs):
        K, M = self.K ,self.M
        N = len(datas)
        pi0 = np.zeros((K, 1))

        # convert probabilities from log to simplex
        tilda_log_pis = kwargs['tilda_log_pis']
        tilda_pis = {}
        for data, tag in zip(datas, tags):
            tilda_pis[tag] = np.exp(tilda_log_pis[tag] - logsumexp(tilda_log_pis[tag], axis=0))
            assert np.rint(sum(tilda_pis[tag].sum(axis=0))) == (data.shape[0])
            pi0 += tilda_pis[tag][:, [0]]
        pi0 += LOG_EPS

        # m-step
        self.log_pi0 = np.log(pi0 / pi0.sum()).copy()