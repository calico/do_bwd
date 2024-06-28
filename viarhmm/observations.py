from __future__ import absolute_import, division
from __future__ import print_function
from math import exp, isfinite, log
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from viarhmm.messages import hmm_expected_states
from viarhmm.utils import ensure_args_are_lists
from viarhmm.utils import ensure_args_not_none
from viarhmm.utils import LOG_EPS, DIV_EPS


class VarInfARDiagGaussianObservations(object):
  """
  Base class of the variational inference method to
  solve Markov switching auto-regressive time-series
  data

  Args:
  ----
    K (int): number of discrete states
    M (int): lag order of the auto-regressive equation
    random_state (int): random_state

  """
  def __init__(self, 
              K, 
              M,
              random_state=None,
              **kwargs):

    self.K = K
    self.M = M
    self.random_state = random_state

    if kwargs["threshold"]:
      self.err_sigmasq_threshold = kwargs["threshold"]
    else:
      self.err_sigmasq_threshold = None

    # initial probability p(s_M), where M is the lag-order
    self.log_pi0 = -np.log(K) * np.ones((K, 1))

    if random_state is None:
      # fixed state transition matrix
      Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
      Ps /= Ps.sum(axis=1, keepdims=True)
      self.log_Ps = np.log(Ps)

      # auto-regressive parameters (K, M+1) 
      self.etas = np.random.normal(0.95, np.sqrt(0.1), (K, M+1))
      self.log_sigmasqs = -5 + npr.randn(K, M+1) * 0.05

      # error variances
      self.log_err_sigmasqs = -5 + npr.randn(K, 1) * 0.05
      
    else:
      # fixed state transition matrix
      rs = npr.RandomState(random_state)
      Ps = .95 * np.eye(K) + .05 * rs.rand(K, K)
      Ps /= Ps.sum(axis=1, keepdims=True)
      self.log_Ps = np.log(Ps)

      # auto-regressive parameters (K, M+1) 
      rs = npr.RandomState(random_state)
      self.etas = rs.normal(0.95, np.sqrt(0.1), (K, M+1))
      rs = npr.RandomState(random_state)
      self.log_sigmasqs = -5 + rs.randn(K, M+1) * 0.05

      # error variances
      rs = npr.RandomState(random_state)
      self.log_err_sigmasqs = -5 + rs.randn(K, 1) * 0.05
    
    return

  @ensure_args_are_lists
  def initalize_variational_params(self, 
                                  datas, 
                                  inputs=None, 
                                  masks=None, 
                                  tags=None, 
                                  random_state=None):
    """
    Function to initialize the varitional parameters. We denote the
    variational parameters with a superscript tilde in the derivation.
    Therefore, we use tilda prefix in all varitional parameters.

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace

    """

    K, M = self.K, self.M
    N = len(datas)
    self.N = N

    # initialize var params as dictionaries
    tilda_log_pi0s = {}
    tilda_log_Ps = {}
    tilda_etas = {}
    tilda_log_sigmasqs = {}

    # variational model parameters
    if random_state is None:

      for tag, data in zip(tags, datas):
        T = data.shape[0]

        # auto regressive coefs
        tilda_etas[tag] = np.random.normal(0.95, np.sqrt(0.1), (K, M+1))
        tilda_log_sigmasqs[tag] = -5 + npr.randn(K, M+1) * 0.05

        # transition probabilities
        tilda_log_pi0s[tag] = -np.log(K) * np.ones((K, 1))
        Ps = 0.95 * np.eye(K) + 0.05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        tilda_log_Ps[tag] = np.log(Ps)

    else:

      for tag, data in zip(tags, datas):
        T = data.shape[0]

        # auto regressive coefs
        rs = npr.RandomState(random_state)
        tilda_etas[tag] = rs.normal(0.95, np.sqrt(0.1), (K, M+1))
        rs = npr.RandomState(random_state)
        tilda_log_sigmasqs[tag] = -5 + rs.randn(K, M+1) * 0.05

        # transition probabilities
        tilda_log_pi0s[tag] = -np.log(K) * np.ones((K, 1))
        rs = npr.RandomState(random_state)
        Ps = 0.95 * np.eye(K) + 0.05 * rs.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        tilda_log_Ps[tag] = np.log(Ps)

    self.tilda_log_pi0s = tilda_log_pi0s.copy()
    self.tilda_log_Ps = tilda_log_Ps.copy()
    self.tilda_etas = tilda_etas.copy()
    self.tilda_log_sigmasqs = tilda_log_sigmasqs.copy()
    
    return

  @ensure_args_are_lists
  def initialize_marginals_joints(self, 
                                  datas, 
                                  inputs=None, 
                                  masks=None, 
                                  tags=None):
    """
    Function to compute the joint and mariginal posterior probabilities
    of being in a state given the observation. We use forward-backward
    algorithm to compute these probabilities.

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace

    """
    # initilaize variables of interest
    pi0 = np.exp(self.log_pi0).ravel().copy()
    Ps = np.expand_dims(np.exp(self.log_Ps), axis=0).copy()

    # compute the partial joint log-likelihood
    lls = self.compute_partial_loglikelihoods(datas, inputs, masks, tags)

    # get marginals and joints using forward-backward algo
    self.states = {}
    self.joints = {}
    for tag in tags:
      ll = lls[tag]
      exp_states, exp_joints, _ = hmm_expected_states(pi0, Ps, ll)
      exp_states = exp_states / exp_states.sum(axis=1, keepdims=True)
      exp_joints = exp_joints / exp_joints.sum(axis=(0,2), keepdims=True)
      self.states[tag] = exp_states.copy()
      self.joints[tag] = exp_joints.copy()

    return

  @ensure_args_are_lists
  def fit_lm(self,
             datas, 
             inputs=None,
             masks=None,
             tags=None,
             random_state=None,
             win=3):
    """Piecewise linear model fit

    Args:
        datas ([list of 1D arrays]): Contains body weight traces
        inputs ([list]): Defaults to None.
        masks ([list]): Defaults to None.
        tags ([list]): Tag assigned to each trace. Defaults to None.
        random_state ([int]): Random seed value. Defaults to None.

    Returns:
        slopes (ndarray): slopes across all traces
    """

    slopes = []
    for data in datas:
        
        T = data.shape[0]
        slope = []
        for t in np.arange(0,T-win):
            X = np.vstack((np.ones((win,)), np.arange(t, t+win))).T
            y = data[t:t+win, [0]]
            reg = LinearRegression().fit(X, y)
            slope.append(reg.coef_[0][1])
        slope = np.asarray(slope)
        if len(slope) != 0:
          slopes.append(slope)
    
    slopes = np.hstack(slopes)
    return slopes

  def fit_kmeans(self,
                 slopes,
                 clusters=5,
                 random_state=0):
    """[summary]

    Args:
        slopes (ndarray): Slopes of all clusters
        clusters (int): Number of clusters. Defaults to 5.
        random_state (int): Random state for kmeans.  Defaults to 0.

    Returns:
        ar0 (ndarray): Contains cluster centers of slopes
        sigmasqs (ndarray): Contains variance of each cluster
    """
    
    values = slopes.reshape(-1, 1)
    kmeans = KMeans(n_clusters=clusters, random_state=2).fit(values)
    ar0 = kmeans.cluster_centers_.ravel()
    labels = kmeans.labels_
    sigmasqs = np.zeros((len(ar0), ))
    for i, label in enumerate(np.unique(labels)):
      sel_slopes = values[labels == label]
      sigmasqs[i] = (1/len(sel_slopes)) * np.std(sel_slopes)**2

    return (ar0, sigmasqs)

  @ensure_args_are_lists
  def data_driven_param_initialization(self,
                                      datas,
                                      inputs=None,
                                      masks=None,
                                      tags=None,
                                      random_state=None):

    K, M, N = self.K, self.M, self.N
    bar_err_sigmasq = []

    for data, tag in zip(datas, tags):
      T = data.shape[0]
      y = data[M:]
      X = np.concatenate((np.ones((T-M, 1)), data[:-M]), axis=1)
      phi = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
      y_hat = np.dot(X, phi)
      err_sigmasq = (1/T) * np.linalg.norm(y - y_hat)**2
      bar_err_sigmasq.append(err_sigmasq)

    # initialize error sigma squares
    log_err_sigmasq = (1/N) * np.log(np.percentile(np.array(bar_err_sigmasq), 90))
    if random_state is None:
      self.log_err_sigmasqs = npr.normal(log_err_sigmasq, 0.01, (K,1))
    else:
      rs = npr.RandomState(random_state)
      self.log_err_sigmasqs = rs.normal(log_err_sigmasq, 0.01, (K,1))

    # initialize etas and log_sigmasqs
    slopes = self.fit_lm(datas)
    ar0, ar0_sigmasq = self.fit_kmeans(slopes, clusters=K, random_state=random_state)
    self.etas[:, 0] = ar0
    self.etas[:, 1] = np.ones((K, ))
    self.log_sigmasqs[:, 0] = np.log(ar0_sigmasq)
    self.log_sigmasqs[:, 1] = (1/K) * np.log(ar0_sigmasq)

    # initialize tilda etas and tilda_log_sigmasqs
    if random_state is None:
      for tag in tags:
        self.etas[tag][:, 0] = np.asarray([npr.normal(ar0[i], np.sqrt(ar0_sigmasq[i]), (1,)) for i in range(K)]).ravel()
        self.etas[tag][:, 1] = np.ones((K, ))
        self.log_sigmasqs[tag][:, 0] = np.log(ar0_sigmasq)
        self.log_sigmasqs[tag][:, 1] = (1/K)*np.log(ar0_sigmasq)
    else:
      for tag in tags:
        rs = npr.RandomState(random_state)
        self.tilda_etas[tag][:, 0] = np.asarray([rs.normal(ar0[i], np.sqrt(ar0_sigmasq[i]), (1,)) for i in range(K)]).ravel()
        self.tilda_etas[tag][:, 1] = np.ones((K, ))
        self.tilda_log_sigmasqs[tag][:, 0] = np.log(ar0_sigmasq)
        self.tilda_log_sigmasqs[tag][:, 1] = (1/K)*np.log(ar0_sigmasq)

    return

  @ensure_args_are_lists
  def update_variational_params(self, 
                                datas, 
                                inputs=None, 
                                masks=None, 
                                tags=None, 
                                random_state=None):
    """[summary]

    Args:
        datas ([list of 1D arrays]): Contains body weight traces
        inputs ([list]): Defaults to None.
        masks ([list]): Defaults to None.
        tags ([list]): Tag assigned to each trace. Defaults to None.
        random_state ([int]): Random seed value. Defaults to None.
    """
    K, M = self.K, self.M
    ar0 = self.etas[:, 0]
    ar0_sigmasq = np.exp(self.log_sigmasqs[:, 0])
    for tag in tags:
      if random_state is None:
        self.tilda_etas[tag][:, 0] = np.asarray([npr.normal(ar0[i], np.sqrt(ar0_sigmasq[i]), (1,)) for i in range(K)]).ravel()
        self.tilda_etas[tag][:, 1] = np.ones((K, ))
        self.tilda_log_sigmasqs[tag][:, 0] = np.log(ar0_sigmasq)
        self.tilda_log_sigmasqs[tag][:, 1] = (1/K)*np.log(ar0_sigmasq)
      else:
        rs = npr.RandomState(random_state)
        self.tilda_etas[tag][:, 0] = np.asarray([rs.normal(ar0[i], np.sqrt(ar0_sigmasq[i]), (1,)) for i in range(K)]).ravel()
        self.tilda_etas[tag][:, 1] = np.ones((K, ))
        self.tilda_log_sigmasqs[tag][:, 0] = np.log(ar0_sigmasq)
        self.tilda_log_sigmasqs[tag][:, 1] = (1/K)*np.log(ar0_sigmasq)

    return

  @ensure_args_are_lists
  def initialize(self,
                datas,
                inputs=None,
                masks=None,
                tags=None):
    """
    Function that calls the initialize parameter functions.

    Args:
        datas ([list of 1D arrays]): Contains body weight traces
        inputs ([list]): Defaults to None.
        masks ([list]): Defaults to None.
        tags ([list]): Tag assigned to each trace. Defaults to None.
    """
    self.initalize_variational_params(datas, inputs, masks, tags, \
      random_state=self.random_state)
    self.data_driven_param_initialization(datas, inputs, masks, tags, \
      random_state=self.random_state)
    self.initialize_marginals_joints(datas, inputs, masks, tags)
    return

  @property
  def params(self):
    """[summary]

    Returns:
        [type]: [description]
    """
    return (self.log_pi0, 
            self.log_Ps,
            self.etas, 
            self.log_sigmasqs, 
            self.log_err_sigmasqs)
  
  @params.setter
  def params(self, value):
    self.log_pi0, self.log_Ps, self.etas, \
      self.log_sigmasqs, self.log_err_sigmasqs = value
    return

  @property
  def var_params(self):
    return (self.tilda_log_pi0s,
            self.tilda_log_Ps,
            self.tilda_etas, 
            self.tilda_log_sigmasqs,
            self.states,
            self.joints)

  @var_params.setter
  def var_params(self, value):
    self.tilda_log_pi0s, self.tilda_log_Ps, \
      self.tilda_etas, self.tilda_log_sigmasqs, \
        self.states, self.joints = value


  @ensure_args_are_lists
  def compute_partial_loglikelihoods(self, 
                                    datas, 
                                    inputs=None, 
                                    masks=None, 
                                    tags=None):
    """
    Function to compute the loglikelihood after taking the expectation
    with respect to the priors, i.e., q(\theta). We can it partial
    loglikelihood because the expectation is with respect to q(\theta)
    and not q(\theta, s_{1:T})

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace

    Returns:
    -------
      lls (list of nd array): contains partial loglikelihood, where
      the shape of each array is T x K, where T is the number of
      time stamps and K is the number of states.

    """
    lls = {}
    for data, input, mask, tag in zip(datas, masks, inputs, tags):
      lls[tag] = self._compute_partial_loglikelihoods(data, input, mask, tag)

    return lls


  @ensure_args_not_none
  def _compute_partial_loglikelihoods(self,
                                      data,
                                      input=None,
                                      mask=None,
                                      tag=None):
    """
    Function to compute the loglikelihood after takine the expectation
    with respect to the priors, i.e., q(theta). We can it partial
    loglikelihood because the expectation is with respect to q(\theta)
    and not q(theta, s_{1:T})

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    Returns:
    -------
      ll (nd array): T x K matrix with the partial loglikelihoods 
      of the data. If the lag-order is M, then the loglikelihood
      until the lag-order is zero because we assume that the data
      until the lag-order is not part of the generative process.

    """
    K, M = self.K, self.M
    T = data.shape[0]
    ll = np.zeros((T, K))

    # variational parameters for the given trace
    tilda_eta = self.tilda_etas[tag].copy()
    tilda_sigmasq = np.exp(self.tilda_log_sigmasqs[tag]).copy()

    for t in np.arange(M, T):
      tmp = - 0.5 * self.log_err_sigmasqs \
          - 0.5 * np.multiply((1 / np.exp(self.log_err_sigmasqs)), \
          (data[t]**2 - (2 * data[t]*tilda_eta[:, [0]]) + \
          (tilda_eta[:, [0]]**2 + tilda_sigmasq[:, [0]]) - \
          (2 * data[t] * sum(data[t-l] * tilda_eta[:, [l]] \
            for l in np.arange(1, M+1))) + \
          (sum(data[t-l]**2 * (tilda_eta[:, [l]]**2 + tilda_sigmasq[:, [l]]) \
            for l in np.arange(1, M+1))) + \
          (2 * tilda_eta[:, [0]] * sum(data[t-l] * tilda_eta[:, [l]] \
            for l in np.arange(1, M+1))) + \
          (2 * sum(data[t-p] * data[t-q] * tilda_eta[:, [p]] * tilda_eta[:, [q]] \
            for p in np.arange(1, M+1) for q in np.arange(1, p)))))
      ll[t, :] = tmp.ravel()

    return ll


  @ensure_args_are_lists
  def compute_loglikelihoods(self, 
                             datas, 
                             inputs=None, 
                             masks=None, 
                             tags=None):
    """
    Function to compute the loglikelihood after taking the expectation
    with respect to the priors, i.e., p(\theta). 

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace

    Returns:
    -------
      lls (list of nd array): contains partial loglikelihood, where
      the shape of each array is T x K, where T is the number of
      time stamps and K is the number of states.

    """
    lls = {}
    for data, input, mask, tag in zip(datas, masks, inputs, tags):
      lls[tag] = self._compute_loglikelihoods(data, input, mask, tag)

    return lls


  @ensure_args_not_none
  def _compute_loglikelihoods(self,
                              data,
                              input=None,
                              mask=None,
                              tag=None):
    """
    Function to compute the loglikelihood after taking the expectation
    with respect to the priors, i.e., p(theta).

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    Returns:
    -------
      ll (nd array): T x K matrix with the partial loglikelihoods 
      of the data. If the lag-order is M, then the loglikelihood
      until the lag-order is zero because we assume that the data
      until the lag-order is not part of the generative process.

    """
    K, M = self.K, self.M
    T = data.shape[0]
    ll = np.zeros((T, K))

    # variational parameters for the given trace
    eta = self.etas.copy()
    sigmasq = np.exp(self.log_sigmasqs).copy()

    for t in np.arange(M, T):
      tmp = - 0.5 * self.log_err_sigmasqs \
          - 0.5 * np.multiply((1 / np.exp(self.log_err_sigmasqs)), \
          (data[t]**2 - (2 * data[t]*eta[:, [0]]) + \
          (eta[:, [0]]**2 + sigmasq[:, [0]]) - \
          (2 * data[t] * sum(data[t-l] * eta[:, [l]] \
            for l in np.arange(1, M+1))) + \
          (sum(data[t-l]**2 * (eta[:, [l]]**2 + sigmasq[:, [l]]) \
            for l in np.arange(1, M+1))) + \
          (2 * eta[:, [0]] * sum(data[t-l] * eta[:, [l]] \
            for l in np.arange(1, M+1))) + \
          (2 * sum(data[t-p] * data[t-q] * eta[:, [p]] * eta[:, [q]] \
            for p in np.arange(1, M+1) for q in np.arange(1, p)))))
      ll[t, :] = tmp.ravel()

    return ll

  @ensure_args_are_lists
  def compute_elbos(self, 
                    datas, 
                    inputs=None, 
                    masks=None, 
                    tags=None):
    """
    Function to compute the evidence-based lower bound (ELBO)
    of all traces together

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace
    
    Returns:
    -------
      elbos (float): elbo in nats

    """
    elbos = 0
    for data, input, mask, tag in zip(datas, inputs, masks, tags):
      elbos += self._compute_elbo(data, input, mask, tag)
    return elbos

  @ensure_args_not_none
  def _compute_elbo(self, 
                    data,
                    input=None,
                    mask=None,
                    tag=None):
    """
    Function to compute the evidence-based lower bound (ELBO)
    of a single trace

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    Returns:
    -------
      elbo (float): elbo value of the trace in nats

    """

    K, M = self.K, self.M
    T = data.shape[0]
    elbo = 0

    # initial and transition probs
    log_pi0 = self.log_pi0.copy()
    log_Ps = self.log_Ps.copy()

    # joints and marginas
    expected_states = self.states[tag].copy()
    expected_joints = self.joints[tag].copy()
    expected_states = expected_states / expected_states.sum(axis=1, keepdims=True)
    expected_joints = expected_joints[0, :, :] / expected_joints.sum(axis=(0,2), keepdims=True)
    expected_joints = np.squeeze(expected_joints, axis=0)

    # variational params
    tilda_eta = self.tilda_etas[tag].copy()
    tilda_sigmasq = np.exp(self.tilda_log_sigmasqs[tag])
    tilda_pi0 = np.exp(self.tilda_log_pi0s[tag] \
      - logsumexp(self.tilda_log_pi0s[tag]))
    tilda_log_Ps = self.tilda_log_Ps[tag].copy()
    
    # elbo from initial prob
    elbo += (np.dot(tilda_pi0.T, log_pi0) \
      - np.dot(tilda_pi0.T, self.tilda_log_pi0s[tag]))
    assert np.isfinite(elbo)

    # elbo from transition matrix prob
    for t in np.arange(M, T):
      elbo += np.dot(expected_states[[t], :], \
        np.multiply(expected_joints, log_Ps)).sum() - \
           np.dot(expected_states[[t], :], \
             np.multiply(expected_joints, tilda_log_Ps)).sum()
    assert np.isfinite(elbo)

    # compute partial log-likelihood
    ll = self._compute_partial_loglikelihoods(data, tag=tag)

    # elbo from partial log-likelihood
    elbo += (np.multiply(ll[M:T, :], expected_states[M:T, :])).sum()
    assert np.isfinite(elbo)

    # elbo from priors
    elbo += 0.5 * (1 + sum(sum(np.log(tilda_sigmasq + LOG_EPS) - self.log_sigmasqs \
      - ((tilda_sigmasq + (tilda_eta - self.etas)**2) / np.exp(self.log_sigmasqs)))))
    assert np.isfinite(elbo)

    return elbo.ravel()

  @ensure_args_are_lists
  def variational_em_step(self,
                          datas,
                          inputs=None,
                          masks=None,
                          tags=None):
    """
    Function that calls the variational e-step and m-step

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace

    """
    self.variational_e_step(datas, inputs, masks, tags)
    self.variational_m_step(datas, inputs, masks, tags)

    return

  @ensure_args_are_lists
  def variational_e_step(self,
                         datas,
                         inputs=None,
                         masks=None,
                         tags=None):

    """
    Function to compute variational e-step. In this step, compute
    the marginals and joints using forward-backward algorithm

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace

    """

    pi0 = np.exp(self.log_pi0).ravel().copy()
    Ps = np.expand_dims(np.exp(self.log_Ps), axis=0).copy()

    for data, input, mask, tag in zip(datas, inputs, masks, tags):

      ll = self._compute_partial_loglikelihoods(data, input, mask, tag=tag)
      exp_states, exp_joints, _ = hmm_expected_states(pi0, Ps, ll)
      exp_states = exp_states / exp_states.sum(axis=1, keepdims=True)
      exp_joints = exp_joints / exp_joints.sum(axis=(0,2), keepdims=True)
      
      # # exp_states = np.maximum(exp_states, 1e-10)
      # exp_states = np.where(exp_states < 1e-10, 0, exp_states)
      # exp_states = exp_states / exp_states.sum(axis=1, keepdims=True)
      # exp_joints = np.maximum(exp_joints, 1e-10)
      # exp_joints = exp_joints[0, :, :] / exp_joints.sum(axis=(0,2), keepdims=True)
      
      self.states[tag] = exp_states.copy()
      self.joints[tag] = exp_joints.copy()

    return

  @ensure_args_are_lists
  def variational_m_step(self,
                         datas,
                         inputs=None,
                         masks=None,
                         tags=None):
    """
    Function to compute variational m-step. We estimate various variational
    parameters using the closed-form expressions.

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace
      
    """

    for data, input, mask, tag in zip(datas, inputs, masks, tags):
      self.update_variational_initial_probability(data, input, mask, tag)
      self.update_variational_transition_matrix(data, input, mask, tag)
      self.update_tilda_sigmasq(data, input, mask, tag)
      self.update_tilda_eta(data, input, mask, tag)

    return

  @ensure_args_not_none
  def update_variational_initial_probability(self,
                                            data,
                                            input=None,
                                            mask=None,
                                            tag=None):
    """
    Function to the maximum likelihood estimate of the 
    variational initial probability. Note that this term
    is independnet of the data.

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    """

    pi0 = np.exp(self.log_pi0).copy()
    tilda_pi0 = pi0 / pi0.sum()
    tilda_log_pi0 = np.log(tilda_pi0 + LOG_EPS)
    tilda_log_pi0 = tilda_log_pi0 - logsumexp(tilda_log_pi0)
    self.tilda_log_pi0s[tag] = tilda_log_pi0.copy()

    return

  @ensure_args_not_none
  def update_variational_transition_matrix(self,
                                          data,
                                          input=None,
                                          mask=None,
                                          tag=None):
    """
    Function to the maximum likelihood estimate of the 
    variational transition matrix.

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    """
    K, M = self.K, self.M
    T = data.shape[0]

    exp_joints = np.squeeze(self.joints[tag].copy(), axis=0)
    exp_states = self.states[tag].copy()
    tilda_Ps = np.divide((T-M)*exp_joints, \
      np.sum(exp_states[M:T, :], axis= 0, keepdims=True).T + DIV_EPS) 
    tilda_log_Ps = np.log(tilda_Ps + LOG_EPS)
    self.tilda_log_Ps[tag] = (tilda_log_Ps - \
      logsumexp(tilda_log_Ps, axis=1, keepdims=True))

    # exp_joints = self.joints[tag].copy()
    # exp_joints = exp_joints / exp_joints.sum(axis=(0,2), keepdims=True)
    # exp_joints = np.squeeze(exp_joints, axis=0)
    # tilda_log_Ps = np.log(exp_joints + LOG_EPS)
    # self.tilda_log_Ps[tag] = (tilda_log_Ps - \
    #   logsumexp(tilda_log_Ps, axis=1, keepdims=True))
    return

  @ensure_args_not_none
  def update_tilda_sigmasq(self,
                          data,
                          input=None,
                          mask=None,
                          tag=None):
    """
    Function to the maximum likelihood estimate of the 
    variational tilda sigma squares parameter.

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    """
    K, M = self.K, self.M
    T = data.shape[0]
    inv_tilda_sigmasq = np.zeros((K, M+1))

    weights = self.states[tag].copy()
    err_sigmasq = np.exp(self.log_err_sigmasqs).ravel()

    for q in range(M+1):
      tilda_sigmasq = np.exp(self.log_sigmasqs[:, q]).ravel()
      if q == 0:
        # tmp_weights = np.maximum(weights, 1e-10)
        # tmp_weights = tmp_weights / tmp_weights.sum(axis=1, keepdims=True)
        sigmasq_second = weights[M:T, :].sum(axis=0)
      else:
        sigmasq_second = sum(weights[t, :] * data[t-q]**2 \
          for t in np.arange(M, T))
      inv_tilda_sigmasq[:, q] = (1 / tilda_sigmasq) + \
        (sigmasq_second / err_sigmasq)

    assert np.isfinite(inv_tilda_sigmasq).all()
    self.tilda_log_sigmasqs[tag] = -np.log(inv_tilda_sigmasq + LOG_EPS).copy()

    return

  @ensure_args_not_none
  def update_tilda_eta(self,
                      data,
                      input=None,
                      mask=None,
                      tag=None):
    """
    Function to find the maximum likelihood estimate of the 
    variational tilda eta parameter.

    Args:
    ----
      data (1D array): contains body weight traces in
      the form of a column vector
      input (list): NA
      mask (bool): NA
      tag (int): unique ID assigned to each body weight trace

    """
    K, M = self.K, self.M
    T = data.shape[0]
    tilda_eta = np.zeros((K, M+1))

    weights = self.states[tag].copy()
    tilda_sigmasq = np.exp(self.tilda_log_sigmasqs[tag])

    for q in range(M+1):
      if q == 0:
        # tmp_weights = np.maximum(weights, 1e-10)
        # tmp_weights = tmp_weights / tmp_weights.sum(axis=1, keepdims=True)
        partial_eta_num_second = sum(weights[[t], :].T * (data[t] \
          - sum(self.tilda_etas[tag][:, [l]] * data[t-l] \
            for l in np.arange(1, M+1))) \
              for t in np.arange(M, T)).ravel()
      else:
        partial_eta_num_second = sum(weights[[t], :].T * data[t-q] * \
          (data[t] - self.tilda_etas[tag][:, [0]] - \
            0.5 * sum(self.tilda_etas[tag][:, [l]] * data[t-l] \
              for l in np.arange(1, M+1) if l != q)) \
                for t in np.arange(M, T)).ravel()

      eta_num_first = self.etas[:, q] / np.exp(self.log_sigmasqs[:, q])
      assert np.isfinite(eta_num_first).all()

      eta_num_second = partial_eta_num_second / np.exp(self.log_err_sigmasqs.ravel())
      assert np.isfinite(eta_num_second).all()

      tilda_eta[:, q] = (eta_num_first + eta_num_second) * tilda_sigmasq[:, q]
      if q == 1:
        tilda_eta[:, q] = 1
      self.tilda_etas[tag][:, q] = tilda_eta[:, q].copy()
    
    assert np.isfinite(tilda_eta).all()

    return
  
  @ensure_args_are_lists
  def m_step(self,
            datas,
            inputs=None,
            masks=None,
            tags=None,
            **kwargs):
    """
    Function to compute m-step. We estimate various model parameters
    using the closed-form expressions.

    Args:
    ----
      datas (list of 1D array): contains body weight traces in
      the form of a column vector
      inputs (list): NA
      masks (list): NA
      tags (list): unique ID assigned to each body weight trace
      
    """
    
    K, M, N = self.K, self.M, self.N
    max_iter = kwargs['num_iters']
    iter = kwargs['iter']

    err_sigmasqs_num = np.zeros((K, 1))
    err_sigmasqs_den = np.zeros((K, 1))
    etas = np.zeros((K, M+1))
    sigmasqs = np.zeros((K, M+1))
    pi0 = np.zeros((K, 1))
    Ps = np.zeros((K, K))

    for data, input, mask, tag in zip(datas, inputs, masks, tags):

      T = data.shape[0]
      weights = self.states[tag]

      # initial probability
      # pi0[:, 0] += np.exp(self.tilda_log_pi0s[tag]).ravel()
      pi0[:, 0] += weights[M, :]
      assert np.isfinite(pi0).all()

      # transition probability
      tilda_Ps = np.exp(self.tilda_log_Ps[tag])
      for i in range(K):
        for j in range(K):
          for t in range(M, T):
            Ps[i, j] += tilda_Ps[i, j] * weights[t, i]
      
      tilda_eta = self.tilda_etas[tag]
      tilda_sigmasq = np.exp(self.tilda_log_sigmasqs[tag])

      psi = np.column_stack((data[t]**2 - (2 * data[t] * tilda_eta[:, [0]]) \
        + (tilda_eta[:, [0]]**2 + tilda_sigmasq[:, [0]]) \
        - (2 * data[t] * sum(data[t-l] * tilda_eta[:, [l]] \
          for l in np.arange(1, M+1)))
        + (sum(data[t-l]**2 * (tilda_eta[:, [l]]**2 + tilda_sigmasq[:, [l]]) \
          for l in np.arange(1, M+1))) \
        + (2 * tilda_eta[:, [0]] * sum(data[t-l] * tilda_eta[:, [l]] \
          for l in np.arange(1, M+1))) \
        + (2 * sum(data[t-p] * data[t-q] * tilda_eta[:, [p]] * tilda_eta[:, [q]] \
          for p in np.arange(1, M+1) for q in np.arange(1, p)))) \
            for t in np.arange(M, T))

      assert np.isfinite(psi).all()
      err_sigmasqs_num[:, 0] += np.sum((psi * weights[M:T, :].T), axis=1)
      err_sigmasqs_den[:, 0] += np.sum(weights[M:T, :].T, axis=1)

      etas += tilda_eta
      sigmasqs += tilda_sigmasq + (tilda_eta - self.etas)**2
      
      assert np.isfinite(etas).all()
      assert np.isfinite(sigmasqs).all()

    # update initial probs
    pi0 += LOG_EPS
    self.log_pi0 = np.log(pi0 / pi0.sum()).copy()

    # update transition matrix probs
    Ps /= Ps.sum(axis=1, keepdims=True) + DIV_EPS
    log_P = np.log(Ps + LOG_EPS)
    self.log_Ps = (log_P - logsumexp(log_P, axis=1, keepdims=True)).copy()

    # update emission matrix params
    self.etas = etas / N
    self.etas[:, 1] = 1

    # if not((iter > max_iter // 8) and (iter < max_iter // 4)):
    log_sigmasqs = np.log((sigmasqs / N) + LOG_EPS)
    # log_sigmasqs = np.maximum(log_sigmasqs, np.log(1e-2))
    self.log_sigmasqs = log_sigmasqs.copy()
    # self.log_sigmasqs = np.log((sigmasqs / N) + LOG_EPS)

    # update error variances
    if (iter > max_iter // 8):
      log_err_sigmasqs = np.log(err_sigmasqs_num + LOG_EPS) \
        - np.log(err_sigmasqs_den + LOG_EPS)

      # if the errors in sigmasqs are bounded, then
      if self.err_sigmasq_threshold:
        print(self.err_sigmasq_threshold)
        log_err_sigmasqs = np.minimum(log_err_sigmasqs, \
          np.log(self.err_sigmasq_threshold))

      self.log_err_sigmasqs = log_err_sigmasqs.copy()
      assert np.isfinite(self.log_err_sigmasqs).all()

    return