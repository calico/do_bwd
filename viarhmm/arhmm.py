from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
from warnings import warn
from functools import partial, total_ordering
from scipy.sparse.construct import random
from tqdm.auto import trange


import autograd.numpy as np
import autograd.numpy.random as npr

from viarhmm.utils import ensure_args_are_lists
from viarhmm.utils import ensure_args_not_none
from viarhmm.messages import hmm_normalizer
from viarhmm.utils import ssm_pbar
from autograd.scipy.special import logsumexp
from viarhmm.observations import VarInfARDiagGaussianObservations

class VIARHMM(object):
  def __init__(self, K, M,
              observations='varinf_ar_diagonal_gaussian',
              observation_kwargs=None,
              random_state=None,
              **kwargs):
    """
    Base class for the variational inference implementation of
    Markov switching auto-regressive model.

    Args:
    ----
      K: number of discrete latent states
      M: lag order
    """

    # intialize    
    self.K, self.M = K, M

    # initalize elbos to zeros
    self.elbos = []

    # create an observation class object
    observation_classes = dict(
        varinf_ar_diagonal_gaussian=VarInfARDiagGaussianObservations,
        )

    if isinstance(observations, str):
      observations = observations.lower()
      if observations not in observation_classes:
        raise Exception("Invalid observation model: {}. Must be one of {}".
          format(observations, list(observation_classes.keys())))

      observation_kwargs = observation_kwargs or {}
      observations = observation_classes[observations](K, M, random_state=random_state, **observation_kwargs)
      self.observations = observations

    return

  @property
  def params(self):
    return self.observations.params
  
  @params.setter
  def params(self, value):
    self.observations.params = value

  @property
  def var_params(self):
    return self.observations.var_params

  @var_params.setter
  def var_params(self, value):
    self.observations.var_params = value

  @ensure_args_are_lists
  def initialize(self, 
                datas, 
                inputs=None, 
                masks=None, 
                tags=None):
    """
    Initialize parameters given data.
    """
    self.observations.initialize(datas, inputs, masks, tags)
    return

  @ensure_args_are_lists
  def log_likelihood(self, 
                    datas, 
                    inputs=None, 
                    masks=None, 
                    tags=None):
    """
    Compute the log probability of the data under the current
    model parameters.

    :param datas: single array or list of arrays of data.
    :return total log probability of the data.
    """
    elbos = self.observations.compute_elbos(datas, inputs, 
      masks, tags)
    return elbos

  def _fit_var_em(self, 
                  datas, 
                  inputs, 
                  masks, 
                  tags, 
                  verbose=2, 
                  num_iters=2000, 
                  tolerance=1e-4,
                  observations_mstep_kwargs={},
                  observations_varestep_kwargs = {},
                  **kwargs):

    elbos = []
    elbo  = self.log_likelihood(datas, inputs, masks, tags)
    pbar = ssm_pbar(num_iters, verbose, "ELBO: {:.10f}", elbo)
    elbos.append(elbo[0])
    old_elbo = elbo
    
    for i, itr in zip(range(num_iters), pbar):

      # Variation EM step
      self.observations.variational_em_step(datas, inputs, masks, tags)

      # M-step
      observations_mstep_kwargs = {'num_iters': num_iters, 'iter': i}
      self.observations.m_step(datas, inputs, masks, tags, **observations_mstep_kwargs)

      # Store progress
      if verbose == 2:
        elbo  = self.log_likelihood(datas, inputs, masks, tags)
        elbos.append(elbo[0])
        pbar.set_description("ELBO: {:.10f}".format(elbo[0]))
        
        # Check for convergence
        if (i > num_iters // 4) and (np.abs(elbo-old_elbo) < tolerance):
          break
        else:
          old_elbo = elbo

    return elbos

  @ensure_args_are_lists
  def fit(self, 
          datas, 
          inputs=None, 
          masks=None, 
          tags=None,
          verbose=2, 
          method="em",
          initialize=True,
          random_state=None,
          **kwargs):

    _fitting_methods = \
      dict(
          em=self._fit_var_em,
          )

    if method not in _fitting_methods:
      raise Exception("Invalid method: {}. Options are {}".
                      format(method, _fitting_methods.keys()))

    if initialize:
      self.initialize(datas,
                      inputs=inputs,
                      masks=masks,
                      tags=tags)

    return _fitting_methods[method](datas,
                                    inputs=inputs,
                                    masks=masks,
                                    tags=tags,
                                    verbose=verbose,
                                    **kwargs)

  def _decode_var_em(self, 
                    datas, 
                    inputs, 
                    masks, 
                    tags, 
                    verbose=2, 
                    num_iters=2000, 
                    tolerance=1e-3,
                    observations_mstep_kwargs={},
                    observations_varestep_kwargs = {},
                    **kwargs):

    elbos = []
    elbo  = self.log_likelihood(datas, inputs, masks, tags)
    pbar = ssm_pbar(num_iters, verbose, "ELBO: {:.10f}", elbo)
    elbos.append(elbo[0])
    old_elbo = elbo

    for i, itr in zip(range(num_iters), pbar):
      self.observations.variational_em_step(datas, inputs, masks, tags)

      # Store progress
      if verbose == 2:
        elbo  = self.log_likelihood(datas, inputs, masks, tags)
        elbos.append(elbo[0])
        pbar.set_description("ELBO: {:.10f}".format(elbo[0]))

        if (i > num_iters // 4) and (np.abs(elbo-old_elbo) < tolerance):
          break
        else:
          old_elbo = elbo

    return elbos

  @ensure_args_are_lists
  def predict(self,
              datas, 
              inputs=None, 
              masks=None, 
              tags=None, 
              verbose=2, 
              method="em",
              initialize=True,
              random_state=None,
              **kwargs):

    _decoding_methods = \
      dict(
          em=self._decode_var_em,
          )

    if initialize:
      self.observations.initalize_variational_params(datas, inputs, masks, tags, \
        random_state=random_state)
      self.observations.update_variational_params(datas, inputs, masks, tags,
        random_state=random_state)
      self.observations.initialize_marginals_joints(datas, inputs, masks, tags)

    return _decoding_methods[method](datas,
                                    inputs=inputs,
                                    masks=masks,
                                    tags=tags,
                                    verbose=verbose,
                                    **kwargs)

