from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
import pandas as pd
import autograd.numpy as np
import autograd.numpy.random as npr

from sklearn.model_selection import train_test_split

from viarhmm.arhmm import VIARHMM
from viarhmm.messages import hmm_normalizer
from viarhmm.settings import SEED, TRAIN_FRACTION, LEN_THRESHOLD
from viarhmm.settings import L1_TREND, TRIM_PRE, MIN_LEN

def str_to_bool(value):
  """
  Convert string to boolean

  Args:
      value (str): string value
  
  Returns:
      bool: boolean value
  """

  if value.lower() in {'false', 'f', '0', 'no', 'n'}:
    return False
  elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
    return True
  raise ValueError(f'{value} is not a valid boolean value')


def load_data(db_name, 
              l1_trend=L1_TREND, 
              min_len=MIN_LEN,
              trim_pre=TRIM_PRE):
  """
  Load data from pickle file

  Args:
      db_name (str): path to pickle file
      l1_trend (bool, optional): If True, load l1-trend data. Defaults to L1_TREND.
      min_len (int, optional): Min length of each bw trace. Defaults to MIN_LEN.
      trim_pre (bool, optional): If True, remove pre-intervention bw traces. Defaults to TRIM_PRE.

  Returns:
      df_raw (pd.DataFrame): Raw data
      df_l1f (pd.DataFrame): L1-trend data
      df_trim (pd.DataFrame): Trimmed data
  """

  print("Extracting raw measurements from %s"%(db_name))
  with open(db_name, 'rb') as f:
    bw_data = pickle.load(f)
    df = pd.DataFrame(bw_data, columns=['mouse_id', 'age_in_days',
                                        'bodyweight'])
    f.close()
    
  df_raw = apply_uniform_sampling(df)
  df_raw = drop_traces_based_on_max_delay(df_raw, min_len=min_len)
  if trim_pre:
    df_trim = remove_pre_intervention(df_raw)
    df_trim = drop_traces_based_on_max_delay(df_trim, min_len=3)
  else:
    df_trim = pd.DataFrame([])

  if l1_trend:
    db_name = db_name.replace("raw", "l1fit")
    print("Extracting l1-trend measurements from %s"%(db_name))
    with open(db_name, 'rb') as f:
      bw_data = pickle.load(f)
      df = convert_to_tall_df(bw_data)
      df_l1f = apply_uniform_sampling(df)
      df_raw = filter_df_raw(df_raw, df_l1f)
    return (df_raw, df_l1f, df_trim)

  return (df_raw, df_raw.copy(), df_trim)


def remove_pre_intervention(df_raw,
                            t0_days=30,
                            t1_days=180):
  df = df_raw.copy()
  cnames = ['bw.' + str(j) + '.days' for j in np.arange(t0_days, t1_days+1, 10)]
  df = df.drop(cnames, axis=1)
  mouse_ids = df.mouse_id.astype('str').tolist()

  df_tmp = df.drop(['mouse_id', 'diet'], axis=1)
  idx = df_tmp[df_tmp.isnull().all(axis=1)].index
  for i in idx:
    print("Removing: ", mouse_ids[i])
  df = df.drop(idx)
  df = df.reset_index(drop=True)

  return df


def drop_traces_based_on_max_delay(df_raw, 
                                   min_len=MIN_LEN):
  """Drop bw traces that are shorter than max_len

  Args:
      df_raw (pandas dataframe): input bw dataframe
      min_len (int, optional): Min length of each trace. Defaults to MIN_LEN.

  Returns:
      df_raw (pandas dataframe): with dropped traces
  """
  mouse_ids = df_raw.mouse_id.astype('str').tolist()

  df = df_raw.copy()
  df = df.drop(['diet', 'mouse_id'], axis=1)
  ind = []
  for i in range(df.shape[0]):
    raw_data = df.iloc[i].to_numpy()
    raw_data = raw_data[~np.isnan(raw_data)]
    if len(raw_data) < min_len:
      print("Removing: ", mouse_ids[i])
      ind.append(i)
  df_raw = df_raw.drop(df_raw.index[ind])
  return df_raw


def apply_uniform_sampling(df_input):
  """This function applies uniform sampling to the bw time-series
  data. The original data is sampled at uneven rate.

  Args:
      df_input (pandas dataframe): raw or l1-trend filtered

  Returns:
      df (pandas dataframe): evenly sampled time-series data
  """

  df_raw = df_input.copy()
  df_raw['diet'] = df_raw['mouse_id'].str.split("-", n=2, expand=True)[1]
  df_raw['age_in_months'] = df_raw['age_in_days'].astype('timedelta64[D]')//np.timedelta64(1, 'M')
  
  def approximate_age_in_days(x):
    if x % 10 < 5:
      return (x - x % 10)
    else:
      return (x + (10 - x % 10))
        
  df_raw['day'] = df_raw['age_in_days'].apply(approximate_age_in_days)
  min_day, max_day = df_raw.day.min(), df_raw.day.max()
  num_of_mice = len(df_raw.mouse_id.unique())
  
  data = np.empty((num_of_mice, int((max_day-min_day)//10) + 1))
  data[:] = np.nan
  df = pd.DataFrame(data, columns=['bw.' + str(x) + '.days' \
                                    for x in np.arange(min_day, max_day + 10, 10)])
  df['mouse_id'] = [''] * num_of_mice
  df['diet'] = [''] * num_of_mice
  
  df_mice = df_raw.groupby('mouse_id')
  mice_grouped = [group for _, group in df_mice]
  
  for i in range(len(mice_grouped)):
      
    df.mouse_id.iloc[i] = str(mice_grouped[i].mouse_id.iloc[0])
    df.diet.iloc[i] = str(mice_grouped[i].diet.iloc[0])
    
    for j in range(mice_grouped[i].shape[0]):
      day = mice_grouped[i].day.iloc[j]
      if np.isnan(df['bw.' + str(day) + '.days'].iloc[i]):
        df['bw.' + str(day) + '.days'].iloc[i] = mice_grouped[i].bodyweight.iloc[j]
      else:
        df['bw.' + str(day) + '.days'].iloc[i] = (mice_grouped[i].bodyweight.iloc[j] + \
          df['bw.' + str(day) + '.days'].iloc[i]) / 2
  return df

def convert_to_tall_df(raw_data):
  """Convert raw data from dictionary to dataframe

  Args:
      raw_data (dict): raw bw measurements in dict form

  Returns:
      df (pandas dataframe): dict -> pandas df conversion
  """
  dfs = []
  for key in raw_data.keys():
    df = pd.DataFrame([], columns=['mouse_id', 'age_in_days', 'bodyweight'])
    age_in_days, body_weight = raw_data[key]
    df['mouse_id'] = [key] * len(age_in_days)
    df['age_in_days'] = age_in_days
    df['bodyweight'] = body_weight
    dfs.append(df)
  df = pd.concat(dfs)
  return df

def filter_df_raw(df_raw_bw, 
                  df_l1f_bw):
  """Check if all traces are available in smoothend data

  Args:
      df_raw_bw (pandas dataframe): raw bw measurements
      df_l1f_bw (pandas dataframe): l1-fit bw measurements

  Returns:
      df_raw_bw (pandas dataframe): filtered raw bw measurements
  """
  mouse_id_raw = df_raw_bw.mouse_id.unique()
  mouse_id_l1f = df_l1f_bw.mouse_id.unique()
  for mouse_id in mouse_id_raw:
    if not(mouse_id in mouse_id_l1f):
      print("Removing: ", mouse_id)
      ind = df_raw_bw.index[df_raw_bw['mouse_id'] == mouse_id]
      df_raw_bw = df_raw_bw.drop(df_raw_bw.index[ind])
  return df_raw_bw

def split_train_validation(df_bw, 
                           len_threshold=LEN_THRESHOLD, 
                           train_fraction=TRAIN_FRACTION, 
                           random_state=SEED):
  """Split data into train and validation

  Args:
      df_bw ([type]): [description]
      len_threshold ([type], optional): [description]. Defaults to LEN_THRESHOLD.
      train_fraction ([type], optional): [description]. Defaults to TRAIN_FRACTION.
      random_state ([type], optional): [description]. Defaults to SEED.

  Returns:
      [type]: [description]
  """
    
  diet_groups = ['AL','20','40','1D','2D']
  
  df = df_bw.copy()
  df_diet = df.groupby('diet')
  diet_grouped = [group for _, group in df_diet]

  diets = []
  tdatas = []
  n_train = []
  vdatas = []
  n_validation = []
  ttags = []
  vtags = []
  
  for i in range(len(diet_grouped)):
      
    diet_type = str(diet_grouped[i].diet.iloc[0])
    diets.append(diet_type)

    data = diet_grouped[i].drop(['mouse_id', 'diet'], axis=1).to_numpy()
    diet_lengths = np.sum(~np.isnan(data), axis=1)
    diet_grouped[i]['lengths'] = diet_lengths
    mask = diet_grouped[i]['lengths'] >= len_threshold
    
    selected_df1 = diet_grouped[i].loc[mask].copy()
    train_df1, validation_df1 = train_test_split(selected_df1, 
                                                 test_size=1-train_fraction,
                                                 random_state=random_state)
    selected_df2 = diet_grouped[i].loc[~mask].copy()
    train_df2, validation_df2 = train_test_split(selected_df2, 
                                                 test_size=1-train_fraction,
                                                    random_state=random_state)
    train_df = pd.concat([train_df1, train_df2])
    validation_df = pd.concat([validation_df1, validation_df2])
    
    print(diets[i], train_df.shape, validation_df.shape)
    ttags += train_df.mouse_id.to_list()
    vtags += validation_df.mouse_id.to_list()
    
    n_train.append(train_df.shape[0])
    n_validation.append(validation_df.shape[0])
    
    train_data = train_df.drop(['mouse_id', 'diet', 'lengths'], axis=1).to_numpy()
    validation_data = validation_df.drop(['mouse_id', 'diet', 'lengths'], axis=1).to_numpy()

    def trim_trailing_nans(y):
      flip_nans = np.isnan(y[::-1])
      i = 0
      while flip_nans[i]:
          i += 1
      return y[:len(y)-i].ravel()
  
    for j in range(train_data.shape[0]):
      x = trim_trailing_nans(train_data[j, :])
      nans, y = np.isnan(x), lambda z: z.nonzero()[0]
      if np.sum(nans) != 0:
        x[nans]= np.interp(y(nans), y(~nans), x[~nans])
      # tdatas.append(np.expand_dims(np.diff(x).T, axis=1))
      tdatas.append(np.expand_dims(x, axis=1))

    for j in range(validation_data.data.shape[0]):
      x = trim_trailing_nans(validation_data[j, :])
      nans, y = np.isnan(x), lambda z: z.nonzero()[0]
      if np.sum(nans) != 0:
        x[nans]= np.interp(y(nans), y(~nans), x[~nans])
      # vdatas.append(np.expand_dims(np.diff(x).T, axis=1))
      vdatas.append(np.expand_dims(x, axis=1))
    
  return (tdatas, vdatas, ttags, vtags, n_train, n_validation, diets)


def train_viarhmm(viarhmm_train_fn,
                 ks,
                 ms,
                 tdatas,
                 inputs,
                 masks,
                 ttags,
                 model_dir,
                 threshold=None):
  """
  Function to train VIARHMM models for different values of k and m.

  Args:
  ----
    viarhmm_train_fn (function): training function
    ks (list): list of number of states
    ms (list): list of lag orders
    tdatas (list): list of training data
    inputs (list): NA
    masks (list): NA
    ttags (list): unique ID assigned to each body weight trace
    model_dir (str): directory to save the trained models
    threshold (float): threshold value for the observation model
  
  Returns:
  -------
    viarhmm_models (list): list of trained VIARHMM models
  """

  viarhmm_models = []
  for k in ks:
    for m in ms:
      cached_msar_model_filename = f'{model_dir}/viarhmm_model_{k}_{m}.pkl'
      if not os.path.isfile(cached_msar_model_filename):
        viarhmm_model = viarhmm_train_fn(k, m, tdatas, inputs, masks, ttags, threshold=threshold)
        with open(cached_msar_model_filename, 'wb') as f:
          pickle.dump(viarhmm_model, f)
      else:
        with open(cached_msar_model_filename, 'rb') as f:
          viarhmm_model = pickle.load(f)
      print(f'Saved {cached_msar_model_filename}.')
      viarhmm_models.append(viarhmm_model)

  return viarhmm_models

def viarhmm_train_fn(k,
                     m,
                     tdatas,
                     inputs,
                     masks,
                     ttags,
                     random_state,
                     threshold=None):
  """
  Function to train VIARHMM model for given k and m.

  Args:
  ----
    k (int): number of states
    m (int): lag order
    tdatas (list): list of training data
    inputs (list): NA
    masks (list): NA
    ttags (list): unique ID assigned to each body weight trace
    random_state (int): random seed
    threshold (float): threshold value for the observation model

  Returns:
  -------
    viarhmm_model (VIARHMM): trained VIARHMM model

  """

  observation_kwargs = {"threshold": threshold}
  viarhmm_model = VIARHMM(k,
                          m,
                          observation_kwargs=observation_kwargs,
                          random_state=random_state)

  elbos = viarhmm_model.fit(tdatas,
                            inputs,
                            masks,
                            ttags)

  viarhmm_model.elbos = elbos.copy()

  return viarhmm_model

def validate_viarhmm(viarhmm_validation_fn,
                    models,
                    ks,
                    ms,
                    vdatas,
                    inputs,
                    masks,
                    vtags,
                    model_dir,
                    threshold=None):
  """
  Function to validate VIARHMM models for different values of k and m.

  Args:
  ----
    viarhmm_validation_fn (function): validation function
    models (list): list of trained VIARHMM models
    ks (list): list of number of states
    ms (list): list of lag orders
    vdatas (list): list of validation data
    inputs (list): NA
    masks (list): NA
    vtags (list): unique ID assigned to each body weight trace
    model_dir (str): directory to save the trained models
    threshold (float): threshold value for the observation model
  
  Returns:
  -------
    validation_models (list): list of trained VIARHMM models

  """
  validation_models = []
  for k in ks:
    for m in ms:
      cached_msar_model_filename = f'{model_dir}/validate_model_{k}_{m}.pkl'
      if not os.path.isfile(cached_msar_model_filename):
        validation_model = viarhmm_validation_fn(models, k, m, vdatas, inputs, masks, vtags, threshold=threshold)
        with open(cached_msar_model_filename, 'wb') as f:
          pickle.dump(validation_model, f)
      else:
        with open(cached_msar_model_filename, 'rb') as f:
          validation_model = pickle.load(f)
      print(f'Saved {cached_msar_model_filename}.')
      validation_models.append(validation_model)

  return validation_models

def viarhmm_validation_fn(model,
                          k,
                          m,
                          vdatas,
                          inputs,
                          masks,
                          vtags,
                          random_state,
                          threshold=None):
  """
  Function to validate VIARHMM model for given k and m.

  Args:
  ----
    model (VIARHMM): trained VIARHMM model
    k (int): number of states
    m (int): lag order
    vdatas (list): list of validation data
    inputs (list): NA
    masks (list): NA
    vtags (list): unique ID assigned to each body weight trace
    random_state (int): random seed
    threshold (float): threshold value for the observation model

  Returns:
  -------
    validation_model (VIARHMM): trained VIARHMM model

  """
  observation_kwargs = {"threshold": threshold}
  validation_model = VIARHMM(k,
                             m,
                             observation_kwargs=observation_kwargs,
                             random_state=random_state)
  log_pi0, log_Ps, etas, log_sigmasqs, \
    log_err_sigmasqs = model.observations.params
  validation_model.observations.params = log_pi0, log_Ps, \
    etas, log_sigmasqs, log_err_sigmasqs
  
  elbos = validation_model.predict(vdatas,
                                   inputs,
                                   masks,
                                   vtags,
                                   random_state=random_state)

  validation_model.elbos = elbos.copy()

  return validation_model


def compute_loglikelihoods(model, 
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
  logprob = []
  log_pi0, log_Ps, etas, log_sigamsqs, log_err_sigmasqs = \
    model.observations.params
  
  pi0 = np.exp(log_pi0).ravel().copy()
  Ps = np.expand_dims(np.exp(log_Ps), axis=0).copy()

  for data, tag in zip(datas, tags):
    lls[tag] = _compute_loglikelihoods(model=model, 
                                       data=data, 
                                       tag=tag)
    logprob.append(hmm_normalizer(pi0, Ps, lls[tag]))
  return logprob


def _compute_loglikelihoods(model,
                            data,
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
  K, M = model.K, model.M
  T = data.shape[0]
  ll = np.zeros((T, K))

  _, _, etas, log_sigmasqs, log_err_sigmasqs = \
    model.observations.params

  # variational parameters for the given trace
  eta = etas.copy()
  sigmasq = np.exp(log_sigmasqs).copy()

  for t in np.arange(M, T):
    tmp = - 0.5 * log_err_sigmasqs \
        - 0.5 * np.multiply((1 / np.exp(log_err_sigmasqs)), \
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


def compute_deviances(model, 
                      datas, 
                      inputs=None, 
                      masks=None, 
                      tags=None):
  """
  Function to compute the deviance of the model.

  Args:
  ----
    datas (list of 1D array): contains body weight traces in
    the form of a column vector
    inputs (list): NA
    masks (list): NA
    tags (list): unique ID assigned to each body weight trace

  Returns:
  -------
    dic (dict): contains the deviance of the model for each
    body weight trace.
  """
  
  dic = {}
  for data, tag in zip(datas, tags):
    dic[tag] = _compute_deviance(model=model,
                                 data=data,
                                 tag=tag)

  return dic

def _compute_deviance(model,
                      data,
                      tag):
  """
  Function to compute the deviance of the model.

  Args:
  ----
    model (VIARHMM): trained VIARHMM model
    data (1D array): contains body weight traces in
    the form of a column vector
    tag (int): unique ID assigned to each body weight trace
  
  Returns:
  -------
    deviance (float): deviance of the model for a given body weight trace

  """
  
  log_pi0, log_Ps, etas, log_sigmasqs, _ = model.observations.params
  _, _, tilda_etas, tilda_log_sigmasqs, _, _ = model.observations.var_params

  pi0 = np.exp(log_pi0).ravel().copy()
  Ps = np.expand_dims(np.exp(log_Ps), axis=0).copy()
  sigmasqs = np.exp(log_sigmasqs)
  tilda_eta = tilda_etas[tag]
  tilda_sigmasq = np.exp(tilda_log_sigmasqs[tag])

  ll = _compute_loglikelihoods(model=model, 
                               data=data, 
                               tag=tag)
  logprob = hmm_normalizer(pi0, Ps, ll)
  pd_1 = np.divide((etas - tilda_eta)**2, tilda_sigmasq)
  pd_2 = np.divide((etas - tilda_eta)**2, sigmasqs) + tilda_sigmasq
  pd = pd_2.sum() - pd_1.sum()
  return (-2*logprob + 2*pd)

def viarhmm_predict_fn(model,
                          k,
                          m,
                          vdatas,
                          inputs,
                          masks,
                          vtags,
                          random_state,
                          threshold=None):
  """
  Function to validate VIARHMM model for given k and m.

  Args:
  ----
    model (VIARHMM): trained VIARHMM model
    k (int): number of states
    m (int): lag order
    vdatas (list): list of validation data
    inputs (list): NA
    masks (list): NA
    vtags (list): unique ID assigned to each body weight trace
    random_state (int): random seed
    threshold (float): threshold value for the observation model

  Returns:
  -------
    validation_model (VIARHMM): trained VIARHMM model

  """
  observation_kwargs = {"threshold": threshold}
  validation_model = VIARHMM(k,
                             m,
                             observation_kwargs=observation_kwargs,
                             random_state=random_state)
  log_pi0, log_Ps, etas, log_sigmasqs, \
    log_err_sigmasqs = model.observations.params
  validation_model.observations.params = log_pi0, log_Ps, \
    etas, log_sigmasqs, log_err_sigmasqs
  
  elbos = validation_model.predict(vdatas,
                                   inputs,
                                   masks,
                                   vtags,
                                   random_state=random_state)

  validation_model.elbos = elbos.copy()

  return validation_model


def predict_viarhmm(viarhmm_predict_fn,
                    models,
                    ks,
                    ms,
                    vdatas,
                    inputs,
                    masks,
                    vtags,
                    model_dir,
                    threshold=None):
  """
  Function to validate VIARHMM models for different values of k and m.

  Args:
  ----
    viarhmm_predict_fn (function): validation function
    models (list): list of trained VIARHMM models
    ks (list): list of number of states
    ms (list): list of lag orders
    vdatas (list): list of validation data
    inputs (list): NA
    masks (list): NA
    vtags (list): unique ID assigned to each body weight trace
    model_dir (str): directory to save the trained models
    threshold (float): threshold value for the observation model

  Returns:
  -------
    validation_models (list): list of trained VIARHMM models

  """
  validation_models = []
  for k in ks:
    for m in ms:
      cached_msar_model_filename = f'{model_dir}/predicted_model_{k}_{m}.pkl'
      if not os.path.isfile(cached_msar_model_filename):
        validation_model = viarhmm_predict_fn(models, k, m, vdatas, inputs, masks, vtags, threshold=threshold)
        with open(cached_msar_model_filename, 'wb') as f:
          pickle.dump(validation_model, f)
      else:
        with open(cached_msar_model_filename, 'rb') as f:
          validation_model = pickle.load(f)
      print(f'Saved {cached_msar_model_filename}.')
      validation_models.append(validation_model)

  return validation_models