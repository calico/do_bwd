import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from viarhmm.settings import MAX_SEED, START_K, LEN_THRESHOLD
from viarhmm.settings import TRAIN_FRACTION, ERR_SIGMA, MODEL_DIR
from viarhmm.preprocess import split_train_validation

PHENOTYPE_ASSAY_DATASET = "data/age_phenotype.csv"

def fit_exponential(prob_vec, state_vec, intervention=None):
  """ Function to extract the start and stop of homeostasis and
  the rate of homeostasis

  Parameters
  ----------
  prob_vec : np.ndarray
    Probability vector of the state
  state_vec : np.ndarray
    State vector
  intervention : str
    Type of intervention
  
  Returns
  -------
  rate : np.ndarray
    Rate of homeostasis
  event : np.ndarray
    Start and stop of homeostasis
  """

  from scipy.signal import find_peaks
  probs = prob_vec.copy()
  state = state_vec.copy()

  # Check intervention
  #   - if "pre": set everything after pre to nan
  #   - if "post": set everything after post to nan
  #   - default mode - "post" because we observe longer bodyweight

  if intervention == 'pre':
    probs[16:] = np.nan
  elif intervention == 'post':
    probs[:15] = np.nan
  else:
    probs[:15] = np.nan

  # Invert he probability curve and find peaks
  tmp = (1 - probs)
  tmp[tmp < 0.05] = 0
  peak_starts, _ = find_peaks(tmp, distance=2)

  # Initialize
  lambds = []

  # Return variables
  #   - rate contains exponential rate at specific peaks
  #   - events contains the start and end of an event
  #       + 1 - start of the event
  #       + 2 - send of the event
  # These events are not necessarily phenotyping event but any
  # pertubation in the steady-state probability curve
  rate = np.zeros((len(probs), ))
  event = np.zeros((len(probs), ))

  for p, peak_start in enumerate(peak_starts):
    start = peak_start.copy()

    # Continue iterating until
    #   a) the probability curve is monotonous
    #   b) the probability at next timepoint is less than 0.9
    while (probs[start+1] > probs[start]):
      start = start + 1
      if probs[start] >= 0.95:
        break

    # Check if both conditions are satisfied
    #   - if the probability of the monotonically increasing
    #     function is above 0.95
    #   - if the endpoint represnts a steady state
    if (probs[start] < 0.95) and (state[start] != 1):
      continue

    # Location where it stops
    data = probs[peak_start:start]

    # Check if the length is sufficiently large
    if (len(data) >= 3):
      x_range = np.arange(0, len(data)).ravel()
      l_range = np.linspace(0.001, 4, 200).ravel()
      logprobs = np.asarray([np.linalg.norm((1-np.exp(-lambd*x_range)) - \
          (data - data[0])) for lambd in l_range])
      lambds = l_range[np.abs(logprobs).argmin()]

      rate[peak_start.astype(int)] = lambds
      event[peak_start.astype(int)] = 1
      event[start.astype(int) - 1] = -1

  return (rate, event)

def reconstructed_signal(bw, state, etas):
  """ Function to reconstruct the body weight signal from the
  state and etas

  Parameters
  ----------
  bw : np.ndarray
    Body weight signal
  state : np.ndarray
    State vector
  etas : np.ndarray
    AR coefficients

  Returns
  -------
  rec : np.ndarray
    Reconstructed body weight signal
  """

  etas_0 = etas[etas[:, 0].argsort()]
  # print(bw.shape, etas_0.shape, state.shape)
  rec = np.zeros(bw.shape)
  rec[:] = np.nan
  rec[0, 0] = bw[0, 0]
  for j in np.arange(1, len(bw)):
    if ~np.isnan(state[j, 0]):
      rec[j, 0] = rec[j-1, 0] + etas_0[int(state[j, 0]), 0]

  return rec

def aggregate_data(df_datas, order, model, tags, datas, n_max, df_ch, st_dir):
  """
  Function to aggregate the data

  Parameters
  ----------
  df_datas : list
    List of dataframes
  order : np.ndarray
    Order of the states
  model : viarhmm.models.VIARHMM
    Model
  tags : list
    List of tags
  datas : list
    List of data
  
  Returns
  -------
  df_datas : list
    List of dataframes
  """
  
  # Get the number of timepoints
  n = n_max

  # For dataset
  _, tilda_log_Ps, tilda_etas, _, tilda_pis, _ = model.observations.var_params
  for (tag, data) in zip(tags, datas):
    p, _ = data.shape
    arr = np.empty((n - p, 1))
    arr[:] = np.nan
    pheno = [''] * n

    if tag in df_ch.index.to_list():
      df_pheno = df_ch.loc[tag].copy()
      if isinstance(df_pheno, pd.core.series.Series):
        df_pheno = pd.DataFrame([df_ch.loc[tag]],
                                  columns=df_ch.columns.to_list(),
                                  index=[tag])
        df_pheno = df_pheno.append([df_pheno]*2,ignore_index=False)

      indices = df_pheno.loc[tag]["Inds"].to_numpy()
      ptypes = df_pheno.loc[tag]["Code"].to_list()
      for i, ind in enumerate(indices):
        pheno[ind] = ptypes[i]

    tilda_pi = tilda_pis[tag] / tilda_pis[tag].sum(axis=1, keepdims=True)
    tilda_pi = tilda_pi[:, order]
    state = np.expand_dims(tilda_pi.argmax(axis=1), axis=1)

    # transitions
    flat_tilda_Ps = np.expand_dims(np.exp(tilda_log_Ps[tag][:, order][order]).flatten(), axis=1)
    nan_arr = np.empty((n-len(flat_tilda_Ps), 1))
    nan_arr[:] = np.nan
    tilda_Ps = np.concatenate((flat_tilda_Ps, nan_arr), axis=0)

    # time varying joints
    with open(os.path.join(st_dir, tag + '.pkl'), 'rb') as f:
        joints = pickle.load(f)
    f.close()
    for t in range(joints.shape[1]):
      tmp = joints[:, t]
      tmp = tmp.reshape(model.K, model.K)
      tmp = tmp[:, order][order]
      joints[:, t] = tmp.flatten()
    nan_arr = np.empty((joints.shape[0], n-joints.shape[1]-1))
    nan_arr[:] = np.nan
    pre_nan_arr = np.empty((joints.shape[0], 1))
    pre_nan_arr[:] = np.nan
    joints = np.concatenate((pre_nan_arr, joints, nan_arr), axis=1)

    # if n - p != 0:
    if (n - p) >=0:
        data = np.concatenate((data, arr), axis=0)
        state = np.concatenate((state, arr), axis=0)
        probs = np.concatenate((tilda_pi.T, np.tile(arr, (1, model.K)).T), axis=1)

    ## get homeostasis
    pre_rates, pre_events = fit_exponential(probs[model.K-1, :], state, intervention='pre')
    post_rates, post_events = fit_exponential(probs[model.K//2, :], state, intervention='post')
    rates = np.concatenate((pre_rates[:15], post_rates[15:]), axis=0)
    events = np.concatenate((pre_events[:15], post_events[15:]), axis=0)
    data_rec = reconstructed_signal(data, state, tilda_etas[tag])

    df_datas.append(data.ravel())
    df_datas.append(state.ravel())
    df_datas.append(tilda_Ps.ravel())
    df_datas.append(pheno)
    df_datas.append(rates)
    df_datas.append(events)
    df_datas.append(data_rec.ravel())
    for j in range(model.K):
        df_datas.append(probs[j, :].ravel())
    df_datas.append(joints)

  return df_datas

def build_dataframe(df, tmodels, vmodels, model_order=3, start_k=START_K):
  """
  Function to build a dataframe from the model

  Parameters
  ----------
  df : pd.DataFrame
    Dataframe containing the body weight data
  tmodels : list
    List of training models
  vmodels : list
    List of validation models
  model_order : int
    Order of the model
  start_k : int
    Starting value of k

  Returns
  -------
  df_res : pd.DataFrame
    Dataframe containing the model data
  """

  def get_models(models, model_order=3, start_k=START_K):
    smodels = []
    for seed in range(MAX_SEED):
      smodels.append(models[seed][model_order-start_k])
    return smodels
  
  def best_elbos(models):
    elbos = []
    for model in models:
      elbos.append(model.elbos[-1])
    seed = np.asarray(elbos).argmax()
    print("Best seed: ", seed)
    return seed

  smodels = get_models(tmodels, model_order=model_order, start_k=start_k)
  seed = best_elbos(smodels)
  st_dir = os.path.join(MODEL_DIR.split('/')[0],
                        "best_model",
                        'seed_' + str(seed),
                        'threshold_' + str(ERR_SIGMA),
                        "state_transitions")

  tdatas, vdatas, ttags, vtags, _, _, _, = \
    split_train_validation(df,
                           len_threshold=LEN_THRESHOLD,
                           train_fraction=TRAIN_FRACTION,
                           random_state=seed)
  btmodel = tmodels[seed][model_order-start_k]
  bvmodel = vmodels[seed][model_order-start_k]

  # get the order of the states
  _, _, tmp, _, _ = btmodel.observations.params
  order = tmp[:, 0].argsort()

  all_tags = ttags + vtags
  row_tags = [(7 + btmodel.K + btmodel.K*btmodel.K)*[tag] for tag in all_tags]
  row_tags = sum(row_tags, [])
  names = ["bw", "states", "transitions", "phenotypes", "homeostasis", "events", "bw_rec"]
  tilda = ["pi_" + str(j) for j in range(btmodel.K)]
  transitions = ["st_" + str(j) + "_" + str(i) \
                  for i in range(btmodel.K) \
                  for j in range(btmodel.K)]

  names = names + tilda + transitions
  row_names = len(all_tags) * names

  df_data = []
  cnames = df.columns.to_list()[:-2]
  cnames = [name.strip('bw.') for name in cnames]
  n_timepoints = len(cnames)

  df_ch = pd.read_csv(PHENOTYPE_ASSAY_DATASET, sep=',')
  df_ch.index = df_ch.MouseID
  df_ch = df_ch.drop('MouseID', axis=1)
  df_ch['Appx'] = np.asarray(np.round(df_ch['AgeInDays'].to_numpy(), -1))
  df_ch['Inds'] = np.asarray((np.round(df_ch['AgeInDays'].to_numpy(), -1) - 30) // 10, dtype=int)
  df_ch['Code'] = [name.upper()[:2] for name in df_ch['Phenotyping'].to_list()]

  # train data
  df_data = aggregate_data(df_data, 
                           order, 
                           btmodel, 
                           ttags, 
                           tdatas, 
                           n_timepoints, 
                           df_ch, 
                           st_dir)
  
  # validation data
  df_data = aggregate_data(df_data, 
                           order, 
                           bvmodel, 
                           vtags, 
                           vdatas, 
                           n_timepoints, 
                           df_ch, 
                           st_dir)
  
  # create dataframe
  arrays = [np.array(row_tags), np.array(row_names)]
  df_res = pd.DataFrame(np.vstack(df_data), index=arrays, columns=cnames)
  return df_res