import os
import numpy as np
import pandas as pd
from scipy import special as sp
from functools import partial

EVENT_DICT = {'FA':'facs',
              'AC':'acoustic.startle',
              'HO':'homecage.wheel',
              # 'PI':'pixi',
              'FR':'frailty',
              # 'RO':'rotarod',
              # 'GR':'grip',
              'CB':'cbc',
              'ME':'metcage',
              'EC':'ecg'}


def phenotype_names(mode,
                    phenotype_name,
                    weighted):
  """
  Function to generate phenotype names based on the mode of processing

  Parameters
  ----------
    mode : str
      Mode of processing the phenotypes
    phenotype_name : str
      Name of the phenotype
    weighted : bool
      If True, then the weighted phenotype is generated
  
  Returns
  -------
    phenotypes : list
      List of phenotype names
  """

  state_names = ["DS", "SS", "GS"]
  events = []

  if mode == "prepost":
    phases = ["pre", "post"]
    time_intervals = []
    phenotypes = []

    if phenotype_name != "state_transitions":
      for state_name in state_names:
        pnames = [phase + "." + phenotype_name.replace("_", '.') + \
                    "." + state_name for phase in phases]
        if weighted:
          pnames = ["weighted." + pname for pname in pnames]
        phenotypes.append(pnames)
        events.append(state_name)
    elif phenotype_name == "state_transitions":
      for sname1 in state_names:
        for sname2 in state_names:
          pnames = [phase + "." + phenotype_name.replace("_", '.') + \
                      "." + sname1 + "." + sname2 for phase in phases]
          if weighted:
            pnames = ["weighted." + pname for pname in pnames]
          phenotypes.append(pnames)
          events.append(sname1 + "." + sname2)
    else:
      raise ValueError("Phenotype not found!")

  elif mode == "timeint":
    phases = []
    time_intervals = np.arange(0, 1260+1, 180)
    phenotypes = []

    if phenotype_name != "state_transitions":
      for state_name in state_names:
        pnames = [phenotype_name.replace("_", '.') + "." + \
                    str(time_intervals[t]) + '.to.' + \
                      str(time_intervals[t+1]) + "." + state_name \
                        for t, _ in enumerate(time_intervals[:-1])]
        if weighted:
          pnames = ["weighted." + pname for pname in pnames]
        phenotypes.append(pnames)
        events.append(state_name)
    elif phenotype_name == "state_transitions":
      for sname1 in state_names:
        for sname2 in state_names:
          pnames = [phenotype_name.replace("_", '.') + "." + \
                    str(time_intervals[t]) + '.to.' + \
                      str(time_intervals[t+1]) + "." + \
                        sname1 + "." + sname2 \
                          for t, _ in enumerate(time_intervals[:-1])]
          if weighted:
            pnames = ["weighted." + pname for pname in pnames]
          phenotypes.append(pnames)
          events.append(sname1 + "." + sname2)
    else:
      raise ValueError("Phenotype not found!")

  elif mode == "resilience":
    phases = ["post"]
    # time_intervals = [0, 180, 540, 900]
    time_intervals = [0, 180, 1600]
    phenotypes = []
    for col in list(EVENT_DICT.keys()):
      pnames = ["resilience.mean." + col + "." + \
                    str(time_intervals[t]) + ".to." + \
                      str(time_intervals[t+1]) \
                        for t, _ in enumerate(time_intervals[:-1])]
      phenotypes.append(pnames)
    events = list(EVENT_DICT.keys())
  else:
    raise ValueError("Phenotype processing mode not found!")

  return phenotypes, events


def get_dataframe(df_input,
                  phenotype_list,
                  transform=False):
  """
  Function to extract the phenotype data from the input dataframe

  Parameters
  ----------
    df_input : pd.DataFrame
      Input dataframe containing the phenotype data
    phenotype_list : list
      List of phenotypes to extract from the dataframe
    transform : str
      Type of transformation to apply to the phenotype data
  
  Returns
  -------
    df_return : pd.DataFrame
      Dataframe containing the phenotype data
  """

  df_return = df_input[phenotype_list]
  df_return.index = df_input["MouseID"].to_list()
  if transform in ["log", "sqrt", "logit"]:
    if transform == 'log':
      df_return = df_return.apply(np.log)
    elif transform == 'sqrt':
      df_return = df_return.apply(np.sqrt)
    elif transform == 'logit':
      df_return = df_return.apply(sp.logit)
    return df_return
  elif not transform:
    return df_return
  else:
    raise ValueError("Transform type not found!")


def describe_events(df_input):
    
  from collections import Counter
  count_dict = dict()
  max_count = dict()
  age_dict = dict()
  age_stats = dict()
  
  df = df_input.copy()
  mouse_ids = df.index.unique().to_list()
  for mouse_id in mouse_ids:
    if len(df.loc[mouse_id]["Code"]) != 2:
      codes = df.loc[mouse_id]["Code"].to_list()
      age_in_days = df.loc[mouse_id]["AgeInDays"].to_numpy()
      keys, values = np.unique(codes, return_counts=True)
      for key, value in zip(keys, values):
        index_key = [i for i, x in enumerate(codes) if x == key]
        age = age_in_days[index_key]
        if key in count_dict:
          count_dict[key].append(value)
          age_dict[key].append(age)
        else:
          count_dict[key] = [value]
          age_dict[key] = [age]
                  
  for i, key in enumerate(age_dict.keys()):
    max_count[key] = np.max(count_dict[key])
    df_age = pd.DataFrame(age_dict[key])
    mu = df_age.describe(include='all').loc['mean']
    std = df_age.describe(include='all').loc['std']
    value = np.vstack((mu, std))
    age_stats[key] = value

  max_count['HO'] -= 1
  age_stats['HO'] = age_stats['HO'][:, 1:]

  return (max_count, age_stats)



def get_resilience_dataframe(df_input,
                             df_ch,
                             phenotype_list):
  """
  Function to extract the resilience phenotype data from the input dataframe

  Parameters
  ----------
    df_input : pd.DataFrame
      Input dataframe containing the phenotype data
    df_ch : pd.DataFrame
      Challenge dataset containing the resilience phenotype data
    phenotype_list : list
      List of phenotypes to extract from the dataframe
  
  Returns
  -------
    df_rc : pd.DataFrame
      Dataframe containing the resilience phenotype data
  """

  # preprocessing of challenge dataset
  df_ch.index = df_ch.MouseID
  df_ch = df_ch.drop('MouseID', axis=1)
  df_ch['Appx'] = np.asarray(np.round(df_ch['AgeInDays'].to_numpy(), -1))
  df_ch['Inds'] = np.asarray((np.round(df_ch['AgeInDays'].to_numpy(), -1) - 30) // 10, dtype=int)
  df_ch['Code'] = [name.upper()[:2] for name in df_ch['Phenotyping'].to_list()]

  max_counts, age_stats = describe_events(df_ch)
  # region_in_days = [0, 180, 540, 900]
  region_in_days = [0, 180, 1600]
  all_occurance = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
  diets = ["AL", "1D", "2D", "20", "40"]
  index = df_input["MouseID"].to_list()
  df_rc = pd.DataFrame([], columns=phenotype_list, index=index)

  # for all phenotyping events
  for r, (colname, _) in enumerate(zip(phenotype_list, region_in_days[:-1])):

    # get event name
    event = colname.split('.')[2]

    # count the number of times
    occurance = all_occurance[:max_counts[event]]

    # generate cnames
    cnames = ["resilience." + txt + '.' + EVENT_DICT[event] for txt in occurance]

    # get average age
    if event is not None:

      age_bin_stats = age_stats[event]
      sindex = np.logical_and(age_bin_stats[0] >= region_in_days[r],
                              age_bin_stats[0] < region_in_days[r+1])
      pnames = [cnames[c] for c, val in enumerate(sindex) if val]

      # create dataframe for the phenotyping event
      if len(cnames) != 0:
        ph_list = df_input.columns.to_list()
        pnames = list(set(pnames) & set(ph_list))
        df_event = df_input[pnames].copy()
        all_data = df_event.astype('float').to_numpy()
        df_rc.loc[:, colname] = np.nanmean(all_data, axis=1)
      else:
        df_rc.loc[:, colname] = np.nan

  return df_rc


def encode_dataframe(df_all,
                     reference,
                     interaction,
                     bw_correction):
  """
  Function to encode the dataframe for TVCPH analysis

  Parameters
  ----------
    df_all : pd.DataFrame
      Dataframe containing the phenotype data
    reference : bool
      If True, then the reference group is removed
    interaction : bool
      If True, then interaction terms are included
    bw_correction : bool
      If True, then bodyweight correction is included
  
  Returns
  -------
    encoded_pd : pd.DataFrame
      Encoded dataframe for TVCPH analysis
  """

  diets = ["AL", "1D", "2D", "20", "40"]

  # remove nans and reset index
  df_all = df_all.dropna()
  df_all = df_all.reset_index(drop=True)

  # list of unique groups
  groups = df_all["group"].unique()

  # one-hot encoder for diet, generation, and group
  encode_cols = ['diet', 'generation', 'group']
  encoded_pd = pd.get_dummies(df_all,
                              columns=encode_cols,
                              prefix=encode_cols,
                              drop_first=False)

  for group in groups:

    encoded_pd["covariate_group_" + str(group)] = \
      np.multiply(encoded_pd["covariate"].to_numpy(),
                  encoded_pd["group_" + str(group)].to_numpy())

    if bw_correction:
      encoded_pd["bodyweight_group_" + str(group)] = \
        np.multiply(encoded_pd["bodyweight"].to_numpy(),
                    encoded_pd["group_" + str(group)].to_numpy())
        
    for diet in diets:

      if (group == 0.0) and (diet != 'AL'):
        continue

      encoded_pd["diet_" + diet + "_group_" + str(group)] = \
        np.multiply(encoded_pd["diet_" + diet].to_numpy(),
                    encoded_pd["group_" + str(group)].to_numpy())
      
      if interaction:
        encoded_pd["bodyweight" + "_diet_" + diet + "_group_" \
          + str(group)] = \
          np.multiply(encoded_pd["bodyweight"].to_numpy(),
                      encoded_pd["diet_" + diet + "_group_" + \
                        str(group)].to_numpy())


  # drop the trivial ones
  if 'G22W1' in list(df_all['generation'].unique()):
    encoded_pd = encoded_pd.drop(['covariate', 'bodyweight', \
                                  'generation_G22W1'], axis=1)
  else:
    selected_generation = list(df_all['generation'].unique())[0]
    encoded_pd = encoded_pd.drop(['covariate', 'bodyweight', \
                                    'generation_' + selected_generation], \
                                      axis=1)
    
  # drop diets because we are interested in group:diet
  for diet in diets:
    encoded_pd = encoded_pd.drop(['diet_' + diet], axis=1)

  # drop reference and groups
  if reference:
    for group in groups:
      encoded_pd = encoded_pd.drop(['group_' + str(group)], axis=1)
      encoded_pd = encoded_pd.drop(['diet_AL_' + 'group_' + str(group)], axis=1)
      encoded_pd = encoded_pd.drop(['bodyweight_' + 'diet_AL_' + \
                                    'group_' + str(group)], axis=1)

  return encoded_pd


def timeint_tvcph_dataframe(df_ph,
                            df_bw,
                            df_ls,
                            reference,
                            bw_correction,
                            interaction,
                            time_intervals,
                            scale):
  """
  Function to generate the TVCPH dataframe

  Parameters
  ----------
    df_ph : pd.DataFrame
      Phenotype dataframe
    df_bw : pd.DataFrame
      Bodyweight dataframe
    df_ls : pd.DataFrame
      Lifespan dataframe
    reference : bool
      If True, then the reference group is removed
    bw_correction : bool
      If True, then bodyweight correction is included
    interaction : bool
      If True, then interaction terms are included
    time_intervals : list
      List of time intervals
    scale : bool
      If True, then the covariates are scaled
  
  Returns
  -------
    encoded_pd : pd.DataFrame
      Encoded dataframe for TVCPH analysis  
  """
  mouse_ids = df_ph.index.to_list()
  time_intervals[1:] = time_intervals[1:] - 10

  df_all = []
  for m, mouse in enumerate(mouse_ids):
    
    # extract lifespan data
    generation = df_ls[df_ls["MouseID"] == mouse]["Cohort"].values[0]
    status = df_ls[df_ls["MouseID"] == mouse]["Status"].astype(float).values[0]

    # extract bodyweight data
    mouse_bw = df_bw.loc[(mouse, "bw")].to_numpy().astype(float)
    mouse_bw = mouse_bw[~np.isnan(mouse_bw)]
    len_bw = len(mouse_bw)
    start = [30+i*10 for i in np.arange(0,len_bw)]
    stop = [30+10+i*10 for i in np.arange(0,len_bw)]

    # extract covariates
    covariates = df_ph.loc[mouse, :].to_numpy()
    if scale:
      covariates = covariates * 100

    df_bw_diet = pd.DataFrame([], columns=["mouse_id", "status", "start", "stop", 
                                           "bodyweight", "covariate",
                                           "diet", "generation", "group"],
                              index=np.arange(0, len_bw))


    df_bw_diet["start"] = start
    df_bw_diet["stop"] = stop
    df_bw_diet["mouse_id"] = [mouse]*len_bw
    df_bw_diet["status"] = 0.0
    if status == 1.0:
      df_bw_diet.loc[len_bw-1, "status"] = 1.0
    df_bw_diet["bodyweight"] = mouse_bw
    df_bw_diet["diet"] = [mouse.split('-')[1]]*len_bw
    df_bw_diet["generation"] = [generation]*len_bw

    df_bw_diet["group"] = pd.cut(start, time_intervals, labels=False).astype(float)
    for g, group in enumerate(df_bw_diet.groupby("group")):
      _, X = group
      df_bw_diet.loc[df_bw_diet["group"] == g, "covariate"] = covariates[g]

      if g == 0:
        df_bw_diet.loc[df_bw_diet["group"] == g, "diet"] = 'AL'
    df_all.append(df_bw_diet)

  df_all = pd.concat(df_all, axis=0)
  df_all = df_all.reset_index(drop=True)

  encoded_pd = encode_dataframe(df_all=df_all,
                                reference=reference,
                                bw_correction=bw_correction,
                                interaction=interaction)

  return encoded_pd


def save_params(phenotypes,
                model,
                status,
                time_intervals,
                model_dir):
  """
  Function to save the model parameters

  Parameters
  ----------
    phenotypes : list
      List of phenotype names
    model : statsmodels object
      Fitted statsmodels object
    status : bool
      If True, then the model is fitted
    time_intervals : list
      List of time intervals
    model_dir : str
      Directory to save the model parameters

  Returns
  -------
    None
  """

  diets = ["AL", "1D", "2D", "20", "40"]
  xdiets = ['bwx' + diet for diet in diets]
  xcov = ['covx' + diet for diet in diets]
  params = ['coef', 'se(coef)', 'p']
  covariates = ['phenotype', 'body_weight'] + diets + xdiets
  iterables = [params, covariates]
  index = pd.MultiIndex.from_product(iterables, names=["param", "covariate"])

  for t, (_, phenotype) in enumerate(zip(time_intervals, phenotypes)):
    
    df = pd.DataFrame(np.nan, columns=[phenotype], index=index)
    cached_csv = os.path.join(model_dir, phenotype + '.csv')

    if not status:
      df.to_csv(cached_csv, sep=',')
    else:
      for param in params:
        pname = "covariate_group_" + str(float(t))
        if model.summary.index.isin([pname]).any():
          df.loc[(param, 'phenotype'), phenotype] = \
            model.summary.loc[pname, param]
          
        pname = "bodyweight_group_" + str(float(t))
        if model.summary.index.isin([pname]).any():
          df.loc[(param, 'body_weight'), phenotype] = \
            model.summary.loc[pname, param]
          
        for diet in diets:
          pname = "diet_" + diet + "_group_" + str(float(t))
          if model.summary.index.isin([pname]).any():
            df.loc[(param, diet), phenotype] = \
              model.summary.loc[pname, param]
            
          pname = "bodyweight_diet_" + diet + "_group_" + str(float(t))
          if model.summary.index.isin([pname]).any():
            df.loc[(param, 'bwx' + diet), phenotype] = \
              model.summary.loc[pname, param]
      df.to_csv(cached_csv, sep=',')

  return