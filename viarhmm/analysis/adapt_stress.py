import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

PHENOTYPE_ASSAY_DATASET = "data/age_phenotype.csv"

def describe_events(df_input):
  """
  Describe the number of times each event occurs and the average age of the 
  event

  Args:
      df_input (pd.DataFrame): Phenotyping dataset

  Returns:
      tuple: Tuple of dictionaries containing the number of times each event 
      occurs and the average age of the event
  """

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

def get_cluster_indices(phenos, start_index):
  """
  Get the cluster indices of the phenotyping events

  Args:
    phenos (list): List of phenotyping events
    start_index (list): List of start indices

  Returns:
    tuple: Tuple of the cluster centers and the cluster indices
  """

  index_all_phenos = np.asarray([p for p, ph in enumerate(phenos) if ph != ''])
  cluster_index = []
  cluster_centers = []
  clusters = {}
  cluster_flag = 0

  for j in range(len(index_all_phenos)-1):
    if (index_all_phenos[j+1] - index_all_phenos[j]) <= 2:
      cluster_index.append(index_all_phenos[j])
      cluster_flag = 1
    elif cluster_flag == 1:
      cluster_index.append(index_all_phenos[j])
      cflag = [True if k in cluster_index else False for k in start_index]
      if np.sum(cflag) != 0:
        ind, = np.where(np.asarray(cflag, dtype=bool) == True)
        # remove anything greater than start_index[ind] from cluster_index
        ind = ind[0]
        cluster_index = [c for c in cluster_index if c < start_index[ind]]
        cluster_centers.append(start_index[ind].ravel())
        clusters[str(start_index[ind])] = cluster_index
      cluster_index = []
      cluster_flag = 0
    else:
      cluster_flag = 0
  if cluster_flag == 1:
    cluster_index.append(index_all_phenos[j+1])
    cflag = [True if k in cluster_index else False for k in start_index]
    if np.sum(cflag) != 0:
      ind, = np.where(np.asarray(cflag, dtype=bool) == True)
      # remove anything greater than start_index[ind] from cluster_index
      ind = ind[0]
      cluster_index = [c for c in cluster_index if c < start_index[ind]]
      cluster_centers.append(start_index[ind].ravel())
      clusters[str(start_index[ind])] = cluster_index

  merge_clusters = list(clusters.values())
  return np.array(cluster_centers).ravel(), merge_clusters

def create_rc_dataframe(df_data, threshold=30):
  """
  Create a dataframe of resilience curves

  Args:
      df_data (pd.DataFrame): Dataframe of the ARHMM model
      threshold (int, optional): Threshold value. Defaults to 30.
  
  Returns:
      tuple: Tuple of the resilience curve dataframe, the std error dataframe, and 
      the pflag dataframe
  """

  df_challenge = pd.read_csv(PHENOTYPE_ASSAY_DATASET, sep=',')
  df_challenge.index = df_challenge.MouseID
  df_challenge = df_challenge.drop('MouseID', axis=1)
  df_challenge['Appx'] = np.asarray(np.round(df_challenge['AgeInDays'].to_numpy(), -1))
  df_challenge['Inds'] = np.asarray((np.round(df_challenge['AgeInDays'].to_numpy(), -1) - 30) // 10, dtype=int)
  df_challenge['Code'] = [name.upper()[:2] for name in df_challenge['Phenotyping'].to_list()]

  max_counts, age_stats = describe_events(df_challenge)

  event_dict = {
                'FA':'facs',
                'AC':'acoustic.startle',
                'HO':'homecage.wheel',
                # 'PI':'pixi',
                'FR':'frailty',
                # 'RO':'rotarod',
                # 'GR':'grip',
                'CB':'cbc',
                'ME':'metcage',
                'EC':'ecg'}

  all_occurance = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
  diets = ["AL", "1D", "2D", "20", "40"]
  mouse_ids = df_data.index.get_level_values(0).unique().to_list()
  index = mouse_ids.copy()

  # location of diet indices
  diet_inds = {diet:[] for diet in diets}
  for mouse in mouse_ids:
    diet = mouse.split('-')[1]
    diet_inds[diet].append(mouse_ids.index(mouse))

  df_pflag = pd.DataFrame(False, columns=list(event_dict.keys()), index=diets)
  columns = ["diets"] + list(event_dict.keys())
  df_rc = pd.DataFrame([], columns=columns)
  df_err = pd.DataFrame([], columns=columns)
  df_rc["diets"] = diets
  df_err["diets"] = diets

    # for all phenotyping events
  for event in event_dict.keys():

    # count the number of times
    occurance = all_occurance[:max_counts[event]]

    # generate cnames
    cnames = ["resilience." + txt + '.' + event_dict[event] for txt in occurance]

    # create dataframe for the phenotyping event
    df_event = pd.DataFrame([], index=index, columns=cnames)

    # get average age
    if event is not None:
      age_bin_stats = age_stats[event]
      sindex = np.logical_and(age_bin_stats[0] > 180,
                              age_bin_stats[0] < 1050)
      dindex = np.array(np.where(sindex == True)[0]).astype(int)
      sindex = np.array(np.where(sindex == False)[0]).astype(int)

    for diet in diets:

      for mouse in mouse_ids:

        grp = mouse.split('-')[1]
        if grp == diet:

          values = np.zeros((len(cnames), ))
          values[:] = np.nan

          # indices of the phenotyping event
          phenos = df_data.loc[(mouse, "phenotypes")].to_list()
          index_phenos = np.asarray([p for p, ph in enumerate(phenos) \
                                      if ph == event])
          lambdas = df_data.loc[(mouse, "homeostasis")].to_numpy().astype(float)

          # indices of the start and stop of the pertubation event
          st_sp = df_data.loc[(mouse, "events")].to_numpy().astype(float)

          index_start, = np.where(st_sp == 1.0)
          index_stop, = np.where(st_sp == -1.0)
          if len(index_start) != len(index_stop):
            index_stop = index_stop[:len(index_start)]

          # adjust start index using threshold
          if threshold is not None:
            threshold_index_start = index_start - (threshold // 10)
          else:
            threshold_index_start = index_start.copy()

          # compute resilience
          cluster_centers, cluster_lists = \
            get_cluster_indices(phenos, threshold_index_start)
          flat_clusters = [item for sublist in cluster_lists for item in sublist]

          # compute resilience
          for i in index_phenos:

            # check if i belongs in the pertubation region
            flag = np.logical_and(i >= threshold_index_start, i <= index_start)

            # if pertubation event is found
            if np.sum(flag) != 0:
              pertubation_index, = np.where(flag == True)
              lambda_index = index_start[pertubation_index]

            # check if i belongs to cluster indices
            elif i in flat_clusters:
              for a, arr in enumerate(cluster_lists):
                if i in arr:
                  pertubation_index, = np.where(threshold_index_start == cluster_centers[a])
                  lambda_index = index_start[pertubation_index]
                  break

            else:
              continue

            # convert index to age in days
            i = i*10 + 30

            # check when age bin it belongs to
            age_index = np.abs(age_bin_stats[0, :] - i).argmin()
            values[age_index] = np.max(lambdas[lambda_index])

          df_event.loc[mouse][cnames] = values

    # drop columns
    df_event = df_event.drop(df_event.columns[sindex], axis=1)

    all_data = df_event.astype('float').to_numpy()
    avg = []
    err = []

    for diet in diets:
      data = all_data[diet_inds[diet], :]
      # if all rows are nan, drop them
      data = data[~np.isnan(data).all(axis=1)]
      avg.append(np.nanmean(np.nanmean(data, axis=1), axis=0))
      tmp = np.nanmean(data, axis=1)
      err.append(np.nanstd(tmp) / np.sqrt(np.sum(~np.isnan(tmp))))

      if diet == 'AL':
        ref_data = np.nanmean(data, axis=1)

    for diet in diets:
      if diet != 'AL':
        data = all_data[diet_inds[diet], :]
        data = data[~np.isnan(data).all(axis=1)]
        data = np.nanmean(data, axis=1)
        _, p = mannwhitneyu(ref_data, data)
        if p < 0.05:
          df_pflag.loc[diet, event] = True

    df_rc[event] = avg
    df_err[event] = err

  return df_rc, df_err, df_pflag

def create_age_rc_dataframe(df_data, intervals, threshold=30):
  """
  Create a dataframe of resilience curves

  Args:
      df_data (pd.DataFrame): Dataframe of the ARHMM model
      intervals (list): List of intervals
      threshold (int, optional): Threshold value. Defaults to 30.
  
  Returns:
      tuple: Tuple of the resilience curve dataframe, the std error dataframe, and 
      the pflag dataframe
  """

  df_challenge = pd.read_csv(PHENOTYPE_ASSAY_DATASET, sep=',')
  df_challenge.index = df_challenge.MouseID
  df_challenge = df_challenge.drop('MouseID', axis=1)
  df_challenge['Appx'] = np.asarray(np.round(df_challenge['AgeInDays'].to_numpy(), -1))
  df_challenge['Inds'] = np.asarray((np.round(df_challenge['AgeInDays'].to_numpy(), -1) - 30) // 10, dtype=int)
  df_challenge['Code'] = [name.upper()[:2] for name in df_challenge['Phenotyping'].to_list()]

  max_counts, age_stats = describe_events(df_challenge)

  event_dict = {
                'FA':'facs',
                'AC':'acoustic.startle',
                'HO':'homecage.wheel',
                # 'PI':'pixi',
                'FR':'frailty',
                # 'RO':'rotarod',
                # 'GR':'grip',
                'CB':'cbc',
                'ME':'metcage',
                'EC':'ecg'}

  all_occurance = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
  diets = ["AL", "1D", "2D", "20", "40"]
  mouse_ids = df_data.index.get_level_values(0).unique().to_list()
  index = mouse_ids.copy()

  # location of diet indices
  diet_inds = {diet:[] for diet in diets}
  for mouse in mouse_ids:
    diet = mouse.split('-')[1]
    diet_inds[diet].append(mouse_ids.index(mouse))

  list_es, list_se, list_pflag = [], [], []
  ref_dict = {}
  for r, region_in_days in enumerate(intervals):

    df_pflag = pd.DataFrame(False, columns=list(event_dict.keys()), index=diets)
    columns = ["diets"] + list(event_dict.keys())
    df_rc = pd.DataFrame([], columns=columns)
    df_err = pd.DataFrame([], columns=columns)
    df_rc["diets"] = diets
    df_err["diets"] = diets

      # for all phenotyping events
    for event in event_dict.keys():

      # count the number of times
      occurance = all_occurance[:max_counts[event]]

      # generate cnames
      cnames = ["resilience." + txt + '.' + event_dict[event] for txt in occurance]

      # create dataframe for the phenotyping event
      df_event = pd.DataFrame([], index=index, columns=cnames)

      # get average age
      if event is not None:
        age_bin_stats = age_stats[event]
        sindex = np.logical_and(age_bin_stats[0] > region_in_days[0],
                            age_bin_stats[0] < region_in_days[1])
        dindex = np.array(np.where(sindex == True)[0]).astype(int)
        sindex = np.array(np.where(sindex == False)[0]).astype(int)

      for diet in diets:

        for mouse in mouse_ids:

          grp = mouse.split('-')[1]
          if grp == diet:

            values = np.zeros((len(cnames), ))
            values[:] = np.nan

            # indices of the phenotyping event
            phenos = df_data.loc[(mouse, "phenotypes")].to_list()
            index_phenos = np.asarray([p for p, ph in enumerate(phenos) \
                                        if ph == event])
            lambdas = df_data.loc[(mouse, "homeostasis")].to_numpy().astype(float)

            # indices of the start and stop of the pertubation event
            st_sp = df_data.loc[(mouse, "events")].to_numpy().astype(float)

            index_start, = np.where(st_sp == 1.0)
            index_stop, = np.where(st_sp == -1.0)
            if len(index_start) != len(index_stop):
              index_stop = index_stop[:len(index_start)]

            # adjust start index using threshold
            if threshold is not None:
              threshold_index_start = index_start - (threshold // 10)
            else:
              threshold_index_start = index_start.copy()

            # compute resilience
            cluster_centers, cluster_lists = \
              get_cluster_indices(phenos, threshold_index_start)
            flat_clusters = [item for sublist in cluster_lists for item in sublist]

            # compute resilience
            for i in index_phenos:

              # check if i belongs in the pertubation region
              flag = np.logical_and(i >= threshold_index_start, i <= index_start)

              # if pertubation event is found
              if np.sum(flag) != 0:
                pertubation_index, = np.where(flag == True)
                lambda_index = index_start[pertubation_index]

              # check if i belongs to cluster indices
              elif i in flat_clusters:
                for a, arr in enumerate(cluster_lists):
                  if i in arr:
                    pertubation_index, = np.where(threshold_index_start == cluster_centers[a])
                    lambda_index = index_start[pertubation_index]
                    break

              else:
                continue

              # convert index to age in days
              i = i*10 + 30

              # check when age bin it belongs to
              age_index = np.abs(age_bin_stats[0, :] - i).argmin()
              values[age_index] = np.max(lambdas[lambda_index])

            df_event.loc[mouse][cnames] = values

      # drop columns
      df_event = df_event.drop(df_event.columns[sindex], axis=1)

      all_data = df_event.astype('float').to_numpy()
      avg = []
      err = []
      for diet in diets:
        data = all_data[diet_inds[diet], :]
        # if all rows are nan, drop them
        data = data[~np.isnan(data).all(axis=1)]
        avg.append(np.nanmean(np.nanmean(data, axis=1), axis=0))
        tmp = np.nanmean(data, axis=1)
        err.append(np.nanstd(tmp) / np.sqrt(np.sum(~np.isnan(tmp))))

        if r == 0:
          ref_dict[diet] = np.nanmean(data, axis=1)

      for diet in diets:
        data = all_data[diet_inds[diet], :]
        data = data[~np.isnan(data).all(axis=1)]
        data = np.nanmean(data, axis=1)

        if r != 0:
          _, p = mannwhitneyu(ref_dict[diet], data)
          if p < 0.05:
            df_pflag.loc[diet, event] = True

      df_rc[event] = avg
      df_err[event] = err

    list_es.append(df_rc)
    list_se.append(df_err)
    list_pflag.append(df_pflag)

  return list_es, list_se, list_pflag