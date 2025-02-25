from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats
from scipy.stats import rankdata, norm

sys.path.append(os.getcwd())

from viarhmm.preprocess import str_to_bool
from do_qtl.lib.models import gxemm
from do_qtl.lib import data_io_v2 as io


import warnings
warnings.filterwarnings("ignore")

## save path of heritability plots
MODEL_DIR = 'data/heritability'
VERBOSE = False

## location of phenotypes files
PREPOST_PHENOTYPE_FILE = 'data/prepost_v1.csv'
TIMEINT_PHENOTYPE_FILE = 'data/timeint_v1.csv'
RESILEN_PHENOTYPE_FILE = 'data/resilience_v1.csv'

## location of genetics files
CHALLENGE_FILE = 'data/age_phenotype.csv'
DIET_COVARIATE_FILE = 'data/diet_covariates.csv'
GEN_COVARIATE_FILE = 'data/generation_covariates.csv'
KINSHIP_FILE = 'data/kinship_matrix.genoprob.csv'

## default phenotype name
PHENOTYPE_NAME = 'weighted.post.state.occupancy.GS'
TRANSFORM = "rankint"
OUTLIER = False
Q_LOW = None
Q_HIGH = None

def parse_args():
  """Pass arguments

  Returns:
      args ([dict]): Contains the arguments and values associated
  """
  parser = argparse.ArgumentParser(
    description="Fit hertiability on JAX body weight features.")

  parser.add_argument("--model_dir", 
                      type=str, 
                      default=MODEL_DIR,
                      help=f'Directory to store fit ARHMM models. Default: {MODEL_DIR}')

  parser.add_argument("--phenotype_name", 
                      type=str, 
                      default=PHENOTYPE_NAME,
                      help=f'Name of the phenotype of interest. Default: {PHENOTYPE_NAME}')

  parser.add_argument("--diet_covariate_file", 
                      type=str,
                      default=DIET_COVARIATE_FILE, 
                      help=f'The covariate file of the diets. Default: {DIET_COVARIATE_FILE}')

  parser.add_argument("--gen_covariate_file", 
                      type=str, 
                      default=GEN_COVARIATE_FILE, 
                      help=f'The generation covariate file. Default: {GEN_COVARIATE_FILE}')

  parser.add_argument("--kinship_file", 
                      type=str, 
                      default=KINSHIP_FILE,
                      help=f'The kinship covariate file. Default: {KINSHIP_FILE}')

  parser.add_argument("--challenge_file", 
                      type=str, 
                      default=CHALLENGE_FILE,
                      help=f'Age at challenge phenotype. Default: {CHALLENGE_FILE}')

  parser.add_argument("--transform",
                      type=str,
                      default=TRANSFORM,
                      help=f'Type of transform for the phenotype. Default: {TRANSFORM}')

  parser.add_argument("--outlier",
                      type=str_to_bool,
                      default=OUTLIER,
                      help=f'Removes outliers from phenotype_name. Default: {OUTLIER}')
  
  parser.add_argument("--q_low", 
                      type=str, 
                      default=Q_LOW,
                      help=f'The lower quantile range for outlier rejection. Default: {Q_LOW}')

  parser.add_argument("--q_high", 
                      type=str, 
                      default=Q_HIGH,
                      help=f'The upper quantile range for outlier rejection. Default: {Q_HIGH}')

  parser.add_argument("--verbose", 
                      type=str_to_bool, 
                      default=VERBOSE, 
                      const=True, 
                      nargs='?', 
                      help=f'Print messages. Default: {VERBOSE}')

  args = parser.parse_args()
  return args

def rankint(x):
  """
  Rank-based inverse normal transformation.

  Args:
    x (array): Phenotype data.
  
  Returns:
    arr_transformed (array): Transformed phenotype data.
  """

  arr = x[~np.isnan(x)]
  N = arr.size
  arr_ranked = rankdata(arr)
  arr_transformed = norm.ppf(arr_ranked/(N+1))
  return np.expand_dims(arr_transformed, axis=1)

def remove_outliers(phenotype,
                    q_range):
  """
  Remove outliers from the phenotype.

  Args:
    phenotype (object): Phenotype object.
    q_range (list): Quantile range.

  Returns:
    all_data (array): Phenotype data.
    all_samples (list): List of samples.
    N_all_samples (int): Number of samples.
  """
  print("q_low: %s, q_high: %s" % (q_range[0], q_range[1]))
  print(phenotype.all_data.shape)
  df_data = pd.DataFrame([], columns=["mouse_ids", "phenotype"])
  df_data["mouse_ids"] = phenotype.all_samples
  df_data["phenotype"] = phenotype.all_data

  ## quantile range
  q_low, q_high = q_range[0], q_range[1]

  # conver string to numerical value
  if q_low == "None":
    q_low = None
  else:
    q_low = float(q_low)

  if q_high == "None":
    q_high = None
  else:
    q_high = float(q_high)
  
  # apply quantile range
  if (q_low is not None) and (q_high is not None):
    low_val = df_data["phenotype"].quantile(q_low)
    high_val = df_data["phenotype"].quantile(q_high)
    df_data = \
    df_data[(df_data["phenotype"] < high_val) & (df_data["phenotype"] > low_val)]
  elif (q_low is not None) and (q_high is None):
    low_val = df_data["phenotype"].quantile(q_low)
    df_data = df_data[(df_data["phenotype"] > low_val)]
  elif (q_low is None) and (q_high is not None):
    high_val = df_data["phenotype"].quantile(q_high)
    df_data = df_data[(df_data["phenotype"] < high_val)]

  all_data = np.expand_dims(df_data["phenotype"].to_numpy(), axis=1)
  all_samples = df_data["mouse_ids"].to_list()
  N_all_samples = len(phenotype.all_samples)

  return all_data, all_samples, N_all_samples

def determine_state(mode, 
                    phenotype_name):
  """
  Determine the state of the phenotype.

  Args:
    mode (str): Mode of the analysis.
    phenotype_name (str): Name of the phenotype.

  Returns:
    state (str): State of the phenotype.
  """
  # extract t0_days
  tmp = phenotype_name.split('.')
  if tmp[0] == 'weighted':
    tmp = tmp[1:]

  # defualt state
  state = "post"
  if mode == "timeint":
    tmp2 = [num for num in tmp if not num.isalpha()]
    t0_days = int(tmp2[0])
    t1_days = int(tmp2[1])
    if (t0_days == 0) and (t1_days == 180):
      state = "pre"
    else:
      state = "post"
  elif mode == "prepost":
    if tmp[0] == "pre":
      state = "pre"
  elif mode == "resilience":
    if tmp[0] == "resilience":
      state = "post"

  return state


def compute_heritability(genotype,
                         phenotype,
                         covariates,
                         diet_covariate,
                         phenotype_name,
                         state,
                         cached_csvname):
  """
  Compute heritability of the phenotype.

  Args:
    genotype (object): Genotype object.
    phenotype (object): Phenotype object.
    covariates (list): List of covariates.
    diet_covariate (object): Diet covariate object.
    phenotype_name (str): Name of the phenotype.
    state (str): State of the phenotype.
    cached_csvname (str): Cached csv name.
  
  Returns:
    df_csv (dataframe): Dataframe of heritability estimates.
  """

  # create data frame
  cnames = ["total.var", "total.var.serr", "total.pve", "total.pve.serr"]
  for diet in diet_covariate.names:
    cnames.append(diet + ".pve")
    cnames.append(diet + ".pve.serr")
  df_csv = pd.DataFrame(np.nan, index=[phenotype_name], columns=cnames)

  # fit model
  if not os.path.isfile(cached_csvname):

    try: 
      model = gxemm.Gxemm(genotype.kinship,
                          phenotype.data,
                          covariates)
      model.fit_pve(get_serr=True)

      # save in data frame
      if state=='pre':
        df_csv.loc[phenotype_name]["var.serr"] = model.total_var
        df_csv.loc[phenotype_name]["total.var.serr"] = model.total_var_serr
        df_csv.loc[phenotype_name]["total.pve"] = model.total_pve
        df_csv.loc[phenotype_name]["total.pve.serr"] = model.total_pve_serr
      else:
        df_csv.loc[phenotype_name]["var.serr"] = model.total_var
        df_csv.loc[phenotype_name]["total.var.serr"] = model.total_var_serr
        df_csv.loc[phenotype_name]["total.pve"] = model.total_pve
        df_csv.loc[phenotype_name]["total.pve.serr"] = model.total_pve_serr
        for diet, p, p_se in zip(diet_covariate.names, model.pve, model.pve_serr):
          df_csv.loc[phenotype_name][diet+".pve"] = p
          df_csv.loc[phenotype_name][diet+".pve.serr"] = p_se
      df_csv.to_csv(cached_csvname, sep=',')

    except ValueError:
      print("Could not compute heritability.")
      print(f'Saved {cached_csvname}.')
      df_csv.to_csv(cached_csvname, sep=',')
      return

  else:
    # load existing dataframe
    df_csv = pd.read_csv(cached_csvname, sep=',', index_col=[0])

  return df_csv

def fit_gxemm(mode,
              model_dir,
              phenotype_file,
              phenotype_name,
              diet_covariate_file,
              gen_covariate_file,
              kinship_file,
              challenge_file,
              transform,
              q_range,
              outlier,
              verbose):

  # create dataframe
  os.makedirs(model_dir, exist_ok=True)
  if verbose:
    print(f'Storing in {model_dir}')

  # create file
  cached_csvname = f'{model_dir}/{phenotype_name}.csv'
  if transform in ["logit", "log", "sqrt", "rankint"]:
    if outlier:
      cached_csvname = f'{model_dir}/outlier.{transform}.{phenotype_name}.csv'
    else:
      cached_csvname = f'{model_dir}/{transform}.{phenotype_name}.csv'
  else:
    if outlier:
      cached_csvname = f'{model_dir}/outlier.{phenotype_name}.csv'
  
  # save file
  if verbose:
    print(f'Saving results in {cached_csvname}')

  # load diet covariate
  diet_covariate = io.Covariate(test_gxe=True, effect='fixed', gxe=True)
  diet_covariate.load(diet_covariate_file)

  # load samples
  df_phenotype = pd.read_csv(phenotype_file, sep=',')

  # test samples
  test_samples = df_phenotype.MouseID.to_list()

  # load phenotype
  phenotype = io.Phenotype()
  phenotype.load(phenotype_file, phenotype_name)

  # apply transformation
  if transform in ["logit", "log", "sqrt", "rankint"]:
    if transform == "logit":
      phenotype.all_data = np.array(sc.special.logit(phenotype.all_data))
    elif transform == "log":
      phenotype.all_data = np.array(np.log(phenotype.all_data))
    elif transform == "sqrt":
      phenotype.all_data = np.array(np.sqrt(phenotype.all_data))
    elif transform == "rankint":
      phenotype.all_data = np.array(rankint(phenotype.all_data))
  
  # remove outliers if any
  if outlier:
    all_data, all_samples, N_all_samples = \
      remove_outliers(phenotype, q_range)
    phenotype.all_data = all_data
    phenotype.all_samples = all_samples
    phenotype.N_all_samples = N_all_samples

  # load generation covariate
  gen_covariate = io.Covariate(test_gxe=False, effect='fixed', gxe=True)
  gen_covariate.load(gen_covariate_file)

  # load kinship file
  genotype = io.Genotype()
  genotype.load_kinship(kinship_file)

  # determine state and load covariates
  state = determine_state(mode, phenotype_name)
  if state=='pre':
    covariates = [gen_covariate]
  else:
    covariates = [diet_covariate, gen_covariate]

  # subset to common samples
  io.intersect_datasets(genotype, 
                        phenotype, 
                        covariates, 
                        at_samples=test_samples)

  # compute heritability
  df_pve = compute_heritability(genotype=genotype,
                                phenotype=phenotype,
                                covariates=covariates,
                                diet_covariate=diet_covariate,
                                phenotype_name=phenotype_name,
                                state=state,
                                cached_csvname=cached_csvname)


  return df_pve

def main():
  args = parse_args()

  if args.verbose:
    print(args)

  ## determine mode
  tmp = args.phenotype_name.split('.')
  if tmp[0] == "weighted":
    tmp = tmp[1:]
  
  if tmp[0] in ['pre', 'post']:
    mode = "prepost"
    phenotype_file = PREPOST_PHENOTYPE_FILE
  elif tmp[0] in ['resilience']:
    mode = "resilience"
    phenotype_file = RESILEN_PHENOTYPE_FILE
  else:
    mode = "timeint"
    phenotype_file = TIMEINT_PHENOTYPE_FILE

  if args.verbose:
    print(phenotype_file)

  fit_gxemm(mode=mode,
            model_dir=args.model_dir,
            phenotype_file=phenotype_file,
            phenotype_name=args.phenotype_name,
            diet_covariate_file=args.diet_covariate_file,
            gen_covariate_file=args.gen_covariate_file,
            kinship_file=args.kinship_file,
            challenge_file=args.challenge_file,
            transform=args.transform,
            outlier=args.outlier,
            q_range=[args.q_low, args.q_high],
            verbose=args.verbose)

  return

if __name__ == "__main__":
  main()