from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse 
from functools import partial

sys.path.append(os.getcwd())

from viarhmm.preprocess import str_to_bool
from viarhmm.preprocess import load_data
from viarhmm.preprocess import split_train_validation
from viarhmm.preprocess import train_viarhmm
from viarhmm.preprocess import viarhmm_train_fn
from viarhmm.preprocess import validate_viarhmm
from viarhmm.preprocess import viarhmm_validation_fn

from viarhmm.settings import START_K, STOP_K, STEP_K, START_M, STOP_M, STEP_M, ERR_SIGMA
from viarhmm.settings import TRAIN_FRACTION, LEN_THRESHOLD, L1_TREND, TRIM_PRE
from viarhmm.settings import SEED, MODEL_DIR, DB_NAME

import warnings
warnings.filterwarnings("ignore")

def parse_args():
  """Pass arguments

  Returns:
      args ([dict]): Contains the arguments and values associated
  """
  parser = argparse.ArgumentParser(
    description="Fit ARHMM models for JAX body weight data."
  )
  parser.add_argument("--db_name", type=str, default=DB_NAME,
    help=f'Name of the raw measurements database. Default: {DB_NAME}')
  parser.add_argument("--model_dir", type=str, default=MODEL_DIR,
    help=f'Directory to store fit ARHMM models. Default: {MODEL_DIR}')
  parser.add_argument("--start_k", type=int, default=START_K,
    help=f'Starting number of hidden states. Default: {START_K}')
  parser.add_argument("--step_k", type=int, default=STEP_K,
    help=f'Increment in the number of hidden states. Default: {STEP_K}')
  parser.add_argument("--stop_k", type=int, default=STOP_K,
    help=f'Bound on the number of hidden states. Default: {STOP_K}')
  parser.add_argument("--start_m", type=int, default=START_M,
    help=f'Starting number of the lag order. Default: {START_M}')
  parser.add_argument("--step_m", type=int, default=STEP_M,
    help=f'Increment in the lag order. Default: {STEP_M}')
  parser.add_argument("--stop_m", type=int, default=STOP_M,
    help=f'Bound on the lag order. Default: {STOP_M}')
  parser.add_argument("--train_fraction", type=float, default=TRAIN_FRACTION,
    help=f'Fraction of training set used for ARHMM fitting, rest is used for '
        f'model selection. Default: {TRAIN_FRACTION}')
  parser.add_argument("--len_threshold", type=int, default=LEN_THRESHOLD,
    help=f'Bound on the length of the body weight segment. Default: {LEN_THRESHOLD}')
  parser.add_argument("--l1_trend", type=str_to_bool, default=L1_TREND, const=True, nargs='?',
	  help=f'Use l1 trend filtered measurements. Default: {L1_TREND}')
  parser.add_argument("--trim_pre", type=str_to_bool, default=TRIM_PRE, const=True, nargs='?',
	    help=f'Trim pre-intervention when training. Default: {TRIM_PRE}')
  parser.add_argument("--seed", type=int, default=SEED,
    help=f'Random seed for hmm initialization. Default: {SEED}')
  parser.add_argument("--err_sigma", type=float, default=ERR_SIGMA,
    help=f'Threshold value of error sigma square. Default: {ERR_SIGMA}')

  args = parser.parse_args()
  return args


def fit_arhmm(db_name, 
              start_k, 
              step_k, 
              stop_k, 
              start_m, 
              step_m, 
              stop_m,
              train_fraction, 
              model_dir, 
              seed, 
              len_threshold, 
              l1_trend,
              trim_pre,
              err_sigma):
  """Fit ARHMM models for body weight data

  Args:
      db_name (str): Name of the database
      start_k (int): Starting number of hidden states
      step_k (int): Increment in the number of hidden states
      stop_k (int): Bound on the number of hidden states
      start_m (int): Starting number of the lag order
      step_m (int): Increment in the lag order
      stop_m (int): Bound on the lag order
      train_fraction (float): Fraction of training set used for ARHMM fitting
      model_dir (str): Directory to store fit ARHMM models
      seed (int): Random seed for hmm initialization
      len_threshold (int): Bound on the length of the body weight segment
      l1_trend (bool): Use l1 trend filtered measurements
      trim_pre (bool): Trim pre-intervention when training
      err_sigma (float): Threshold value of error sigma square

  Returns:
      [tuple]: Tuple of viarhmm models and validation models
      
  """
  
  print(f'Using body weight measurements from {db_name}.')
  print(f'Fititng MSAR models hidden states k=range(start={start_k}, stop={stop_k}, '
        f'step={step_k}) and lag orders m=range(start={start_m}, stop={stop_m}, '
        f'step={step_m}).')
  ks = range(start_k, stop_k, step_k)
  ms = range(start_m, stop_m, step_m)
  print(f'Fitting models for hidden states: {ks}')
  print(f'Fitting models for lag orders: {ms}')
  os.makedirs(model_dir, exist_ok=True)
  print(f'Storing in {model_dir}')

  # load dataframes
  df_raw_bw, df_l1f_bw, _ = load_data(db_name, l1_trend)

  # select the dataframe of interest
  df_bw = df_raw_bw.copy()
  if l1_trend:
    df_bw = df_l1f_bw.copy()

  print(f'Splitting into train and val data with train fraction={train_fraction}')
  tdatas, vdatas, ttags, vtags, n_train, \
    n_validation, diets = split_train_validation(df_bw,
                                                len_threshold=len_threshold,
                                                train_fraction=train_fraction,
                                                random_state=seed)

  # begin training
  print("Length of triaining: %d"%(len(tdatas)))
  print("Length of validation: %d"%(len(vdatas)))
  print("Starting VIARHMM training process")
  _train_fn = partial(viarhmm_train_fn, random_state=seed)
  viarhmm_models = train_viarhmm(viarhmm_train_fn=_train_fn,
                                 ks=ks,
                                 ms=ms,
                                 tdatas=tdatas,
                                 inputs=None,
                                 masks=None,
                                 ttags=ttags,
                                 model_dir=model_dir,
                                 threshold=err_sigma)
  
  print("Extracting states for validation set")
  _validation_fn = partial(viarhmm_validation_fn, random_state=seed)
  validation_models = validate_viarhmm(viarhmm_validation_fn=_validation_fn,
                                       models=viarhmm_models[0],
                                       ks=ks,
                                       ms=ms,
                                       vdatas=vdatas,
                                       inputs=None,
                                       masks=None,
                                       vtags=vtags,
                                       model_dir=model_dir,
                                       threshold=err_sigma)

  return viarhmm_models, validation_models


def main():
  args = parse_args()
  print(args)
  model_dir = os.path.join(args.model_dir, 'seed_' + str(args.seed))

  if args.err_sigma:
    model_dir = os.path.join(model_dir, 'threshold_' + str(args.err_sigma))
                            
  fit_arhmm(db_name=args.db_name,
            start_k=args.start_k,
            step_k=args.step_k,
            stop_k=args.stop_k,
            start_m=args.start_m,
            step_m=args.step_m,
            stop_m=args.stop_m,
            train_fraction=args.train_fraction,
            model_dir=model_dir,
            len_threshold=args.len_threshold,
            l1_trend=args.l1_trend,
            trim_pre=args.trim_pre,
            err_sigma=args.err_sigma,
            seed=args.seed)
  return

if __name__ == "__main__":
    main()