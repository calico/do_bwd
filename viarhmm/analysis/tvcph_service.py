import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

from viarhmm.preprocess import str_to_bool
from viarhmm.analysis.tvcph_utils import phenotype_names
from viarhmm.analysis.tvcph_utils import get_dataframe
from viarhmm.analysis.tvcph_utils import get_resilience_dataframe
from viarhmm.analysis.tvcph_utils import timeint_tvcph_dataframe
from viarhmm.analysis.tvcph_utils import save_params

import warnings
warnings.filterwarnings("ignore")



LS_FILEPATH = 'data/lifespan.csv'
TIMEINT_FILEPATH = 'data/timeint.csv'
PREPOST_FILEPATH = 'data/prepost.csv'
RESILEN_FILEPATH = 'data/resilience.csv'
CH_FILEPATH = 'data/age_phenotype.csv'
BW_FILEPATH = 'data/bodyweight.csv'
MODEL_DIR = 'data/tvcph'
PHENOTYPE_NAME = 'state_occupancy'

MODE = "timeint"
INTERACTION = True
BW_CORRECTION = True
REFERENCE = True
WEIGHTED = True
TRANSFORM = False
VERBOSE = True
MODEL_TYPE = 'tvcph'


def parse_args():
  """[summary]

  Returns:
      [type]: [description]
  """

  parser = argparse.ArgumentParser(
    description="Fit TVCPH models for JAX bodyweight data.")

  parser.add_argument("--mode", 
                      type=str, 
                      default=MODE,
                      help=f'Phenotype mode. Default: {MODE}')


  parser.add_argument("--bw_filepath", 
                      type=str, 
                      default=BW_FILEPATH,
                      help=f'Path of the bodyweight meausrements. Default: {BW_FILEPATH}')

  parser.add_argument("--ls_filepath", 
                      type=str, 
                      default=LS_FILEPATH,
                      help=f'Path of the lifespan meausrements. Default: {LS_FILEPATH}')

  parser.add_argument("--ch_filepath", 
                      type=str, 
                      default=CH_FILEPATH,
                      help=f'Path of the challenge based phenotypes. Default: {CH_FILEPATH}')

  parser.add_argument("--phenotype_name", 
                      type=str, 
                      default=PHENOTYPE_NAME,
                      help=f'Phenotype of interest for survival analysis. Default: {PHENOTYPE_NAME}')

  parser.add_argument("--interaction", 
                      type=str_to_bool, 
                      default=INTERACTION, 
                      const=True, nargs='?', 
                      help=f'Add interaction terms. Default: {INTERACTION}')

  parser.add_argument("--weighted", 
                      type=str_to_bool, 
                      default=WEIGHTED, 
                      const=True, nargs='?', 
                      help=f'Use weighted phenotypes. Default: {WEIGHTED}')

  parser.add_argument("--bw_correction", 
                      type=str_to_bool, 
                      default=BW_CORRECTION, 
                      const=True, nargs='?', 
                      help=f'Add body weight correction. Default: {BW_CORRECTION}')

  parser.add_argument("--reference", 
                      type=str_to_bool, 
                      default=REFERENCE, 
                      const=True, nargs='?', 
                      help=f'Use AL as reference group. Default: {REFERENCE}')

  parser.add_argument("--model_dir", 
                      type=str, 
                      default=MODEL_DIR,
                      help=f'Directory to store fit CPH models. Default: {MODEL_DIR}')

  parser.add_argument("--model_type", 
                      type=str, 
                      default=MODEL_TYPE,
                      help=f'Survival model type. Default: {MODEL_TYPE}')

  parser.add_argument("--transform",
                      type=str,
                      default=TRANSFORM,
                      help=f'Type of transform for the phenotype. Default: {TRANSFORM}')

  parser.add_argument("--verbose", 
                      type=str_to_bool, 
                      default=VERBOSE, 
                      const=True, 
                      nargs='?', 
                      help=f'Print messages. Default: {VERBOSE}')

  args = parser.parse_args()
  return args


def fit_tvcph(df_tvcph):
  """Fit time varying Cox PH model

  Args:
      df_tvcph (pandas): dataframe crafted for tvcph model

  Returns:
      tvcph (object): contains fitted model parameters
      status (bool): True if model was fitted, False otherwise
  """
  # set defaul status
  status = True

  # Get unique mice with their diet
  unique_mice = df_tvcph[['mouse_id', 'diet']].drop_duplicates()

  # Stratified train-test split based on diet
  train_ids, val_ids = train_test_split(
      unique_mice['mouse_id'], 
      test_size=0.2, 
      random_state=42, 
      stratify=unique_mice['diet']
  )

  # Subset the full dataset based on train and validation IDs
  train_df = df_tvcph[df_tvcph['mouse_id'].isin(train_ids)]
  val_df = df_tvcph[df_tvcph['mouse_id'].isin(val_ids)]

  # Verify the distribution of diets in both sets
  print("Diet distribution in training set:")
  print(train_df['diet'].value_counts())
  train_df = train_df.drop(columns=["diet"])

  print("\nDiet distribution in validation set:")
  print(val_df['diet'].value_counts())
  val_df = val_df.drop(columns=["diet"])

  # Range of penalizer values to test
  penalizer_values = np.logspace(-4, 1, 10)

  # Dictionary to store penalizer and corresponding c-index
  results = {}

  for penalizer in penalizer_values:
    # Fit the model with the current penalizer
    ctv = CoxTimeVaryingFitter(penalizer=penalizer)
    ctv.fit(train_df,
            id_col="mouse_id",
            event_col="status",
            start_col="start",
            stop_col="stop",
            show_progress=True)
    
    # Predict partial hazard on validation set
    partial_hazard = ctv.predict_partial_hazard(val_df)
    
    # Calculate c-index on validation set
    c_index = concordance_index(
        val_df['stop'],  # Observed times
        -partial_hazard.squeeze(),  # Negative hazards for proper concordance calculation
        val_df['status']             # Event indicators
    )
    results[penalizer] = c_index

  # Find the best penalizer
  best_penalizer = max(results, key=results.get)

  # Print results
  print("Penalizer values and corresponding c-index:")
  for penalizer, c_index in results.items():
    print(f"Penalizer: {penalizer}, C-index: {c_index}")
  print(f"\nBest penalizer: {best_penalizer}, Highest C-index: {results[best_penalizer]}")

  # Fit the model with the best penalizer
  df_tvcph = df_tvcph.drop(columns=["diet"])
  tvcph = CoxTimeVaryingFitter(penalizer=best_penalizer)
  try:
    tvcph.fit(df_tvcph,
              id_col="mouse_id",
              event_col="status",
              start_col="start",
              stop_col="stop",
              show_progress=True)
    tvcph.print_summary()
  except ValueError:
    print("\n Could not compute TVCPH.")
    status = False
    return [], status

  return tvcph, status


def fit_survival(model_dir,
                 mode,
                 phenotype_name,
                 bw_filepath,
                 ls_filepath,
                 ch_filepath,
                 ph_filepath,
                 bw_correction,
                 reference,
                 weighted,
                 interaction,
                 transform,
                 model_type,
                 verbose):
  """_summary_

  Args:
      model_dir (_type_): _description_
      mode (_type_): _description_
      phenotype_name (_type_): _description_
      bw_filepath (_type_): _description_
      ls_filepath (_type_): _description_
      ch_filepath (_type_): _description_
      ph_filepath (_type_): _description_
      bw_correction (_type_): _description_
      reference (_type_): _description_
      weighted (_type_): _description_
      interaction (_type_): _description_
      transform (_type_): _description_
      model_type (_type_): _description_
      verbose (_type_): _description_
  """

  # cached model name
  cached_model_filename = mode + '_' + phenotype_name

    # change model directory name based on settings
  if bw_correction:
    cached_model_filename = cached_model_filename + '_bw'

  if reference:
    cached_model_filename = cached_model_filename + '_ref'

  if weighted:
    cached_model_filename = cached_model_filename + '_wgt'

  if interaction:
    cached_model_filename = cached_model_filename + '_xterm'

  if transform:
    cached_model_filename = cached_model_filename + '_' + transform

  # load data
  df_bw = pd.read_csv(bw_filepath, sep=',', index_col=[0,1])
  df_ls = pd.read_csv(ls_filepath)
  df_ls["Status"] = df_ls["Status"].apply(lambda x: int(x == "Dead"))
  df_in = pd.read_csv(ph_filepath, sep=',')
  df_ch = pd.read_csv(ch_filepath, sep=',')

  # creates a list of phenotypes depending on the mode
  phenotypes, events = phenotype_names(mode=mode,
                                       phenotype_name=phenotype_name,
                                       weighted=weighted)

  # add scale
  scale = False
  if phenotype_name in ['state_occupancy', 'state_transitions',
                        'max_abs_growthrate_percent']:
    scale = True

  
  # for every phenotype extract the phenotype dataframe
  for phenotype_list, event in zip(phenotypes, events):

    cached_model = cached_model_filename + '_' + event + '.pkl'
    cached_model = os.path.join(model_dir, cached_model)
    # print(cached_model)

    # if cached_model_filename does not exist
    if not os.path.isfile(cached_model):

      if (mode == "prepost") or (mode == "timeint"):
        df_ph = get_dataframe(df_input=df_in,
                              phenotype_list=phenotype_list,
                              transform=transform)

        if (mode == "timeint"):
          time_intervals = np.arange(0, 1260+1, 180)
          df_tvcph = timeint_tvcph_dataframe(df_ph=df_ph,
                                            df_bw=df_bw,
                                            df_ls=df_ls,
                                            time_intervals=time_intervals,
                                            reference=reference,
                                            bw_correction=bw_correction,
                                            interaction=interaction,
                                            scale=scale)
        else:
          time_intervals = np.array([0, 180, 1600])
          df_tvcph = timeint_tvcph_dataframe(df_ph=df_ph,
                                            df_bw=df_bw,
                                            df_ls=df_ls,
                                            time_intervals=time_intervals,
                                            reference=reference,
                                            bw_correction=bw_correction,
                                            interaction=interaction,
                                            scale=scale)


      elif mode == "resilience":
        print(phenotype_list)
        # time_intervals = np.array([0, 180, 540, 900])
        time_intervals = np.array([0, 180, 1600])
        df_ph = get_resilience_dataframe(df_input=df_in,
                                         df_ch=df_ch,
                                         phenotype_list=phenotype_list)
        df_tvcph = timeint_tvcph_dataframe(df_ph=df_ph,
                                          df_bw=df_bw,
                                          df_ls=df_ls,
                                          time_intervals=time_intervals,
                                          reference=reference,
                                          bw_correction=bw_correction,
                                          interaction=interaction,
                                          scale=scale)
      else:
        raise ValueError("Mode not found!")

      # fit tvcph
      tvcph_model, status = fit_tvcph(df_tvcph)
      save_params(phenotype_list, 
                  tvcph_model,
                  status,
                  time_intervals, 
                  model_dir)

      # save model
      with open(cached_model, 'wb') as f:
        pickle.dump(tvcph_model, f)

    else:
      with open(cached_model, 'rb') as f:
        tvcph_model = pickle.load(f)

    if hasattr(tvcph_model, 'log_likelihood_'):
      if verbose:
        tvcph_model.print_summary()

  return

def main():

  args = parse_args()
  print(args)

  os.makedirs(args.model_dir, exist_ok=True)
  print(f'Storing in {args.model_dir}')

  # choose phenotype filepath
  if args.mode == "prepost":
    ph_filepath = PREPOST_FILEPATH
  elif args.mode == "resilience":
    ph_filepath = RESILEN_FILEPATH
  elif args.mode == "timeint":
    ph_filepath = TIMEINT_FILEPATH
  else:
    raise ValueError("Phenotype processing mode not found!")

  if args.verbose:
    print(ph_filepath)

  fit_survival(model_dir=args.model_dir,
               mode=args.mode,
               phenotype_name=args.phenotype_name,
               bw_filepath=args.bw_filepath,
               ls_filepath=args.ls_filepath,
               ch_filepath=args.ch_filepath,
               ph_filepath=ph_filepath,
               bw_correction=args.bw_correction,
               reference=args.reference,
               weighted=args.weighted,
               interaction=args.interaction,
               transform=args.transform,
               model_type=args.model_type,
               verbose=args.verbose)
  return


if __name__ == "__main__":
  main()
