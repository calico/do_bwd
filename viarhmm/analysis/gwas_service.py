from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy as sc

sys.path.append(os.getcwd())

from viarhmm.preprocess import str_to_bool
from viarhmm.analysis.heritability_service import rankint
from viarhmm.analysis.heritability_service import remove_outliers
from viarhmm.analysis.heritability_service import determine_state
from do_qtl.lib.models import gxemm
from do_qtl.lib.models import emma
from do_qtl.lib import data_io as io

import warnings
warnings.filterwarnings("ignore")

## save path of heritability plots
MODEL_DIR = 'data/gwas'
VERBOSE = False

## location of phenotypes files
PREPOST_PHENOTYPE_FILE = 'data/prepost_v1.csv'
TIMEINT_PHENOTYPE_FILE = 'data/timeint_v1.csv'
RESILEN_PHENOTYPE_FILE = 'data/resilience_v1.csv'

## location of genetics files
CHALLENGE_FILE = 'data/age_phenotypes.csv'
DIET_COVARIATE_FILE = 'data/diet_covariates.csv'
GEN_COVARIATE_FILE = 'data/generation_covariates.csv'
KINSHIP_PATH = '/group/diversity_outcross/genetics/jax/geno_data_proc'

## default phenotype name
METHOD = 'gxemm'
PHENOTYPE_NAME = 'weighted.state.occupancy.180.to.360.DS'
CHROMOSOME = '1'
APPROX = True
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

  parser.add_argument("--kinship_path", 
                      type=str, 
                      default=KINSHIP_PATH,
                      help=f'The kinship covariate file path. Default: {KINSHIP_PATH}')

  parser.add_argument("--challenge_file", 
                      type=str, 
                      default=CHALLENGE_FILE,
                      help=f'Age at challenge phenotype. Default: {CHALLENGE_FILE}')

  parser.add_argument("--method", 
                      type=str, 
                      default=METHOD,
                      help=f'The method used for computing GWAS. Default: {METHOD}')

  parser.add_argument("--chromosome", 
                      type=str, 
                      default=CHROMOSOME,
                      help=f'The chromosome number on which GWAS is run. Default: {CHROMOSOME}')

  parser.add_argument("--approx", 
                      type=str_to_bool, 
                      default=APPROX, 
                      const=True, 
                      nargs='?', 
                      help=f'Use approx model. Default: {APPROX}')

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


def compute_gwas(genotype,
                 phenotype,
                 covariates,
                 genoprobs,
                 method,
                 approx,
                 cached_csvname):
  """
  Compute GWAS.

  Args:
    genotype (io.Genotype): Genotype object.
    phenotype (io.Phenotype): Phenotype object.
    covariates (list): List of covariates.
    genoprobs (np.array): Array of genotype probabilities.
    method (str): Method used for computing GWAS.
    approx (bool): Use approx model.
    cached_csvname (str): Name of the cached csv file.

  Returns:
    pd.DataFrame: Dataframe containing the GWAS results.
  """
  # run GWAS
  if not os.path.isfile(cached_csvname):

    try:

      # if method is gxemm
      if (method == 'gxemm'):

        model = gxemm.Gxemm(genotype.kinship,
                            phenotype.data,
                            covariates)

        if ~approx:
          # this creates a generator
          results = model.run_gwas(genoprobs, approx=approx, perm=10)

        else:
          # this creates a generator
          results = model.run_gwas(genoprobs, approx=approx)
      
      # if method is emma
      elif (method == 'emma'):

        model = emma.Emma(genotype.kinship,
                          phenotype.data,
                          covariates)

        if ~approx:
          # this creates a generator
          results = model.run_gwas(genoprobs, approx=approx, perm=10)

        else:
          # this creates a generator
          results = model.run_gwas(genoprobs, approx=approx)


      # output association statistics
      header = ['variant.id'] + \
              ['additive.LOD', 'additive.p.value'] + \
              ['interaction.LOD', 'interaction.p.value']
      
      df_strain = pd.DataFrame([], columns=header)
      for r, result in enumerate(results):
        if len(result) < len(header):
          result = result + [np.nan]*(len(header)-len(result))
        df_strain.loc[r] = result
        
      df_strain.to_csv(cached_csvname, sep=',')

    except ValueError:

      # output association statistics
      header = ['variant.id'] + \
              ['additive.LOD', 'additive.p.value'] + \
              ['interaction.LOD', 'interaction.p.value']
      df_strain = pd.DataFrame([], columns=header)
      print("Could not compute GWAS.")
      df_strain.to_csv(cached_csvname, sep=',')
      return

  else:
    
    df_strain = pd.read_csv(cached_csvname, sep=',', index_col=[0])

  return df_strain

def fit_gxemm(mode,
              model_dir,
              phenotype_file,
              phenotype_name,
              diet_covariate_file,
              gen_covariate_file,
              kinship_path,
              challenge_file,
              chromosome,
              method,
              approx,
              transform,
              outlier,
              q_range,
              verbose):
  """
  Fit the GXEMM model.

  Args:
    mode (str): Mode of the phenotype.
    model_dir (str): Directory to store fit ARHMM models.
    phenotype_file (str): Phenotype file.
    phenotype_name (str): Name of the phenotype of interest.
    diet_covariate_file (str): Covariate file of the diets.
    gen_covariate_file (str): Generation covariate file.
    kinship_path (str): Kinship covariate file path.
    challenge_file (str): Age at challenge phenotype.
    chromosome (str): Chromosome number on which GWAS is run.
    method (str): Method used for computing GWAS.
    approx (bool): Use approx model.
    transform (str): Type of transform for the phenotype.
    outlier (bool): Removes outliers from phenotype_name.
    q_range (list): List containing the lower and upper quantile range.
    verbose (bool): Print messages.

  Returns:
    pd.DataFrame: Dataframe containing the GWAS results.
  """

  # create dataframe
  os.makedirs(model_dir, exist_ok=True)
  if verbose:
    print(f'Storing in {model_dir}')

  # if transform
  fname = phenotype_name
  if transform in ["logit", "log", "sqrt", "rankint"]:
    fname = transform + "." + phenotype_name
    if outlier:
      fname = "outlier." + transform + "." + phenotype_name
  else:
    if outlier:
      fname = "outlier." + phenotype_name

  # check if approx model
  if approx:
    model_dir = os.path.join(model_dir, method + '_approx', fname)
  else:
    model_dir = os.path.join(model_dir, method, fname)

  # create model directory
  os.makedirs(model_dir, exist_ok=True)
  if verbose:
    print(f'Creating directory {model_dir}')

  # create cached_csvname
  cached_csvname = 'gwas_chr_' + chromosome + '.csv'
  if verbose:
    print(f'Storing in {cached_csvname}')
  cached_csvname = os.path.join(model_dir, cached_csvname)

  # load diet covariate
  diet_covariate = io.Covariate(test_gxe=True, effect='fixed', gxe=True)
  diet_covariate.load(diet_covariate_file)

  # load samples
  df_phenotype = pd.read_csv(phenotype_file, sep=',')
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
    print(phenotype.all_data.shape)

  # load generation covariate
  gen_covariate = io.Covariate(test_gxe=False, effect='fixed', gxe=True)
  gen_covariate.load(gen_covariate_file)

  # append kinship_path
  kinship_file = os.path.join(kinship_path,
                              'kinship_loco_200131.Rdata_'+chromosome+'.csv')

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

  # load probability of founder of origin
  genoprob_file = os.path.join(kinship_path,
                               'prob.8state.allele.qtl2_200131.Rdata.'+chromosome+'.csv')
  genoprobs = genotype.load_genoprobs(genoprob_file)

  # compute GWAS
  df_csv = compute_gwas(genotype=genotype,
                        phenotype=phenotype,
                        covariates=covariates,
                        genoprobs=genoprobs,
                        method=method,
                        approx=approx,
                        cached_csvname=cached_csvname)

  return df_csv

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
            kinship_path=args.kinship_path,
            challenge_file=args.challenge_file,
            chromosome=args.chromosome,
            method=args.method,
            approx=args.approx,
            transform=args.transform,
            outlier=args.outlier,
            q_range=[args.q_low, args.q_high],
            verbose=args.verbose)

  return

if __name__ == "__main__":
  main()
