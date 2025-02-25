from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import pickle
import csv
import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import rankdata, norm

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
MODEL_DIR = './finemap_parallel'
VERBOSE = False

## location of phenotypes files
PREPOST_PHENOTYPE_FILE = 'data/prepost_v1.csv'
TIMEINT_PHENOTYPE_FILE = 'data/timeint_v1.csv'
RESILEN_PHENOTYPE_FILE = 'data/resilience_v1.csv'

## location of genetics files
CHALLENGE_FILE = 'data/age_phenotype.csv'
DIET_COVARIATE_FILE = 'data/diet_covariates.csv'
GEN_COVARIATE_FILE = 'data/generation_covariates.csv'
KINSHIP_PATH = '/group/diversity_outcross/genetics/jax/geno_data_proc'

## default phenotype name
METHOD = 'gxemm'
PHENOTYPE_NAME = 'weighted.post.state.occupancy.GS'
CHROMOSOME = '7'
APPROX = True
TRANSFORM = "rankint"
OUTLIER = False
START = 74.088491
END = 74.188491
Q_LOW = None
Q_HIGH = None


def parse_args():
  """Pass arguments

  Returns:
      args ([dict]): Contains the arguments and values associated
  """
  parser = argparse.ArgumentParser(
    description="Fit hertiability on JAX body weight features."
  )

  parser.add_argument("--model_dir", 
                      type=str, 
                      default=MODEL_DIR, 
                      help=f'Directory to store fit ARHMM models. Default: {MODEL_DIR}'
  )

  parser.add_argument("--phenotype_name", 
                      type=str, 
                      default=PHENOTYPE_NAME, 
                      help=f'Name of the phenotype of interest. Default: {PHENOTYPE_NAME}'
  )

  parser.add_argument("--diet_covariate_file", 
                      type=str,
                      default=DIET_COVARIATE_FILE, 
                      help=f'The covariate file of the diets. Default: {DIET_COVARIATE_FILE}'
  )

  parser.add_argument("--gen_covariate_file", 
                      type=str, 
                      default=GEN_COVARIATE_FILE, 
                      help=f'The generation covariate file. Default: {GEN_COVARIATE_FILE}'
  )

  parser.add_argument("--kinship_path", 
                      type=str, 
                      default=KINSHIP_PATH,
                      help=f'The kinship covariate file. Default:{KINSHIP_PATH}'
  )

  parser.add_argument("--challenge_file", 
                      type=str, 
                      default=CHALLENGE_FILE,
                      help=f'Age at challenge phenotype. Default: {CHALLENGE_FILE}'
  )

  parser.add_argument("--method", 
                      type=str, 
                      default=METHOD,
                      help=f'The method for computing GWAS. Default: {METHOD}'
  )

  parser.add_argument("--chromosome", 
                      type=str, 
                      default=CHROMOSOME,
                      help=f'The chromosome number. Default: {CHROMOSOME}'
  )

  parser.add_argument("--approx", 
                      type=str_to_bool, 
                      default=APPROX, 
                      const=True, 
                      nargs='?', 
                      help=f'Use approx model. Default: {APPROX}'
  )

  parser.add_argument("--transform",
                      type=str,
                      default=TRANSFORM,
                      help=f'Type of transform for the phenotype. Default: {TRANSFORM}')

  parser.add_argument("--outlier",
                      type=str_to_bool,
                      default=OUTLIER,
                      help=f'Removes outliers from phenotype_name. Default: {OUTLIER}')

  parser.add_argument("--start", 
                      type=float, 
                      default=START, 
                      help=f'Start region of the fine mapping. Default:{START}'
  )

  parser.add_argument("--end", 
                      type=float, 
                      default=END,
                      help=f'End region of the fine mapping. Default: {END}'
  )

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
                      help=f'Display print statements. Default: {VERBOSE}'
  )

  args = parser.parse_args()
  return args


def run_finemap(genotype,
                phenotype,
                covariates,
                genotype_file,
                gen_covariate,
                diet_covariate,
                chromosome,
                start,
                end,
                method,
                approx,
                state,
                cached_csvname):
  """_summary_

  Args:
      genotype (_type_): _description_
      phenotype (_type_): _description_
      covariates (_type_): _description_
      genotypes (_type_): _description_
      gen_covariate (_type_): _description_
      diet_covariate (_type_): _description_
      method (_type_): _description_
      approx (_type_): _description_
      cached_csvname (_type_): _description_

  Returns:
      _type_: _description_
  """

  # output finemapping statistics and effect sizes
  if state == 'pre':
    header = ['chromosome:location'] + \
              ['additive.LOD', 'additive.p.value'] + \
              ['additive.intercept'] + \
              ['additive.effect.size.%s'%diet for diet in diet_covariate.names[1:]] + \
              ['additive.effect.size.%s'%gen for gen in gen_covariate.names[1:]] + \
              ['additive.effect.size.allele'] + \
              ['additive.intercept.serr'] + \
              ['additive.effect.size.serr.%s'%diet for diet in diet_covariate.names[1:]] + \
              ['additive.effect.size.serr.%s'%gen for gen in gen_covariate.names[1:]] + \
              ['additive.effect.size.serr.allele']
  else:
    header = ['chromosome:location'] + \
              ['additive.LOD', 'additive.p.value'] + \
              ['additive.intercept'] + \
              ['additive.effect.size.%s'%diet for diet in diet_covariate.names[1:]] + \
              ['additive.effect.size.%s'%gen for gen in gen_covariate.names[1:]] + \
              ['additive.effect.size.allele'] + \
              ['additive.intercept.serr'] + \
              ['additive.effect.size.serr.%s'%diet for diet in diet_covariate.names[1:]] + \
              ['additive.effect.size.serr.%s'%gen for gen in gen_covariate.names[1:]] + \
              ['additive.effect.size.serr.allele'] + \
              ['interaction.LOD', 'interaction.p.value'] + \
              ['interaction.intercept'] + \
              ['interaction.effect.size.%s'%diet for diet in diet_covariate.names[1:]] + \
              ['interaction.effect.size.%s'%gen for gen in gen_covariate.names[1:]] + \
              ['interaction.effect.size.allele'] + \
              ['interaction.effect.size.allele_x_%s'%diet for diet in diet_covariate.names[1:]] + \
              ['interaction.intercept.serr'] + \
              ['interaction.effect.size.serr.%s'%diet for diet in diet_covariate.names[1:]] + \
              ['interaction.effect.size.serr.%s'%gen for gen in gen_covariate.names[1:]] + \
              ['interaction.effect.size.serr.allele'] + \
              ['interaction.effect.size.serr.allele_x_%s'%diet for diet in diet_covariate.names[1:]]

  # run fine map
  if not os.path.isfile(cached_csvname):

    genotypes = genotype.load_genotypes(genotype_file, 
                                        chromosome=chromosome, 
                                        start=start, 
                                        end=end)

    # if method is gxemm
    if (method == 'gxemm'):

      model = gxemm.Gxemm(genotype.kinship,
                          phenotype.data,
                          covariates)

      results = model.run_finemap(genotypes, 
                                  approx=approx, 
                                  perm=10)

    elif (method == 'emma'):

      model = emma.Emma(genotype.kinship,
                        phenotype.data,
                        covariates)

      results = model.run_finemap(genotypes, 
                                  approx=approx,
                                  perm=10)

    with open(cached_csvname, 'w', buffering=1, newline='') as csvfile:
      handle = csv.writer(csvfile)
      handle.writerow(header)
      for result in results:
        handle.writerow(result)

  else:

    # identify, new_start as last variant analysed
    new_start = 0
    with open(cached_csvname, 'r') as csvfile:
      handle = csv.reader(csvfile)
      for last_line in handle:
        pass
      new_start = int(last_line[0].split(':')[1])+1

    # load genotype
    genotypes = genotype.load_genotypes(genotype_file, 
                                        chromosome=chromosome, 
                                        start=start, 
                                        end=end)

    remain = len([g for g in genotypes])
    if remain==0:
      sys.exit(0)
    genotypes = genotype.load_genotypes(genotype_file, 
                                        chromosome=chromosome, 
                                        start=new_start, 
                                        end=end)

    # if method is gxemm
    if (method == 'gxemm'):

      model = gxemm.Gxemm(genotype.kinship,
                          phenotype.data,
                          covariates)

      results = model.run_finemap(genotypes, 
                                  approx=approx, 
                                  perm=10)

    elif (method == 'emma'):

      model = emma.Emma(genotype.kinship,
                        phenotype.data,
                        covariates)

      results = model.run_finemap(genotypes, 
                                  approx=approx,
                                  perm=10)

    with open(cached_csvname, 'a+', buffering=1, newline='') as csvfile:
      # apply to remaining variants
      handle = csv.writer(csvfile)
      for result in results:
        handle.writerow(result)

  return 

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
              chr_loc,
              verbose):
  """_summary_

  Args:
      mode (_type_): _description_
      model_dir (_type_): _description_
      phenotype_file (_type_): _description_
      phenotype_name (_type_): _description_
      diet_covariate_file (_type_): _description_
      gen_covariate_file (_type_): _description_
      kinship_path (_type_): _description_
      challenge_file (_type_): _description_
      chromosome (_type_): _description_
      method (_type_): _description_
      approx (_type_): _description_
      transform (_type_): _description_
      outlier (_type_): _description_
      q_range (_type_): _description_
      chr_loc (_type_): _description_
      verbose (_type_): _description_

  Returns:
      _type_: _description_
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
  cached_csvname = 'gwas_chr_' + chromosome + '_start_' + str(chr_loc[0]) \
    + '_end_' + str(chr_loc[1]) + '.csv'
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

  # load genotypes, in a given locus
  genotype_file = os.path.join(kinship_path,
                               'chr' + chromosome +'_imputed.tab.gz')
  start = np.int64(np.float64(chr_loc[0]) * 1e6) 
  end = np.int64(np.float64(chr_loc[1]) * 1e6)
  print(start, end)
  if chromosome != 'X':
    chromosome = int(chromosome)

  # compute GWAS
  run_finemap(genotype=genotype,
              phenotype=phenotype,
              covariates=covariates,
              genotype_file=genotype_file,
              gen_covariate=gen_covariate,
              diet_covariate=diet_covariate,
              chromosome=chromosome,
              start=start,
              end=end,
              method=method,
              approx=approx,
              state=state,
              cached_csvname=cached_csvname)

  return

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
            chr_loc=[args.start, args.end],
            verbose=args.verbose)

  return

if __name__ == "__main__":
  main()
