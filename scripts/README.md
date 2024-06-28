## SLURM Scripts

### Run Autoregressive Hidden Markov Model (ARHMM)
The script below creates SLURM jobs for the VIARHMM. Each job created uses a fixed random state and number of latent states. In our analysis, we used 10 different random states and varied the step size from 2 to 10 for each random state. The train and validation models for the various random states and step sizes are stored in the [`models/all_models`](../models/all_models/) folder.  We recomend using the pre-trained models, as training new models from scratch can be time-consuming.
```
(env_name)$ bash scripts/viarhmm.sh
```

### Run Time-Varying Cox PH (TVCPH)
The script below creates SLURM jobs for the time-varying Cox proportional hazard (TVCPH) models. The list of phenotypes included in our analysis are listed in `ls_phenotypes.txt` file. Along with the phenotype, we also pass metadata information which includes the type of phenotype (`mode` = `prepost` or `timeint` or `resilience`) and if the phenotype is weighted or not (a `boolean` value). Some of the phenotypes also account for the uncertainity of the state by using a weighted average approach, where the weights are obtained from the posterior probabilities of the latent state. The results are stored in the [`data/tvcph`](../data/tvcph/) folder.
```
(env_name)$ bash scripts/tvcph.sh
```

### Run Heritability Analysis (H2)
The heritability analysis was done using the Gene x Environment Mixed Model ([GxEMM](https://github.com/calico/do_qtl)). To run the script, you will need to clone the GxEMM package and ensure that the `do_qtl` is in the `do_bwd` folder. The script below creates a SLURM job for each phenotype listed in the `gwas_phenotypes.txt` and computes its heritability. The results are stored in the [`data/heritability`](../data/heritability/) folder.
```
(env_name)$ bash scripts/heritability.sh
```

### Run Genome-Wide Association Studies (GWAS)
The GWAS analysis was also done using the GxEMM package. The script creates a SLURM job for each chromsome and phenotype listed in the `gwas_phenotypes.txt`. The results are stored in the [`data/gwas`](../data/gwas) folder.
```
(env_name)$ bash scripts/gwas.sh
```
After the genetic mapping jobs are completed and the peaks are identified, fine-mapping analysis can be done is a similar way using the GxEMM package. The results are stored in the [`data/fmap`](../data/fmap) folder. We used the [`rqtl2`](https://kbroman.org/qtl2/) package to generate the fine-mapping plots.
```
(env_name)$ bash scripts/finemap.sh
```
