## Processed Datasets

* `age_phenotype.csv`
  - The file contains the precise age (in days) at which the mouse was assayed for a particular phenotype.

* `bodyweight.csv`
  - The file contains body weight and corresponding latent states. The row indices are explained below:
    * `bw`: Body weight sampled at 10 day interval. Starts at 30 day and ends at 1640 days. Enteries are filled with nans if the mouse has died prior to 1640 days.
    * `states`: Latent state assigned at each time point obtained using a variational-inference based autoregressive hidden Markov model.
    * `transitions`: Mouse-specific transition state probabilities (`tilda_Ps`).
    * `phenotypes`: A two-letter code assigned to particular time point if there is a phenotyping event at the steady state perturbation event. If there is no phenotyping event, then it is a blank string.
    * `homeostasis`: It is the adapation to stress after a specific perturbation event. The value is assigned to the time point which denotes the end of the perturbation event.
    * `events`: Time points that indicate the start of a perturbation event are set to 1.0. Similarly, time points that indicate the end of a perturbation event are set to 2.0.
    * `bw_rec`: Reconstructed body weight obtained using the `tilda_eta` values as the autoregressive coefficients.
    * `pi_x`: The probability of being in the latent state `x`, where `x` represents growth, steady, or decline.
    * `st_i_j`: The probability of transitioning from state `j` to state `i`,  where `i` and `j` represent the latent states at time `t`.

* `diet_covariates.csv`
  - This file contains diet as covariates

* `generation_covariates.csv`
  - This file contains generation as covariates

* `kinship_matrix.genoprob.csv`
  - This file contains the kinship matrix

* `lifespan.csv`
  - This file contains contains mouse ids, date of birth and death, and lifespan (in days).

* `pll.csv`
  - This file contains proportion of life lived body weight-derived phenotypes at every 10% interval.

* `prepost.csv`
  - This file contains pre- and post-intervention phenotypes. The pre-intervention phase starts at 30 days and ends at 180 days. The post-interventions phase starts at 180 days and continues until end of lifespan.

* `timeint.csv`
  - This file contains body weight-derived phenotypes computed at every six-month interval.
