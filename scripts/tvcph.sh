#!/bin/bash
export PYTHONPATH=.
while IFS=" " read phenotype_name mode weighted ; do
	sbatch --nodes 1 --ntasks-per-node 1 --mem 4G --cpus-per-task 1 --partition standard scripts/do_tvcph.sh $phenotype_name $mode $weighted
done < scripts/ls_phenotypes.txt
