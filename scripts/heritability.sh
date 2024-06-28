#!/bin/bash
export PYTHONPATH=.
while read line; do
	sbatch --nodes 1 --ntasks-per-node 1 --mem 8G --cpus-per-task 1 --partition standard scripts/do_heritability.sh $line
done < scripts/gwas_phenotypes.txt
