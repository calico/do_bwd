#!/bin/bash
export PYTHONPATH=.

while read line; do
	for k in $(seq 1 19); do
		sbatch --nodes 1 --ntasks-per-node 1 --mem 10G --cpus-per-task 1 --partition standard scripts/do_gwas.sh $line $k
	done
	sbatch --nodes 1 --ntasks-per-node 1 --mem 10G --cpus-per-task 1 --partition standard scripts/do_gwas.sh $line X
done < scripts/gwas_phenotypes.txt
