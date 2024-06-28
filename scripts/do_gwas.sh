#!/bin/bash
export PYTHONPATH=.
python viarhmm/analysis/gwas_service.py --phenotype_name $1 --chromosome $2 --verbose True
