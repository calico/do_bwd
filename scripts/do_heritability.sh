#!/bin/bash
export PYTHONPATH=.
python viarhmm/analysis/heritability_service.py --phenotype_name $1 --verbose True
