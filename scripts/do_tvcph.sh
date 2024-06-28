#!/bin/bash
export PYTHONPATH=.
python viarhmm/analysis/tvcph_service.py --phenotype_name $1 --mode $2 --weighted $3 