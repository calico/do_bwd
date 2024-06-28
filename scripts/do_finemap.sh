#!/bin/bash
export PYTHONPATH=.
python viarhmm/analysis/finemap_service.py --phenotype_name $1 --chr $2 --start $3 --end $4 --verbose True
