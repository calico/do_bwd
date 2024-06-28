#!/bin/bash
export PYTHONPATH=.
python viarhmm/analysis/arhmm_service.py --start_k $1 --stop_k $2 --seed $3 --err_sigma 2.0
