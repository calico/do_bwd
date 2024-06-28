export PYTHONPATH=.
for s in $(seq 0 9); do
	for k in $(seq 2 10); do
		sbatch --nodes 1 --ntasks-per-node 1 --mem 8G --cpus-per-task 1 --partition standard scripts/do_viarhmm.sh $k $(($k+1)) $s
	done
done
