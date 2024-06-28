
#!/bin/bash

# Input file format: string number float float
input_file="viarhmm/scripts/fm_phenotypes.txt"

# Define the interval size for the for loop
interval=0.1  # Adjust this value as needed

# Read values from the input file and process
while IFS=" " read -r phenotype chr start end; do
  echo "Phenotype: $phenotype, Chr: $chr, Start: $start, End: $end"

  # Calculate the number of iterations based on the interval size
  num_iterations=$(awk -v start="$start" -v end="$end" -v interval="$interval" 'BEGIN { print int((end - start) / interval) }')

  # Run a for loop with the specified interval size
  start_value=$start
  for ((i = 0; i <= num_iterations-1; i++)); do
    current_value=$(echo "$start + $(($i+1)) * $interval" | bc)
    echo "Iteration $i: $start_value $current_value"
    sbatch --nodes 1 --ntasks-per-node 1 --mem 8G --cpus-per-task 1 --partition standard scripts/do_finemap.sh $phenotype $chr $start_value $current_value
    # assign start_value to current_value for the next iteration
    start_value=$current_value
    # Add your desired actions using $current_value
  done
  
  echo "Iteration $(($i+1)): $start_value $end"
  sbatch --nodes 1 --ntasks-per-node 1 --mem 8G --cpus-per-task 1 --partition standard viarhmm/scripts/do_finemap.sh $phenotype $chr $start_value $end

done < "$input_file"
