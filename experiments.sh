#!/bin/bash

# Array of parameter values to use in the Python script
parameter_values=(0.5 1.5 3 7 14)

# Loop through each parameter value
for value in "${parameter_values[@]}"
do
  echo "Running main.py with parameters=$value"
  # Run the Python script with the current parameter value
  python3 main.py --parameters "$value"  --explainer cf-gnn --dataset citeseer
  wait
  # Wait for the Python script to finish before continuing to the next iteration
done

echo "All runs completed."