#!/bin/bash

# Additional parameters for the new script
models=("mistral_7B" "llama_3B" "llama_8B" "smollm2_2B" "phi_4B" "qwen_0.5B_Q4_GPTQ" "qwen_1.5B_Q4_GPTQ" "qwen_3B_Q4_GPTQ" "qwen_7B_Q4_GPTQ" "qwen_14B_Q4_GPTQ")
explainers=("gnnexplainer")
max_model_lens=(8192 8192 8192 8192 8192 8192 8192 8192 8192 8192)

# Loop through each model and explainer
for i in "${!models[@]}"
do
  for explainer in "${explainers[@]}"
  do
    echo "Running main.py with model=${models[$i]}, explainer=$explainer, max_model_len=${max_model_lens[$i]}"
    # Run the Python script with the current model, explainer, and max_model_len
    python3 main.py --model_name "${models[$i]}" --dataset aids --explainer "$explainer" --max_model_len "${max_model_lens[$i]}" --max_tokens 4096
    wait
    # Wait for the Python script to finish before continuing to the next iteration
  done
done

echo "All runs completed."