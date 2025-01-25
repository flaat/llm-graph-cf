#!/bin/bash

# Define the models, datasets, and explainers to evaluate
models=("qwen_14BQ_Q4_GPTQ" "mistral_7B" "llama_3B" "llama_8B" "smollm2_2B" "phi_4B")
datasets=("aids")
explainers=("gnnexplainer")
echo "Model Name | FTC | CTC | NDIFF | EDIFF"
# Loop through each combination of model, dataset, and explainer
for dataset in "${datasets[@]}"; do
    for explainer in "${explainers[@]}"; do
        echo "Explainer: $explainer"

        for model in "${models[@]}"; do    
            python3 src/evaluation.py --model_name "$model" --dataset_name "$dataset" --explainer "$explainer" --task "graph"
        done
    done
done