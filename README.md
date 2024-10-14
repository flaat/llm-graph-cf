# Natural Language Counterfactual Explanations for Graphs Using Large Language Models

## Overview

This script is designed to generate explanations using a language model based on specified parameters and datasets. It allows users to customize various aspects of the explanation generation process through command-line arguments.


## Requirements

- **Python 3.12**
- Required Python packages (install using `pip install -r requirements.txt` if a requirements file is provided)

## Usage

Run the script from the command line:

```bash
python main.py [options]
```


## Command-Line Arguments

The script accepts the following command-line arguments:

### `--parameters`

- **Type:** `str`
- **Default:** `'0.5'`
- **Description:** Parameter value used in the explanation generation process.

### `--temperature`

- **Type:** `float`
- **Default:** `0.1`
- **Description:** Controls the randomness of the language model's output. A lower value makes the output more deterministic, while a higher value increases randomness.

### `--top_p`

- **Type:** `float`
- **Default:** `0.8`
- **Description:** Implements nucleus sampling by selecting tokens with a cumulative probability up to `top_p`. This controls the diversity of the output.

### `--dataset`

- **Type:** `str`
- **Default:** `'cora'`
- **Description:** The name of the dataset to be used. Ensure the dataset is available in your environment.

### `--max_tokens`

- **Type:** `int`
- **Default:** `2048`
- **Description:** The maximum number of tokens to generate in the output.

### `--repetition_penalty`

- **Type:** `float`
- **Default:** `1.05`
- **Description:** Penalty applied to reduce the likelihood of repeating the same token. Values greater than `1.0` discourage repetition.

### `--explainer`

- **Type:** `str`
- **Default:** `'cf-gnnfeatures'`
- **Description:** Specifies the explanation method to be used. Options depend on the implementations available in the `src.llms` module.

## Examples

### Running with Default Parameters

```bash
python main.py
```

### Running with Custom Parameters

```bash
python main.py --parameters '0.7' --temperature 0.5 --top_p 0.9 --dataset 'pubmed' --max_tokens 1024 --repetition_penalty 1.1 --explainer 'your_explainer'
```
You can run the experiments using the experiments script. 
```bash
bash experiments.sh
```

To evaluate the results type:
```bash
python src/evaluation.py
```

## Data Folder
The folder data contains data coming from CF-GNNExplainer and CF-GNNFeatures explainers using the graph incident format. The explainers are **NOT** included in this repo!

