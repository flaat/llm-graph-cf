from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from data.kb import cora_description, citeseer_description, get_system_prompt
import random
import numpy as np
import time
MODEL_MAPPING = {
    "phi_4B": "microsoft/Phi-3.5-mini-instruct",  # 128k
    "phi_7B": "microsoft/Phi-3-small-128k-instruct",  # 128k
    "phi_14B": "microsoft/Phi-3-medium-128k-instruct",  # 128k
    "phi_14B_Q8_GGUF": "ssmits/Phi-3-medium-128k-instruct-Q8_0-GGUF",  # 128k
    "mistral_7B": "mistralai/Mistral-7B-Instruct-v0.3",  # 32k
    "smollm2_2B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # 8k
    "llama_8B": "meta-llama/Llama-3.1-8B-Instruct",  # 128k
    "llama_3B": "meta-llama/Llama-3.2-3B-Instruct",  # 128k
    "qwen_0.5B": "Qwen/Qwen2.5-0.5B-Instruct",  # 128k
    "qwen_1.5B": "Qwen/Qwen2.5-1.5B-Instruct",  # 128k
    "qwen_3B": "Qwen/Qwen2.5-3B-Instruct",  # 128k
    "qwen_7B": "Qwen/Qwen2.5-7B-Instruct",  # 128k
    "qwen_14B": "Qwen/Qwen2.5-14B-Instruct",  # 128k
    "qwen_32B": "Qwen/Qwen2.5-32B-Instruct",  # 128k

    "qwen_0.5B_Q8_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",  # 128k
    "qwen_1.5B_Q8_GPTQ": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",  # 128k
    "qwen_3BQ_Q8_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",  # 128k
    "qwen_7BQ_Q8_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",  # 128k
    "qwen_14BQ_Q8_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",  # 128k
    "qwen_32BQ_Q8_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",  # 128k
    
    "qwen_0.5B_Q4_GPTQ": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",  # 128k
    "qwen_1.5B_Q4_GPTQ": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",  # 128k
    "qwen_3B_Q4_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",  # 128k
    "qwen_7B_Q4_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  # 128k
    "qwen_14B_Q4_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",  # 128k
    "qwen_32B_Q4_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",  # 128k
}

def set_full_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_explanations_mistral_vllm(model_name, temperature, top_p, dataset, max_tokens, repetition_penalty, explainer, prompt_type:str, max_model_len: int =7000):
    set_full_reproducibility()

    perturbation_type: str = "node features" if explainer == "cf-gnnfeatures" else "adjacency matrix"
    model_name = MODEL_MAPPING[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_tokens, top_k=10)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_name, gpu_memory_utilization=0.1, max_model_len=max_model_len, max_num_seqs=1)  # Reduced GPU utilization to avoid OOM

    data_path: str = f"data/{dataset}/{explainer}"
    extension: str = "json"
    dataset_description: str = cora_description if dataset == "cora" else citeseer_description

    with open(f"{data_path}/random_seed_42_counterfactual_graph_descriptions.{extension}", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    with open(f"{data_path}/random_seed_42_factual_graph_descriptions.{extension}", 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    responses = {}
    prompt = ""
    for key, value in tqdm(data1.items()):
        
        if prompt_type == "few":
            from .prompts import FEW_SHOT_PROMPT
            
            prompt = FEW_SHOT_PROMPT.format(data2[key], data1[key], dataset_description)
            
        
        elif prompt_type == "zero":
            from .prompts import ZERO_SHOT_PROMPT
            
            prompt = ZERO_SHOT_PROMPT.format(data2[key], data1[key])
            
        
        elif prompt_type == "standard":
            from .prompts import STANDARD_PROMPT
                 
            prompt = STANDARD_PROMPT.format(data2[key], data1[key], dataset_description)
    
        else:
            
            raise ValueError("prompt_type values does not exist!")

        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": get_system_prompt(perturbation_type=perturbation_type, task="graph")},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template to prepare the input text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if len(text) > max_model_len:
            continue
        
        try:
            with torch.no_grad():
                start = time.time()
                outputs = llm.generate([text], sampling_params=sampling_params)
                end = time.time()
                print(f"*************Time taken for generation: {end - start} mode: {model_name}*************")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("CUDA OOM detected. Attempting to free up memory.")
                # Free up GPU memory
                torch.cuda.empty_cache()
            # Retry with reduced GPU utilization if possible
            try:
                # Reduce GPU memory utilization for retry
                llm = LLM(model=model_name, gpu_memory_utilization=1.0, max_model_len=max_model_len, max_num_seqs=1)
                with torch.no_grad():
                    outputs = llm.generate([text], sampling_params=sampling_params)
            except RuntimeError as retry_e:
                if "CUDA out of memory" in str(retry_e):
                    print("Retry failed due to CUDA OOM. Skipping this prompt.")
                    continue
            except AssertionError as retry_assert_e:
                print(f"Assertion error during retry: {retry_assert_e}")
                continue
            except Exception as retry_other_e:
                print(f"Unexpected error during retry: {retry_other_e}")
                continue
            else:
                print(f"Unexpected error: {e}")
            continue
        except AssertionError as assert_e:
            print(f"Assertion error: {assert_e}")
            continue

        # Process the outputs if generated successfully
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text

        print(generated_text)
        responses[key] = (prompt, generated_text)

        # Delete large variables to release memory
        del generated_text
        del outputs

    # Save the responses to a JSON file
    output_file = f"data/results/{explainer}_{model_name.split('/')[1]}_{dataset}_{prompt_type}_Response.json"
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(responses, output, indent=4)

    print(f"Responses saved to {output_file}.")
