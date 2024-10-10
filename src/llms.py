from transformers import AutoTokenizer
import json
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from data.kb import cora_description, citeseer_description, get_system_prompt


def get_eplanations(parameters, temperature, top_p, dataset, max_tokens, repetition_penalty, explainer):
    
    perturbation_type: str = "node features" if explainer == "cf-gnnfeatures" else "adjacency matrix"

    model_name = f"Qwen2.5-{parameters}B-Instruct-GPTQ-Int4"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/" + model_name)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_tokens, top_k=10)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model="Qwen/" + model_name, tensor_parallel_size=1, quantization="gptq", enable_prefix_caching=True, gpu_memory_utilization=0.9)
    
    data_path: str = f"data/{dataset}/{explainer}"
    extension: str = "json"
    
    dataset_description: str = cora_description if dataset == "cora" else citeseer_description
    

    with open(f"{data_path}/random_seed_42_counterfactual_graph_descriptions.{extension}", 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)

    with open(f"{data_path}/random_seed_42_factual_graph_descriptions.{extension}", 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    responses = {}

    for key, value in tqdm(data1.items()):
        

        
        prompt = (
            f"Given the factual graph: {data2[key]} and given the counterfactual example: {data1[key]} "
            f"and given the knowledge base about the dataset: {dataset_description}, fill the dictionary!"
            f"and provide an explanation about the change in classification for the target node, please evaluate also the influences of neighbors nodes."
        )

        if len(prompt) > 30000:
            continue
        
        
        # Prepare the messages for the model
        messages = [
            {"role": "system", "content": get_system_prompt(perturbation_type=perturbation_type)},
            {"role": "user", "content": prompt}
        ]

        # Apply chat template to prepare the input text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.no_grad():
            outputs = llm.generate([text], sampling_params=sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            
        print(generated_text)
        responses[key] = (prompt, generated_text)

        del generated_text
        
    # Save the responses to a JSON file
    output_file = f"data/results/{explainer}_{model_name}_{dataset}_Response.json"
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(responses, output, indent=4)

    print(f"Responses saved to {output_file}.")
