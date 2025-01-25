import json
import re
import ast
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx 
import re
import argparse
import os


matching_graphs = 0
matching_class = 0
matching_cf_features = 0
matching_f_features = 0
matching_f_neigh = 0
matching_cf_neigh = 0
factual_nodes_features = None
counterfactual_nodes_neigh_target = None
factual_nodes_neigh_target = None

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
    "qwen_3BQ_Q4_GPTQ": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",  # 128k
    "qwen_7BQ_Q4_GPTQ": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",  # 128k
    "qwen_14BQ_Q4_GPTQ": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",  # 128k
    "qwen_32BQ_Q4_GPTQ": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",  # 128k
}


def extract_factual_and_counterfactual_info(phrase):
    # Split the phrase into factual and counterfactual parts
    split_text = re.split(r"given the factual graph:|and given the counterfactual example:", phrase)
    
    # Assign the split parts to variables
    factual_part = split_text[0].strip() 
    counterfactual_part = split_text[1].strip() 
    
    # Define a pattern to extract node ID and classification
    pattern = r"The node (\d+) .*? it is classified as ([\w_]+)"
    
    # Find all factual tuples
    factual_tuples = re.findall(pattern, factual_part)
    factual_tuples = {int(node_id):classification for node_id, classification in factual_tuples}
    
    # Find all counterfactual tuples
    counterfactual_tuples = re.findall(pattern, counterfactual_part)
    counterfactual_tuples =  {int(node_id):classification for node_id, classification in counterfactual_tuples}
    
    return factual_tuples, counterfactual_tuples


def TNU(infos_dict, merged_data, key):
    """
    This metric assesses the ability of the LLM to accurately identify and reference the target node within the graph
    structure. Given the importance of the target node as the focal point of the classification task, correctly
    pinpointing it is crucial for generating valid explanations. Formally, given a graph G(V, E), the target
    node t ∈ V, and a node v ∈ V predicted by the LLM, we define the Target Node Identification (TNI) metric as:
    TNI(v) = 1 if v = t, 0 otherwise.
    Args:
        infos_dict (dict): Dictionary containing information about the target node.
        merged_data (dict): Dictionary containing merged data with target node information.
        key (str): Key to access specific data in the merged_data dictionary.
    Returns:
        None
    """
    
    global matching_graphs
    
    try:
        # Compare Target_Node_ID
        merged_target_id = str(merged_data[key]['Target_Node_ID'])
        infos_target_id = str(infos_dict['Target_Node_ID'])

        if merged_target_id == infos_target_id:
            matching_graphs += 1
            
    except Exception:
        pass


def CCU(infos_dict, merged_data, key):
    """
    This metric evaluates the capacity of the LLM to correctly
    comprehend and express the change in class assignment of the target node from the factual graph G to
    the counterfactual graph G′. Let G′ = (V, E′) be
    the factual, such that fθ (G′) = c′ where c′ is the
    counterfactual class of the target node t such that
    fθ (G′)̸ = fθ (G), and let cLLM be the counterfactual
    class predicted by the LLM, CCI is defined as:
    CCI(cLLM) =
    (
    1 if cLLM = c′,
    0 otherwise.
    Args:
        infos_dict (dict): Dictionary containing information about the counterfactual target node class.
        merged_data (dict): Dictionary containing merged data with target node information.
        key (str): Key to access the specific target node in the merged data.
    Returns:
        None
    """
    
    global matching_class
    
    try:
        target_class = merged_data[key]["Counterfactual_Target_Node_class"]
        cf_class = infos_dict["Counterfactual_Target_Node_class"]
    
        if target_class == cf_class:
            matching_class += 1
            
    except Exception:
        pass


def CFTNF(infos_dict, merged_data, key):
    
    global matching_cf_features
    
    try:    
        counterfactual_nodes_features = infos_dict["Counterfactual_Target_Node_Features"].replace(" ","").split(",") if type(infos_dict["Counterfactual_Target_Node_Features"]) == str else infos_dict["Counterfactual_Target_Node_Features"]
        counterfactual_nodes_features_target = merged_data[key]["Counterfactual_Target_Node_Features"].replace(" ","").split(",") if type(merged_data[key]["Counterfactual_Target_Node_Features"]) == str else merged_data[key]["Counterfactual_Target_Node_Features"]
        if Counter(counterfactual_nodes_features) == Counter(counterfactual_nodes_features_target):
        
            matching_cf_features += 1
    except Exception:
        pass


def FTNF(infos_dict, merged_data, key):
    
    global matching_f_features
    global factual_nodes_features
    
    try:
        factual_nodes_features = infos_dict["Factual_Target_Node_Features"].replace(" ","").split(",") if type(infos_dict["Factual_Target_Node_Features"]) == str else infos_dict["Factual_Target_Node_Features"]
        factual_nodes_features_target = merged_data[key]["Factual_Target_Node_Features"].replace(" ","").split(",") if type(merged_data[key]["Factual_Target_Node_Features"]) == str else merged_data[key]["Factual_Target_Node_Features"]
        if Counter(factual_nodes_features) == Counter(factual_nodes_features_target):
        
            matching_f_features += 1  
    except Exception:
        pass
    
def CFTNN(infos_dict, merged_data, key):
    
    global matching_cf_neigh
    global counterfactual_nodes_neigh_target
    try:
        counterfactual_nodes_neigh = infos_dict["Counterfactual_Target_Node_Neighbors"].replace(" ","").split(",") if type(infos_dict["Counterfactual_Target_Node_Neighbors"]) == str else infos_dict["Counterfactual_Target_Node_Neighbors"]
        counterfactual_nodes_neigh_target = merged_data[key]["Counterfactual_Target_Node_Neighbors"].replace(" ","").split(",") if type(merged_data[key]["Counterfactual_Target_Node_Neighbors"]) == str else merged_data[key]["Counterfactual_Target_Node_Neighbors"]
        if Counter(counterfactual_nodes_neigh) == Counter(counterfactual_nodes_neigh_target):
            
            matching_cf_neigh += 1
    
    except Exception:
        pass



def FTNN(infos_dict, merged_data, key):
    
    global matching_f_neigh
    global factual_nodes_neigh_target
    
    try:
        factual_nodes_neigh = infos_dict["Factual_Target_Node_Neighbors"].replace(" ","").split(",") if type(infos_dict["Factual_Target_Node_Neighbors"]) == str else infos_dict["Factual_Target_Node_Neighbors"]
        factual_nodes_neigh_target = merged_data[key]["Factual_Target_Node_Neighbors"].replace(" ","").split(",") if type(merged_data[key]["Factual_Target_Node_Neighbors"]) == str else merged_data[key]["Factual_Target_Node_Neighbors"]
        if Counter(factual_nodes_neigh) == Counter(factual_nodes_neigh_target):
            
            matching_f_neigh += 1

    except Exception:
        
        pass 
    
    
    
def FTC(dict1, dict2, key):
    """
    Check if the values of the factual target graph class are equal in both dictionaries for the given key.
    
    Args:
        dict1 (dict): First dictionary containing factual target graph class.
        dict2 (dict): Second dictionary containing factual target graph class.
        key (str): Key to access the specific target node in both dictionaries.
    
    Returns:
        bool: True if the values are equal, False otherwise.
    """
    try:
        class1 = dict1[key]["Factual_Target_Graph_Class"]
        class2 = dict2["Factual_Target_Graph_Class"]
        return class1 == class2
    except KeyError:
        return False
    
def CTC(dict1, dict2, key):
    """
    Check if the values of the counterfactual target graph class are equal in both dictionaries for the given key.
    
    Args:
        dict1 (dict): First dictionary containing counterfactual target graph class.
        dict2 (dict): Second dictionary containing counterfactual target graph class.
        key (str): Key to access the specific target node in both dictionaries.
    
    Returns:
        bool: True if the values are equal, False otherwise.
    """
    try:
        class1 = dict1[key]["CounterFactual_Target_Graph_Class"]
        class2 = dict2["CounterFactual_Target_Graph_Class"]
        return class1 == class2
    except KeyError:
        return False

def NDIFF(dict1, dict2, key):
    
    
    try:
        neigh1 = dict1[key]["Node_Differences"]
        neigh2 = dict2["Node_Differences"]
        return neigh1 == neigh2
    except KeyError:
        return False
    
    
def EDIFF(dict1, dict2, key):
    """
    Check if the values of the edge differences are equal in both dictionaries for the given key.
    
    Args:
        dict1 (dict): First dictionary containing edge differences.
        dict2 (dict): Second dictionary containing edge differences.
        key (str): Key to access the specific target node in both dictionaries.
    
    Returns:
        bool: True if the values are equal, False otherwise.
    """
    try:
        edges1 = dict1[key]["Edge_differences"]
        edges2 = dict2["Edge_Differences"]
        return edges1 == edges2
    except KeyError:
        return False

ftc = 0
ctc = 0
ndiff = 0
ediff = 0


def evaluate_graph_classification(model_name: str, dataset: str, folder: str):
    global ftc
    global ctc
    global ndiff
    global ediff
    # Define file paths
    file1_path = f'data/{dataset}/{folder}/factual_dict.json'
    file3_path = f"data/results/{folder}_{MODEL_MAPPING[model_name].split('/')[1]}_{dataset}_Response.json"

    # Load JSON data
    data1 = load_json(file1_path)
    data3 = load_json(file3_path)

    length = len(data1)
    for key, value in data3.items():
        

        if not isinstance(value, list) or len(value) < 2:
            #print(f"Invalid data format for key {key}: {value}")
            continue
        
        text = value[1]
        infos_dict = extract_infos_dict(text)
        
        if infos_dict is None:
            #print(f"No valid infos dictionary found for key {key}")
            continue   
    
        if FTC(data1, infos_dict, key):
            ftc += 1
        if CTC(data1, infos_dict, key):
            ctc += 1
        if NDIFF(data1, infos_dict, key):
            ndiff += 1
        if EDIFF(data1, infos_dict, key):
            ediff += 1
            
    print(f"{MODEL_MAPPING[model_name].split('/')[1]} & {ftc/length:.3f} & {ctc/length:.3f} & {ndiff/length:.3f} & {ediff/length:.3f} \\\ ")

def evaluate(model_name: str, dataset: str, folder: str):
    
    
    
    # Define file paths
    file1_path = f'data/{dataset}/{folder}/random_seed_42_counterfactual_target_dict_repr.json'
    file2_path = f'data/{dataset}/{folder}/random_seed_42_factual_target_dict_repr.json'
    file3_path = f"data/results/{folder}_{MODEL_MAPPING[model_name].split('/')[1]}_{dataset}_Response.json"

    # Load JSON data
    data1 = load_json(file1_path)
    data2 = load_json(file2_path)
    data3 = load_json(file3_path)

    # Merge data1 and data2 into merged_data
    merged_data = merge_dictionaries(data1, data2)
    text_dict = {}
    # Process data3 and compute metrics
    total_graphs = len(merged_data)
    global matching_graphs
    global matching_class
    global matching_cf_features
    global matching_f_features
    global matching_f_neigh
    global matching_cf_neigh
    global factual_nodes_features
    global counterfactual_nodes_neigh_target
    global factual_nodes_neigh_target
    
    os.makedirs(f"data/figures/{model_name}/{explainer}", exist_ok=True)


    for key, value in data3.items():
        
        # Check if key exists in merged_data
        if key not in merged_data:
            #print(f"Key {key} not found in merged data.")
            continue
        factual, counterfactual = extract_factual_and_counterfactual_info(value[0])

        if not isinstance(value, list) or len(value) < 2:
            #print(f"Invalid data format for key {key}: {value}")
            continue
        
        text = value[1]
        infos_dict = extract_infos_dict(text)
        
        if infos_dict is None:
            #print(f"No valid infos dictionary found for key {key}")
            continue
        
        
        TNU(infos_dict, merged_data, key)


        CCU(infos_dict, merged_data, key)
        
        
        CFTNF(infos_dict, merged_data, key)

        
        FTNF(infos_dict, merged_data, key)
        
        
        CFTNN(infos_dict, merged_data, key)
        
        
        FTNN(infos_dict, merged_data, key)
                
        
        
        if matching_graphs + matching_class + matching_cf_features + matching_f_features + matching_cf_neigh + matching_f_neigh > 5:
            
        
            text_dict[key] = infos_dict["Natural_Language_Explanation"] + f"Original target nodes features: {factual_nodes_features}"
            
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the figsize as needed
            
            
            for i, t in enumerate(["Factual", "Counterfactual"]):
                # Create a directed graph
                G_factual = nx.DiGraph()

                # Add nodes and edges based on the dictionary
                
                target_node_id = merged_data[key]["Target_Node_ID"]
                factual_neighbors = merged_data[key][f"{t}_Target_Node_Neighbors"]
                fix_error = "c" if t == "Counterfactual" else "C"
                # Add the target node
                G_factual.add_node(target_node_id, label=merged_data[key][f"{t}_Target_Node_{fix_error}lass"], features=merged_data[key][f"{t}_Target_Node_Features"])
                node_classes = factual if t == "Factual" else counterfactual

                neigh = factual_nodes_neigh_target if t == "Factual" else counterfactual_nodes_neigh_target
                # Add neighbors
                if factual_neighbors:

                    for neighbor in neigh:
                        G_factual.add_node(neighbor, label=node_classes.get(int(neighbor), "Unknown"))  # Add neighbor nodes with class label

                        G_factual.add_edge(target_node_id, neighbor)
                        
                node_labels = {node: f'\n\n{G_factual.nodes[node]["label"]}' for node in G_factual.nodes()}
                # Draw the graph
                if t != "Counterfactual":
                    pos = nx.spring_layout(G_factual, k=0.5, scale=1.5)  # Position nodes using Fruchterman-Reingold force-directed algorithm
                nx.draw(G_factual, pos, with_labels=True, labels=node_labels, node_size=2000, ax=axs[i], node_color='skyblue', font_size=12, font_weight='bold', arrowsize=20)

                nx.draw(G_factual, pos, with_labels=True, node_size=2000, ax=axs[i], node_color='skyblue', font_size=14, font_weight='bold', arrowsize=20)
                axs[i].set_xlim([1.3*x for x in axs[i].get_xlim()])
                axs[i].set_ylim([1.3*y for y in axs[i].get_ylim()])
                # Display the plot
                axs[i].set_title(f"{t} Graph Representation", fontsize=24)
            fig.savefig(f"data/figures/{model_name}/{explainer}/{key}.png", bbox_inches='tight')
            fig.tight_layout()
            plt.close()
        
        
            
                
    # Compute ratios
    ratio_target_id = matching_graphs / total_graphs if total_graphs > 0 else 0
    ratio_class = matching_class / total_graphs if total_graphs > 0 else 0
    ratio_cf_feat = matching_cf_features / total_graphs if total_graphs > 0 else 0
    ratio_f_feat = matching_f_features / total_graphs if total_graphs > 0 else 0
    ratio_cf_neigh = matching_cf_neigh / total_graphs if total_graphs > 0 else 0
    ratio_f_neigh = matching_f_neigh / total_graphs if total_graphs > 0 else 0

    output_file = f"data/results/best_response_{folder}_{model_name}_{dataset}.json"
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(text_dict, output, indent=4)

    print(f"{MODEL_MAPPING[model_name].split('/')[1]} & {ratio_target_id:.3f} & {ratio_class:.3f} & {ratio_cf_feat:.3f} & {ratio_f_feat:.3f} & {ratio_cf_neigh:.3f} & {ratio_f_neigh:.3f} \\\ ")


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def merge_dictionaries(data1, data2):
    """Merge two dictionaries, filling missing values from data2."""
    merged = {}
    for key in data1:
        if key not in data2:
            print(f"Key {key} not found in data2.")
            continue
        merged[key] = {}
        for subkey in data1[key]:
            value1 = data1[key][subkey]
            value2 = data2[key].get(subkey)
            if is_missing(value1):
                merged[key][subkey] = value2
            else:
                merged[key][subkey] = value1
    return merged

def is_missing(value):
    """Check if a value is considered missing."""
    return value is None or value == '' or value == [] or value == {}

def extract_infos_dict(text):
    """Extract and evaluate the 'infos' dictionary from a code block in text."""
    # Extract the code block containing the infos dictionary
    pattern = r'```(.*?)```'
    pattern_2 = r'infos\s*=\s*(\{.*?\})'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        #print("Code block not found in the string data.")
        match = re.search(pattern_2, text, re.DOTALL)
        if not match:
            return None
    code_block = match.group(1).strip()

    # Extract the dictionary part
    dict_pattern = r'\s*\s*(\{.*?\})\s*$'
    dict_match = re.search(dict_pattern, code_block, re.DOTALL)
    if not dict_match:
        #print("Infos dictionary not found in the code block.")
        return None
    dict_str = dict_match.group(1)

    # Correct the string
    fixed_dict_str = escape_unescaped_apostrophes(dict_str)

    # Safely evaluate the dictionary string to get a Python dictionary
    try:
        infos_dict = ast.literal_eval(fixed_dict_str)
    except Exception as e:
        #print(f"Error evaluating infos dictionary: {e}")
        return None
    return infos_dict

def escape_unescaped_apostrophes(text):
    """Escape unescaped apostrophes in all string literals within the text."""
    # Pattern to match string literals enclosed in single quotes
    pattern = r"(')((?:[^'\\]|\\.)*?)'"  # Matches single-quoted strings
    def replacer(match):
        quote, content = match.groups()
        # Escape unescaped apostrophes in the content
        content_escaped = content.replace("'", "\\'")
        return f"{quote}{content_escaped}'"
    # Apply the replacement to the entire text
    fixed_text = re.sub(pattern, replacer, text)
    return fixed_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance.')
    parser.add_argument('--model_name', type=str, default="mistral_7B", help='Name of the model to evaluate')
    parser.add_argument('--dataset_name', type=str, default="cora", help='Name of the dataset to evaluate')
    parser.add_argument('--explainer', type=str, default="cf-gnnfeatures", help='Name of the explainer to evaluate')
    parser.add_argument('--task', type=str, default="node", help='Name of the explainer to evaluate')

    args = parser.parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    explainer = args.explainer
    task = args.task
    
    if task == "graph":
        evaluate_graph_classification(model_name=model_name, dataset=dataset_name, folder=explainer)
    else:
    #print("Model Name | TNU | CCU | FTNF | CFTNF | FTNN | CFTNN")
        evaluate(model_name=model_name, dataset=dataset_name, folder=explainer)
