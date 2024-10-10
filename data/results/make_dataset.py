import json

def convert_json_to_jsonl(input_file, output_file):
    """
    Reads a JSON file where each key corresponds to a list containing a question and an answer.
    Writes a JSONL file with each line containing a JSON object with 'question' and 'answer' keys.
    
    Parameters:
    - input_file: Path to the input JSON file.
    - output_file: Path to the output JSONL file.
    """
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Open the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Iterate over each key in the JSON data
        for key in data:
            # Ensure the value is a list with at least two elements
            if isinstance(data[key], list) and len(data[key]) >= 2:
                question = data[key][0]
                answer = data[key][1]
                # Create a dictionary with 'question' and 'answer' keys
                qa_pair = {
                    "question": question,
                    "answer": answer
                }
                # Write the dictionary as a JSON object to the JSONL file
                f_out.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
            else:
                print(f"Skipping key '{key}' because it does not contain a valid question-answer pair.")

    print(f"Conversion complete. The output file is saved as '{output_file}'.")

# Example usage:
input_file = 'input.json'   # Replace with your input JSON file path
output_file = 'output.jsonl'  # Replace with your desired output JSONL file path


if __name__ == "__main__":


    convert_json_to_jsonl("data/results/cf-gnnfeatures_Qwen2.5-14B-Instruct-GPTQ-Int4_cora_Response.json", "data/results/dataset_cora.jsonl")
