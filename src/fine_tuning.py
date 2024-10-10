import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pre-trained model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)
model.to(device)

# Load your custom question-answer dataset
dataset = load_dataset('json', data_files={'train': 'data/results/dataset_cora.jsonl'})

# Optionally, split the dataset
dataset = dataset['train'].train_test_split(test_size=0.1)

# Tokenize the dataset
def tokenize_function(examples):
    questions = [q.strip(".,:;-_^<>|\\\"' ") for q in examples['question']]
    answers = [a.strip(".,:;-_^<>|\\\"' ") for a in examples['answer']]
    
    # Define prompt format
    inputs = ["Question: " + q + "\nAnswer:" for q in questions]
    targets = answers
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=2048, truncation=True, padding="max_length")
    
    # Tokenize targets (answers)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=2048, truncation=True, padding="max_length")
    
    # Replace the input IDs with labels for the targets
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['question', 'answer'],
)

# Set up data collator
from transformers import DataCollatorForSeq2Seq


data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    return_tensors='pt',
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./qwen_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    fp16=True,
    bf16=False,
    warmup_steps=50,
    weight_decay=0.01,
    dataloader_num_workers=4,
    report_to="none",
    run_name="qwen_finetuning_qa",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],  # 'test' because of train_test_split
    data_collator=data_collator,
)

# Begin fine-tuning
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./qwen_finetuned")
tokenizer.save_pretrained("./qwen_finetuned")


# Load the fine-tuned model for evaluation or inference
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    "./qwen_finetuned",
    trust_remote_code=True,
)
fine_tuned_model.to(device)

fine_tuned_tokenizer = AutoTokenizer.from_pretrained(
    "./qwen_finetuned",
    trust_remote_code=True,
)

# Generate answers using the fine-tuned model
def generate_answer(question, max_length=2048, temperature=0.7, top_p=0.95):
    input_text = "Question: " + question.strip() + "\nAnswer:"
    input_ids = fine_tuned_tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = (input_ids != fine_tuned_tokenizer.pad_token_id).long()
    output = fine_tuned_model.generate(
        input_ids=input_ids,
        max_length=max_length,
        do_sample=True,
        attention_mask=attention_mask,
        temperature=temperature,
        top_p=top_p,
        top_k=10,
        eos_token_id=fine_tuned_tokenizer.eos_token_id,
        pad_token_id=fine_tuned_tokenizer.pad_token_id,
    )
    generated_text = fine_tuned_tokenizer.decode(
        output[0], skip_special_tokens=True
    )
    # Extract the answer from the generated text
    answer = generated_text.split("Answer:")[-1].strip()
    return answer

# Example usage
question = "Please explain to me what is counterfactual explanation on graph"
answer = generate_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
