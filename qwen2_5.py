import json
import random
import re
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. Data Loading and Parsing ---

def load_json_from_file(filepath):
    """
    Loads JSON data from a specified file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None

def load_and_prepare_data(data, task_name):
    """
    Parses the nested JSON and flattens it for a specific task.
    """
    prepared_data = []
    for item in data:
        history = item["conversation_history"]
        for _, details in item["tutor_responses"].items():
            if task_name in details["annotation"]:
                prepared_data.append({
                    "history": history,
                    "tutor_response": details["response"],
                    "label": details["annotation"][task_name]
                })
    return prepared_data

# --- 2. Data Augmentation Strategies ---

def augment_dialogue_shuffling(history_text):
    """
    Implements the dialogue shuffling augmentation technique.
    """
    turns = re.split(r'(Tutor:|Student:)', history_text)
    if turns[0] == '':
        turns = turns[1:]
    interaction_pairs = [turns[i] + turns[i+1] for i in range(0, len(turns), 2)]
    random.shuffle(interaction_pairs)
    return "".join(interaction_pairs)

def augment_class_balancing(df, target_column='label'):
    """
    Implements class balancing via random down-sampling of the majority class ('Yes').
    """
    yes_df = df[df[target_column] == 'Yes']
    other_df = df[df[target_column] != 'Yes']
    if len(yes_df) > 0:
        yes_df_downsampled = yes_df.sample(frac=0.45, random_state=42)
        balanced_df = pd.concat([yes_df_downsampled, other_df])
        print(f"Original 'Yes' count: {len(yes_df)}. Down-sampled 'Yes' count: {len(yes_df_downsampled)}")
        print(f"Total samples before balancing: {len(df)}. After balancing: {len(balanced_df)}")
        return balanced_df
    else:
        print("No 'Yes' samples to down-sample. Using original data.")
        return df

# --- 3. Prompt Construction ---

def create_prompt(example, task_name):
    """
    Constructs the final task-aware prompt for the LLM.
    """
    task_templates = {
        "Providing_Guidance": "assess whether the tutor's response offers effective explanations or hints to guide the student."
    }
    task_description = task_templates.get(task_name, f"evaluate the tutor's response for {task_name}.")
    instruction = (
        "The following is a tutoring dialogue in the domain of mathematics. "
        "The student's last utterance may contain a mistake, and the AI tutor responds to it. "
        f"Your task is to {task_description}"
    )
    prompt = (
        f"Dialogue History:\n{example['history']}\n\n"
        f"Tutor's Response to Evaluate:\n{example['tutor_response']}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Evaluation (Yes/To some extent/No):\n"
    )
    full_text = prompt + example['label']
    return {"text": full_text}

# --- UPDATED: Function to Plot Loss Curves ---
def plot_loss_curves(trainer, task_name):
    """
    Extracts and plots the training and validation loss from the trainer's state history.
    """
    print(f"\n--- Generating Loss Plot for {task_name} ---")
    
    # The log history is stored in the trainer's state
    log_history = trainer.state.log_history
    
    train_logs = [log for log in log_history if 'loss' in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]

    if not train_logs or not eval_logs:
        print("Could not find sufficient log data to create a plot.")
        return

    train_steps = [log['step'] for log in train_logs]
    train_loss = [log['loss'] for log in train_logs]
    
    eval_steps = [log['step'] for log in eval_logs]
    eval_loss = [log['eval_loss'] for log in eval_logs]

    plt.figure(figsize=(12, 8))
    plt.plot(train_steps, train_loss, label='Training Loss', marker='o')
    plt.plot(eval_steps, eval_loss, label='Validation Loss', marker='o')
    
    plt.title(f'Training and Validation Loss for {task_name}')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f"{task_name}_loss_plot.png"
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Loss plot saved to '{plot_filename}'")

# --- 4. Main Execution and Training Setup ---

def setup_training_for_task(json_data, task_name):
    """
    Runs the entire preprocessing, training, and evaluation pipeline for a given task.
    """
    print(f"\n{'='*25}")
    print(f"Setting up pipeline for task: {task_name}")
    print(f"{'='*25}\n")

    raw_data = load_and_prepare_data(json_data, task_name=task_name)
    if not raw_data:
        return
    df = pd.DataFrame(raw_data)

    balanced_df = augment_class_balancing(df, 'label')
    balanced_df['history'] = balanced_df['history'].apply(augment_dialogue_shuffling)

    train_df, temp_df = train_test_split(
        balanced_df, test_size=0.30, random_state=42, stratify=balanced_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}\n")
    
    test_set_path = f"{task_name}_test_set.csv"
    test_df.to_csv(test_set_path, index=False)
    print(f"Test set for '{task_name}' saved to '{test_set_path}'")

    train_dataset = Dataset.from_pandas(train_df).map(lambda x: create_prompt(x, task_name=task_name))
    val_dataset = Dataset.from_pandas(val_df).map(lambda x: create_prompt(x, task_name=task_name))

    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        device_map="auto"
    )
    
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        return tokenized_output

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

    output_directory = f"./{task_name}_model_output"
    
    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_dir=f'{output_directory}/logs',
        load_best_model_at_end=True,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    print(f"--- Starting Training for {task_name} ---")
    trainer.train()
    print(f"--- Training Finished for {task_name} ---")
    
    trainer.save_model(output_directory)
    print(f"Best model saved to {output_directory}")
    
    # --- UPDATED: Call the plotting function with the trainer object ---
    plot_loss_curves(trainer, task_name)


if __name__ == "__main__":
    json_filepath = 'data/trainset.json'
    loaded_data = load_json_from_file(json_filepath)
    if loaded_data:
        tasks_to_run = ["Providing_Guidance"]
        for task in tasks_to_run:
            setup_training_for_task(loaded_data, task)

