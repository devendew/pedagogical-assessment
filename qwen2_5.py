import json
import random
import re
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

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
    It randomly permutes the tutor-student interaction pairs.
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
        yes_df_downsampled = yes_df.sample(frac=0.25, random_state=42)
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
        "Mistake_Identification": "assess whether the tutor's response successfully identifies the mistake made by the student.",
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

# --- 4. Main Execution and Training Setup ---

def setup_training_for_task(json_data, task_name):
    """
    Runs the entire preprocessing, training, and evaluation pipeline for a given task.
    """
    print(f"\n{'='*25}")
    print(f"Setting up pipeline for task: {task_name}")
    print(f"{'='*25}\n")

    # 1. Load data
    raw_data = load_and_prepare_data(json_data, task_name=task_name)
    if not raw_data:
        print(f"No data found for task '{task_name}'. Skipping.")
        return
    df = pd.DataFrame(raw_data)

    # 2. Augment data (Downsampling first, as requested)
    print("--- Applying Class Balancing (Down-sampling) ---")
    balanced_df = augment_class_balancing(df, 'label')
    print("\n--- Applying Dialogue Shuffling ---")
    balanced_df['history'] = balanced_df['history'].apply(augment_dialogue_shuffling)
    print("Dialogue histories have been shuffled.\n")

    # 3. --- NEW: Create Train, Validation, and Test Splits ---
    print("--- Creating Stratified Train/Val/Test Splits (70/15/15) ---")
    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=0.30,
        random_state=42,
        stratify=balanced_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['label']
    )
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}\n")

    # 4. Create Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 5. Format datasets with prompts
    train_dataset = train_dataset.map(lambda x: create_prompt(x, task_name=task_name))
    val_dataset = val_dataset.map(lambda x: create_prompt(x, task_name=task_name))
    test_dataset_formatted = test_dataset.map(lambda x: create_prompt(x, task_name=task_name), remove_columns=test_dataset.column_names)

    train_dataset = train_dataset.remove_columns(["label"])
    val_dataset = val_dataset.remove_columns(["label"])


    # 6. Setup Model and Tokenizer
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

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset_formatted.map(tokenize_function, batched=True)

    # 7. Configure Training
    output_directory = f"/tmp/{task_name}_output"
    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        logging_dir=f'./logs_{task_name}',
        load_best_model_at_end=True,
        save_total_limit=2,
        # --- FIX: Use older arguments for compatibility ---
        eval_strategy="steps",  # <-- ADD THIS
        eval_steps=100,
        save_strategy="steps",         # <-- ADD THIS for clarity
        save_steps=100,
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

    # 8. --- NEW: Run Inference on the Test Set ---
    print(f"\n--- Running Inference on Test Set for {task_name} ---")
    predictions = trainer.predict(tokenized_test_dataset)
    
    predicted_token_ids = np.argmax(predictions.predictions, axis=-1)
    
    labels = predictions.label_ids
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    predicted_labels = [pred.split()[-1] if pred.split() else "" for pred in decoded_preds]
    true_labels = [label.split()[-1] if label.split() else "" for label in decoded_labels]

    correct = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
    total = len(true_labels)
    accuracy = correct / total
    
    print(f"\nTest Set Accuracy: {accuracy:.4f}")
    
    print("\n--- Example Predictions ---")
    for i in range(min(5, len(test_df))):
        print(f"\nExample {i+1}:")
        print(f"  Tutor Response: {test_df.iloc[i]['tutor_response']}")
        print(f"  True Label:     {true_labels[i]}")
        print(f"  Predicted Label:  {predicted_labels[i]}")


if __name__ == "__main__":
    json_filepath = 'data/trainset.json'
    loaded_data = load_json_from_file(json_filepath)
    if loaded_data:
        tasks_to_run = ["Mistake_Identification", "Providing_Guidance"]
        for task in tasks_to_run:
            setup_training_for_task(loaded_data, task)

