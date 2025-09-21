import json
import random
import re
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

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
            # Ensure the annotation for the specific task exists
            if task_name in details["annotation"]:
                prepared_data.append({
                    "history": history,
                    "tutor_response": details["response"],
                    "label": details["annotation"][task_name]
                })
    return prepared_data


# --- 2. Data Augmentation Strategies (as per Section 3.2 of the paper) ---

def augment_dialogue_shuffling(history_text):
    """
    Implements the dialogue shuffling augmentation technique[cite: 53].
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
    Implements class balancing via random down-sampling of the majority class ('Yes')[cite: 74, 75].
    """
    yes_df = df[df[target_column] == 'Yes']
    other_df = df[df[target_column] != 'Yes']

    if len(yes_df) > 0:
        yes_df_downsampled = yes_df.sample(frac=0.005, random_state=42)
        balanced_df = pd.concat([yes_df_downsampled, other_df])
        print(f"Original 'Yes' count: {len(yes_df)}. Down-sampled 'Yes' count: {len(yes_df_downsampled)}")
        print(f"Total samples before balancing: {len(df)}. After balancing: {len(balanced_df)}")
        return balanced_df
    else:
        print("No 'Yes' samples to down-sample. Using original data.")
        return df
    

# --- 3. Prompt Construction (as per Section 3.1 & 3.2) ---

def create_prompt(example, task_name):
    """
    Constructs the final task-aware prompt for the LLM based on the specific track[cite: 77, 78].
    """
    task_templates = {
        "Mistake_Identification": "assess whether the tutor's response successfully identifies the mistake made by the student.",
        "Providing_Guidance": "assess whether the tutor's response offers effective explanations or hints to guide the student."
    }
    
    # Use a default message if task_name isn't in our templates
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
    
    # For fine-tuning, the full text will be `prompt + label`
    full_text = prompt + example['label']
    # print("Full Prompt:\n", full_text)  # Debugging line to inspect the prompt structure
    
    return {"text": full_text}


# --- 4. Main Execution and Training Setup ---

def setup_training_for_task(json_data, task_name):
    """
    Runs the entire preprocessing and training setup pipeline for a given task.
    """
    print(f"\n{'='*25}")
    print(f"Setting up training for task: {task_name}")
    print(f"{'='*25}\n")

    # 1. Load data for the specific task
    raw_data = load_and_prepare_data(json_data, task_name=task_name)
    if not raw_data:
        print(f"No data found for task '{task_name}'. Skipping.")
        return
    
    df = pd.DataFrame(raw_data)
    print("--- Initial Data ---")
    print(df['label'].value_counts())
    print("\n")

    # 2. Apply Data Augmentation [cite: 6]
    print("--- Applying Dialogue Shuffling ---")
    df['history'] = df['history'].apply(augment_dialogue_shuffling)
    print("Dialogue histories have been shuffled.\n")

    print("--- Applying Class Balancing ---")
    balanced_df = augment_class_balancing(df)
    print("\n")
    
    # 3. Create Hugging Face Dataset and format with prompts
    hf_dataset = Dataset.from_pandas(balanced_df)
    formatted_dataset = hf_dataset.map(lambda x: create_prompt(x, task_name=task_name))

    # 4. Setup Model and Tokenizer
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct" # Accessible version of the Qwen2.5 series [cite: 84]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        device_map="auto"  # This automatically handles device placement
        )
    
    def tokenize_function(examples):
        """Tokenizes text and creates the 'labels' needed for loss calculation."""
        # Tokenize the input text as before
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        
        # Create the 'labels' by copying 'input_ids'. The Trainer will handle the rest.
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()
        
        return tokenized_output

    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=hf_dataset.column_names)
    output_directory = f"/tmp/{task_name}_output"
    training_args = TrainingArguments(
        output_dir=output_directory,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=f'./logs_{task_name}',
    )


    # 6. Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # print(f"--- Training Setup Complete for {task_name} ---")
    # print("To start training, run: trainer.train()")
    # print("\n--- Example of a Final Formatted Input ---")
    # print(formatted_dataset[0]['text'])

    print(f"--- Starting Training for {task_name} ---")
    # This is the line you add to start the training process
    trainer.train()
    
    print(f"--- Training Finished for {task_name} ---")


if __name__ == "__main__":
    # Specify the path to your JSON data file
    json_filepath = 'data/trainset.json'
    
    # Load the data from the file
    loaded_data = load_json_from_file(json_filepath)
    
    # Proceed only if data was loaded successfully
    if loaded_data:
        tasks_to_run = ["Mistake_Identification", "Providing_Guidance"]
        for task in tasks_to_run:
            setup_training_for_task(loaded_data, task)