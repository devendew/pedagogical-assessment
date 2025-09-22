import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import os

def create_inference_prompt(row, task_name="Providing_Guidance"):
    """
    Constructs the inference prompt for the LLM, without the label.
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
        f"Dialogue History:\n{row['history']}\n\n"
        f"Tutor's Response to Evaluate:\n{row['tutor_response']}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Evaluation (Yes/To some extent/No):\n"
    )
    return prompt

def run_inference():
    """
    Loads the saved model and test set, runs inference, and evaluates the results.
    """
    # --- 1. Configuration ---
    task_name = "Providing_Guidance"
    model_path = f"./{task_name}_model_output"
    test_set_path = f"{task_name}_test_set.csv"
    
    print("--- 1. Loading Model and Tokenizer ---")
    
    # --- Check if the model path exists before loading ---
    if not os.path.exists(model_path):
        print(f"\nERROR: Model directory not found at '{model_path}'")
        print("Please ensure you are running this script from the same directory where the model was saved.")
        return

    try:
        base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"  # base model used for training

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load fine-tuned model from saved directory
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Load and Prepare Test Data ---
    print(f"\n--- 2. Loading Test Data from '{test_set_path}' ---")
    try:
        test_df = pd.read_csv(test_set_path)
        print(f"Loaded {len(test_df)} examples from the test set.")
    except FileNotFoundError:
        print(f"Error: Test set file not found at '{test_set_path}'.")
        return

    true_labels = []
    predicted_labels = []
    
    valid_labels = ["Yes", "No", "To some extent"]

    print("\n--- 3. Running Inference on Test Set ---")
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Predicting"):
        prompt = create_inference_prompt(row)
        true_label = row['label']
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_text = full_text[len(prompt):].strip()

        predicted_label = prediction_text.split()[0] if prediction_text.split() else "N/A"
        
        if "extent" in predicted_label:
            predicted_label = "To some extent"
        
        if predicted_label not in valid_labels:
            predicted_label = "No"
            
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # --- 4. Evaluate the Results ---
    print("\n--- 4. Evaluation Metrics ---")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    print("\n--- Overall Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    print("\n--- Detailed Classification Report ---")
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    print(report)

    print("\n--- Confusion Matrix ---")
    labels = sorted(list(set(true_labels) | set(predicted_labels)))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = 'True'
    cm_df.columns.name = 'Predicted'
    print(cm_df)
    
if __name__ == "__main__":
    run_inference()
