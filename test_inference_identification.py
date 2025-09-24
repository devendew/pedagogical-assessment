import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# --- 1. Model Class Definition (Same as training) ---

class GTEQwen2_1FF_Classifier(nn.Module):
    """
    Memory-optimized version following Averroes approach.
    Uses GTE-Qwen2-1.5B backbone with single feed-forward layer.
    """
    def __init__(self, backbone_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", num_labels=3):
        super().__init__()
        
        # Load backbone with memory optimizations
        self.backbone = AutoModel.from_pretrained(
            backbone_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            use_cache=False
        )
        
        # Get hidden size
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "n_embd", 1536)  # Default for 1.5B model
            
        # Single feed-forward layer (1FF)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Get embeddings from backbone
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                return_dict=True
            )
            
            # Use mean pooling (correct for GTE/encoder models)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            # Apply dropout and classification
            
            logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, logits)

# --- 2. Prompt Construction Function ---

def create_averroes_prompt(example):
   
    system_prompt = """You are an expert evaluator for an AI math tutoring system. Your role is to analyze a dialogue where a student has just made a mistake and an AI tutor has responded.

**Your Task:**
The student's last utterance contains a mathematical error. The AI tutor has just provided a response to this error. You must assess whether the tutor's response *successfully identifies* the mistake made by the student.

A response is considered a successful **mistake identifier** if it includes a question, explanation, or prompt that directly targets the student's misunderstanding or error in reasoning.

You will be given the conversation history and the tutor's immediate response.

- **<CONVERSATION_HISTORY>** contains the dialogue leading up to the student's error.
- **<TUTOR_RESPONSE>** contains the AI tutor's reply that you must evaluate.

Evaluate the <TUTOR_RESPONSE> and provide your final judgment as a single letter. These are your options:
- **A** → The response clearly focuses on the student's mistake and is directly relevant to the solution.
- **B** → The response is unrelated to the mistake, irrelevant to the solution steps, or confusing.
- **C** → The response is only partially relevant .

**Put Your Output In The Following Format:** <think>Your complete reasoning process</think><answer>Your final judgment (A, B, or C)</answer>"""

    prompt = (f"{system_prompt}\n\n"
              f"<CONVERSATION_HISTORY>\n{example['history']}\n</CONVERSATION_HISTORY>\n\n"
              f"<TUTOR_RESPONSE>\n{example['tutor_response']}\n</TUTOR_RESPONSE>")
    
    return {"text": prompt, "label": example["label"]}

# --- 3. Model Loading and Testing Functions ---

def load_trained_model(model_path="./gte-qwen2-1.5b-mistake-identification"):
    """
    Load the trained model, tokenizer, and label mappings.
    Updated to handle key cleaning and loading errors correctly.
    """
    print(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist!")

    # Load label mappings
    try:
        with open(f"{model_path}/label_mappings.pkl", "rb") as f:
            label_mappings = pickle.load(f)
            label2id = label_mappings["label2id"]
            id2label = label_mappings["id2label"]
    except FileNotFoundError:
        print("Label mappings not found, using default mappings...")
        label2id = {"Yes": 0, "To some extent": 1, "No": 2}
        id2label = {v: k for k, v in label2id.items()}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize model architecture
    model = GTEQwen2_1FF_Classifier(backbone_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- START OF FIX ---

    # Step 1: Load the state_dict from the file
    state_dict = None
    model_file = None
    
    if os.path.exists(f"{model_path}/model.safetensors"):
        try:
            from safetensors.torch import load_file
            model_file = f"{model_path}/model.safetensors"
            state_dict = load_file(model_file, device=str(device))
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
    elif os.path.exists(f"{model_path}/pytorch_model.bin"):
        model_file = f"{model_path}/pytorch_model.bin"
        state_dict = torch.load(model_file, map_location=device)
    else:
        raise FileNotFoundError(
            f"Could not find model weights in {model_path}. "
            "Looked for model.safetensors and pytorch_model.bin."
        )

    # Step 2: Clean the state_dict keys
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned_state_dict[k[6:]] = v  # Remove "model." prefix
        else:
            cleaned_state_dict[k] = v

    # Step 3: Load the cleaned state_dict into the model
    # Set strict=False to ignore non-essential keys like "class_weights"
    model.load_state_dict(cleaned_state_dict, strict=False)
    
    # --- END OF FIX ---

    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Model file used: {model_file}")
    print(f"Label mappings: {id2label}")

    return model, tokenizer, label2id, id2label

def load_test_data(test_file="test_set.csv"):
    """
    Load the test dataset.
    """
    print(f"Loading test data from {test_file}...")
    
    try:
        test_df = pd.read_csv(test_file)
        print(f"Test set size: {len(test_df)}")
        print(f"Test set label distribution:\n{test_df['label'].value_counts()}")
        return test_df
    except FileNotFoundError:
        print(f"Test file {test_file} not found!")
        return None

def evaluate_model(model, tokenizer, test_df, label2id, id2label, batch_size=8):
    """
    Evaluate the model on test data.
    """
    print("Starting model evaluation...")
    
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        # Process row by row from the dataframe
        for i in range(0, len(test_df), batch_size):
            batch_end = min(i + batch_size, len(test_df))
            batch_df = test_df.iloc[i:batch_end]
            
            # Create prompts for this batch
            texts = []
            labels = []
            
            for idx, row in batch_df.iterrows():
                # Create the prompt for this row
                example = {
                    "history": row["history"],
                    "tutor_response": row["tutor_response"],
                    "label": row["label"]
                }
                prompted_example = create_averroes_prompt(example)
                texts.append(prompted_example["text"])
                labels.append(label2id[row["label"]])
            
            # Tokenize batch
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            labels_tensor = torch.tensor(labels).to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_tensor)
            
            # Handle both dictionary and tuple outputs
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
                logits = outputs.get("logits")
            else:
                loss, logits = outputs
            
            # Convert logits to float32 to avoid BFloat16 issues with NumPy
            if logits.dtype == torch.bfloat16:
                logits = logits.float()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels)
            all_logits.extend(logits.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {batch_end}/{len(test_df)} samples")
    
    return all_predictions, all_labels, all_logits

def generate_evaluation_report(predictions, true_labels, id2label, save_path="./gte-qwen2-1.5b-mistake-identification"):
    """
    Generate comprehensive evaluation report with metrics and visualizations.
    """
    print("Generating evaluation report...")
    
    # Convert numerical labels to text
    pred_labels = [id2label[pred] for pred in predictions]
    true_labels_text = [id2label[label] for label in true_labels]
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    class_report = classification_report(true_labels_text, pred_labels, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(true_labels_text, pred_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels_text, pred_labels, labels=list(id2label.values()))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(id2label.values()), 
                yticklabels=list(id2label.values()))
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{save_path}/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class metrics visualization
    classes = list(id2label.values())
    precision = [class_report[cls]['precision'] for cls in classes]
    recall = [class_report[cls]['recall'] for cls in classes]
    f1_score = [class_report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics - Test Set')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/test_per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results_df = pd.DataFrame({
        'true_label': true_labels_text,
        'predicted_label': pred_labels,
        'correct': [t == p for t, p in zip(true_labels_text, pred_labels)]
    })
    results_df.to_csv(f'{save_path}/test_results.csv', index=False)
    
    # Save metrics summary
    metrics_summary = {
        'accuracy': accuracy,
        'macro_avg': class_report['macro avg'],
        'weighted_avg': class_report['weighted avg'],
        'per_class': {cls: class_report[cls] for cls in classes}
    }
    
    with open(f'{save_path}/test_metrics.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to {save_path}/")
    print(f"- test_confusion_matrix.png")
    print(f"- test_per_class_metrics.png") 
    print(f"- test_results.csv")
    print(f"- test_metrics.json")
    
    return accuracy, class_report

# --- 4. Main Testing Function ---

def main():
    """
    Main function to load model and evaluate on test set.
    """
    print("=== Model Testing and Evaluation ===\n")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load trained model
    model, tokenizer, label2id, id2label = load_trained_model()
    
    # Load test data
    test_df = load_test_data()
    if test_df is None:
        return
    
    # Evaluate model
    predictions, true_labels, logits = evaluate_model(
        model, tokenizer, test_df, label2id, id2label, batch_size=8
    )
    
    # Generate evaluation report
    accuracy, class_report = generate_evaluation_report(
        predictions, true_labels, id2label
    )
    
    print(f"\n=== Final Test Results ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {class_report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {class_report['weighted avg']['f1-score']:.4f}")

if __name__ == "__main__":
    main()
