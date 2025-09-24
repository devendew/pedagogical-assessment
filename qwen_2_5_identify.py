import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# --- 1. Data Loading (Simplified) ---

def load_json_from_file(filepath):
    """Loads JSON data from a specified file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None

def load_and_prepare_data_mistake_identification(data):
    """Prepares data specifically for Mistake Identification task."""
    prepared_data = []
    for item in data:
        history = item["conversation_history"]
        for _, details in item["tutor_responses"].items():
            if "Mistake_Identification" in details["annotation"]:
                prepared_data.append({
                    "history": history,
                    "tutor_response": details["response"],
                    "label": details["annotation"]["Mistake_Identification"]
                })
    return prepared_data




# --- 2. Prompt Construction (Averroes-style) ---

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
# --- 3. Compute Metrics Function ---

def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 score for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 4. GTE-Qwen2-1.5B-1FF Model (Memory Optimized) ---

class GTEQwen2_1FF_Classifier(nn.Module):
    """
    Memory-optimized version following Averroes approach.
    Uses GTE-Qwen2-1.5B backbone with single feed-forward layer.
    Supports class-weighted loss.
    """
    def __init__(self, backbone_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct", num_labels=3, class_weights=None):
        super().__init__()
        
        # Load backbone with memory optimizations
        self.backbone = AutoModel.from_pretrained(
            backbone_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to save memory
            use_cache=False
        )
        
        # Freeze backbone to save memory during training (optional)
        # Uncomment if memory is still an issue
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Get hidden size
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.backbone.config, "n_embd", 1536)  # Default for 1.5B model
            
        # # Single feed-forward layer (1FF)
        # self.classifier = nn.Linear(hidden_size, num_labels)
        # self.dropout = nn.Dropout(0.6)
# A more powerful two-layer classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_labels)
)        
        # Store class weights
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)



    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.
        Delegates to the backbone model.
        """
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            # Fallback for models that don't have this method
            if hasattr(self.backbone, 'config'):
                self.backbone.config.use_cache = False
            if hasattr(self.backbone, 'gradient_checkpointing'):
                self.backbone.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        Delegates to the backbone model.
        """
        if hasattr(self.backbone, 'gradient_checkpointing_disable'):
            self.backbone.gradient_checkpointing_disable()
        else:
            # Fallback for models that don't have this method
            if hasattr(self.backbone, 'gradient_checkpointing'):
                self.backbone.gradient_checkpointing = False

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
            #pooled = self.dropout(pooled)
            logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            # Use class-weighted loss if weights are provided
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, logits)

# --- 5. Training Setup (Memory Optimized) ---

def setup_mistake_identification_training(json_data):
    """
    Sets up training pipeline specifically for Mistake Identification task.
    Optimized for memory efficiency based on Averroes approach.
    """
    print("Setting up Mistake Identification pipeline (Averroes approach)")
    
    # Load and prepare data
    raw_data = load_and_prepare_data_mistake_identification(json_data)
    if not raw_data:
        return
    
    df = pd.DataFrame(raw_data)
    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")


    
    # Split data into 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    print(f"Train set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save test set for later use
    test_df.to_csv("test_set.csv", index=False)
    print("Test set saved as 'test_set.csv'")
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)


    # 3. Apply the new task-aware prompt to both datasets
    train_dataset = train_dataset.map(create_averroes_prompt)
    val_dataset = val_dataset.map(create_averroes_prompt)
    
    # Label mapping
    label2id = {"Yes": 0, "To some extent": 1, "No": 2}
    id2label = {v: k for k, v in label2id.items()}
    
    # Model and tokenizer setup
    backbone_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Use shorter max_length to save memory
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=512
        )
        tokenized["labels"] = [label2id[lbl] for lbl in examples["label"]]
        return tokenized
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
    
    # Calculate class weights for imbalanced dataset
    train_labels = [label2id[label] for label in train_df['label']]
    unique_labels = np.unique(train_labels)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_labels
    )
    
    # Convert to torch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    print(f"Class weights: {dict(zip([id2label[i] for i in unique_labels], class_weights.cpu().numpy()))}")

    
    # Initialize model with class weights
    model = GTEQwen2_1FF_Classifier(backbone_name, num_labels=3, class_weights=class_weights)
    
    # Training arguments (following Averroes settings)
    training_args = TrainingArguments(
        output_dir="./gte-qwen2-1.5b-mistake-identification",
        num_train_epochs=5,
        
        per_device_train_batch_size=4,  # Reduced batch size for memory
        gradient_accumulation_steps=16,  # Maintain effective batch size of 64
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,
        dataloader_pin_memory=False,  # Save memory
        gradient_checkpointing=True,   # Save memory at cost of speed
        logging_dir='./logs',
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,  # Higher F1 is better
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=True,
    )
  
    # Optimizer with linear schedule (as used by Averroes)
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)
    
    # Calculate training steps
    train_dataloader_len = len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    num_training_steps = train_dataloader_len * training_args.num_train_epochs

    #num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Create trainer with compute_metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics  # Add metrics computation
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save final model and tokenizer
    trainer.save_model("./gte-qwen2-1.5b-mistake-identification")
    tokenizer.save_pretrained("./gte-qwen2-1.5b-mistake-identification")
    
    # Save label mappings
    import pickle
    with open("./gte-qwen2-1.5b-mistake-identification/label_mappings.pkl", "wb") as f:
        pickle.dump({"label2id": label2id, "id2label": id2label}, f)
    
    print("Training completed and model saved!")
    
    # Plot training and validation loss
    plot_training_loss(trainer)
    
    return trainer

def plot_training_loss(trainer):
    """Plot training and validation loss curves."""
    # Extract loss values from trainer logs
    train_losses = []
    val_losses = []
    steps = []
    
    for log in trainer.state.log_history:
        if "loss" in log and "learning_rate" in log:  # typical training logs have these
            train_losses.append(log["loss"])
            steps.append(log["step"])

        if "eval_loss" in log:
            val_losses.append(log["eval_loss"])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot validation loss
    plt.subplot(1, 2, 2)
    eval_steps = [log["step"] for log in trainer.state.log_history if "eval_loss" in log]
    plt.plot(eval_steps, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./gte-qwen2-1.5b-mistake-identification/training_validation_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also plot both losses on same graph
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(eval_steps, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./gte-qwen2-1.5b-mistake-identification/combined_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Loss plots saved to model directory!")

# --- 6. Main execution ---

if __name__ == "__main__":
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    json_filepath = 'data/trainset.json'
    loaded_data = load_json_from_file(json_filepath)
    
    if loaded_data:
        trainer = setup_mistake_identification_training(loaded_data)
    else:
        print("Failed to load data.")