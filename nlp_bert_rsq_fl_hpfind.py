
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


dataset = pd.read_json('/DATA/cs24mtech12016/repos/pedagogical-assessment/data/trainset.json')
dataset

#tutor names in the tutor_responses column

unique_tutor_names = dataset['tutor_responses'].explode().unique()
unique_tutor_names

df = dataset.copy()

# Define all tutor names
all_tutors = ['Llama318B','Llama31405B', 'Phi3', 'Expert', 'Gemini', 'Novice', 'Mistral', 'GPT4', 'Sonnet']

# Create new DataFrame with desired structure
new_df = pd.DataFrame(columns=[
    'conversation_id', 'conversation_history', 'tutor_name', 'response',
    'Mistake_Identification', 'Providing_Guidance'
])

# Populate the DataFrame
for index, row in df.iterrows():
    conversation_id = row['conversation_id']
    conversation_history = row['conversation_history']

    try:
        tutor_responses = row['tutor_responses']

        # Iterate over all tutors, ensuring all are included
        for tutor_name in all_tutors:
            response_data = tutor_responses.get(tutor_name)  # Get data for current tutor


            if response_data is not None:  # Check if tutor data exists
                response = response_data.get('response', np.nan)  # Use np.nan for missing response
                annotation = response_data.get('annotation', {})
            else:
                response = np.nan
                annotation = {}

            new_row = {
                'conversation_id': conversation_id,
                'conversation_history': conversation_history,
                'tutor_name': tutor_name,
                'response': response,
                'Mistake_Identification': annotation.get('Mistake_Identification', np.nan),
                
                'Providing_Guidance': annotation.get('Providing_Guidance', np.nan),
                
            }
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    except (KeyError, TypeError):
        print(f"Skipping row {index} due to missing or invalid data.")

df=new_df.copy()

df.head(5)

df.columns

import pandas as pd
import re
import numpy as np
from transformers import AutoTokenizer

# Load the tokenizer for accurate token counting
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

def split_conversation_by_speaker(conversation_history):
    """
    Splits a conversation history into tutor and student parts.
    All text following "Tutor:" goes into tutor_text until "Student:" appears, and vice versa.
    """
    lines = conversation_history.split('\n')

    tutor_text = []
    student_text = []
    current_speaker = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Tutor:"):
            current_speaker = "tutor"
            # Remove the "Tutor:" prefix
            content = line[len("Tutor:"):].strip()
            if content:  # Only add if there's content after the prefix
                tutor_text.append(content)
        elif line.startswith("Student:"):
            current_speaker = "student"
            # Remove the "Student:" prefix
            content = line[len("Student:"):].strip()
            if content:  # Only add if there's content after the prefix
                student_text.append(content)
        else:
            # Continue with the current speaker
            if current_speaker == "tutor":
                tutor_text.append(line)
            elif current_speaker == "student":
                student_text.append(line)

    # Join all tutor and student parts
    tutor_content = " ".join(tutor_text)
    student_content = " ".join(student_text)

    return tutor_content, student_content

def analyze_speaker_lengths(df):
    """
    Analyzes the length of tutor and student parts in conversations.
    """
    tutor_word_counts = []
    student_word_counts = []
    tutor_token_counts = []
    student_token_counts = []
    combined_token_counts = []

    for _, row in df.iterrows():
        tutor_text, student_text = split_conversation_by_speaker(row['conversation_history'])

        # Count words
        tutor_words = len(tutor_text.split())
        student_words = len(student_text.split())

        # Count tokens using BERT tokenizer
        tutor_tokens = len(tokenizer.encode(tutor_text))
        student_tokens = len(tokenizer.encode(student_text))
        combined_tokens = len(tokenizer.encode(tutor_text + " " + student_text))

        tutor_word_counts.append(tutor_words)
        student_word_counts.append(student_words)
        tutor_token_counts.append(tutor_tokens)
        student_token_counts.append(student_tokens)
        combined_token_counts.append(combined_tokens)

    # Calculate statistics
    avg_tutor_words = np.mean(tutor_word_counts)
    avg_student_words = np.mean(student_word_counts)

    avg_tutor_tokens = np.mean(tutor_token_counts)
    avg_student_tokens = np.mean(student_token_counts)
    avg_combined_tokens = np.mean(combined_token_counts)

    max_tutor_tokens = max(tutor_token_counts)
    max_student_tokens = max(student_token_counts)
    max_combined_tokens = max(combined_token_counts)

    # Count conversations that would exceed BERT's token limit
    exceed_limit_count = sum(1 for count in combined_token_counts if count > 700)
    exceed_limit_percentage = (exceed_limit_count / len(combined_token_counts)) * 100

    return {
        "word_stats": {
            "tutor": {
                "average": avg_tutor_words,
                "max": max(tutor_word_counts)
            },
            "student": {
                "average": avg_student_words,
                "max": max(student_word_counts)
            }
        },
        "token_stats": {
            "tutor": {
                "average": avg_tutor_tokens,
                "max": max_tutor_tokens,
                "counts": tutor_token_counts
            },
            "student": {
                "average": avg_student_tokens,
                "max": max_student_tokens,
                "counts": student_token_counts
            },
            "combined": {
                "average": avg_combined_tokens,
                "max": max_combined_tokens,
                "exceed_limit_count": exceed_limit_count,
                "exceed_limit_percentage": exceed_limit_percentage,
                "counts": combined_token_counts
            }
        }
    }

def add_speaker_texts_to_df(df):
    """
    Adds separate columns for tutor text and student text to the dataframe.
    """
    tutor_texts = []
    student_texts = []
    tutor_token_counts = []
    student_token_counts = []
    combined_token_counts = []

    for _, row in df.iterrows():
        tutor_text, student_text = split_conversation_by_speaker(row['conversation_history'])
        tutor_texts.append(tutor_text)
        student_texts.append(student_text)

        # Also add token counts
        tutor_tokens = len(tokenizer.encode(tutor_text))
        student_tokens = len(tokenizer.encode(student_text))
        combined_tokens = len(tokenizer.encode(tutor_text + " " + student_text))

        tutor_token_counts.append(tutor_tokens)
        student_token_counts.append(student_tokens)
        combined_token_counts.append(combined_tokens)

    df['tutor_text'] = tutor_texts
    df['student_text'] = student_texts
    df['tutor_tokens'] = tutor_token_counts
    df['student_tokens'] = student_token_counts
    df['combined_tokens'] = combined_token_counts

    return df

# Example usage
results = analyze_speaker_lengths(df)
print(f"Average Tutor Words: {results['word_stats']['tutor']['average']:.2f}")
print(f"Average Student Words: {results['word_stats']['student']['average']:.2f}")
print(f"Average Tutor Tokens: {results['token_stats']['tutor']['average']:.2f}")
print(f"Maximum Tutor Tokens: {results['token_stats']['tutor']['max']}")
print(f"Average Student Tokens: {results['token_stats']['student']['average']:.2f}")
print(f"Maximum Student Tokens: {results['token_stats']['student']['max']}")
print(f"Average Combined Tokens: {results['token_stats']['combined']['average']:.2f}")
print(f"Maximum Combined Tokens: {results['token_stats']['combined']['max']}")
print(f"Conversations Exceeding 700 Tokens: {results['token_stats']['combined']['exceed_limit_count']} ({results['token_stats']['combined']['exceed_limit_percentage']:.2f}%)")

# Add speaker texts to your dataframe
df = add_speaker_texts_to_df(df)

df["response"][0]

df["tutor_text"][0]

df["conversation_history"][0]

def extract_question(conversation):
    lines = conversation.strip().split('\xa0')  # Split on \xa0 which behaves like newlines in your data
    question_parts = []
    collecting = False

    for line in lines:
        line = line.strip()
        if line.startswith("Tutor:"):
            content = line[len("Tutor:"):].strip()
            if "The question is:" in content:
                # Start collecting right after "The question is:"
                question_start = content.split("The question is:", 1)[-1].strip()
                question_parts.append(question_start)
                collecting = True
            elif collecting:
                question_parts.append(content)
        elif line.startswith("Student:") and collecting:
            break
        elif collecting:
            question_parts.append(line)

    return " ".join(question_parts).strip()

df['question'] = df['conversation_history'].apply(extract_question)

df["question"][0]

df["tutor_text"] = df["tutor_text"].apply(lambda x: x.replace('\xa0', ' '))
df["student_text"] = df["student_text"].apply(lambda x: x.replace('\xa0', ' '))
df["conversation_history"] = df["conversation_history"].apply(lambda x: x.replace('\xa0', ' '))

df["tutor_text"][0]

df["conversation_history"][0]

df["student_text"][0]

# Example usage
results = analyze_speaker_lengths(df)
print(f"Average Tutor Words: {results['word_stats']['tutor']['average']:.2f}")
print(f"Average Student Words: {results['word_stats']['student']['average']:.2f}")
print(f"Average Tutor Tokens: {results['token_stats']['tutor']['average']:.2f}")
print(f"Maximum Tutor Tokens: {results['token_stats']['tutor']['max']}")
print(f"Average Student Tokens: {results['token_stats']['student']['average']:.2f}")
print(f"Maximum Student Tokens: {results['token_stats']['student']['max']}")
print(f"Average Combined Tokens: {results['token_stats']['combined']['average']:.2f}")
print(f"Maximum Combined Tokens: {results['token_stats']['combined']['max']}")
print(f"Conversations Exceeding 700 Tokens: {results['token_stats']['combined']['exceed_limit_count']} ({results['token_stats']['combined']['exceed_limit_percentage']:.2f}%)")

# Add speaker texts to your dataframe
df = add_speaker_texts_to_df(df)

# Stratified train/validation/test split (using main task for stratification)
# Drop rows with NaN values in 'Mistake_Identification' before splitting
from sklearn.model_selection import train_test_split
import pandas as pd

df = df.dropna(subset=['Mistake_Identification'])

train_val_df, test_df = train_test_split(
    df, stratify=df['Mistake_Identification'], test_size=0.15, random_state=42
)
val_size = 0.176
train_df, val_df = train_test_split(
    train_val_df, stratify=train_val_df['Mistake_Identification'], test_size=val_size, random_state=42
)

"""##Model"""

import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
import optuna
from optuna.pruners import MedianPruner
import json

import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.auto import tqdm
import optuna
from optuna.pruners import MedianPruner
import json

# 1. Label mapping
def map_labels(df):
    label_map = {"No": 0, "To some extent": 1, "Yes": 2}
    for col in [
        'Mistake_Identification',
        
        'Providing_Guidance'
        
    ]:
        df[col] = df[col].map(label_map)
    return df

# 2. Tokenizer and token-length check
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def check_token_lengths(df, max_len=512):
    texts = df['response'] +" [SEP] " + df['student_text']+ " [SEP] " + df['question']
    token_counts = texts.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
    num_exceed = token_counts.gt(max_len).sum()
    print(f"{num_exceed} out of {len(df)} examples exceed {max_len} tokens.")
    return token_counts

# 3. Dataset class
class TutorDataset(Dataset):
    def __init__(self, df, max_len=512):
        self.texts = (df['response'] +" [SEP] " + df['student_text']+ " [SEP] " + df['question'] ).tolist()
        self.labels = df[[
            'Mistake_Identification',
            
            'Providing_Guidance',
            
        ]].values
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        enc = tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# 4. Modified BERT multi-task model with configurable hyperparameters
class MultiTaskBert(nn.Module):
    def __init__(self, dropout_rate=0.1, freeze_layers=6, fc_hidden_size=1024):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad=False

        # Freeze specified number of encoder layers
        for layer in self.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad=False

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        self.act = nn.ReLU()
        self.ident_head  = nn.Linear(fc_hidden_size, 3)
        
        self.guide_head  = nn.Linear(fc_hidden_size, 3)
        

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.pooler_output
        x = self.dropout(x)
        x = self.act(self.fc1(x)); x=self.dropout(x)
        x = self.act(self.fc2(x)); x=self.dropout(x)
        return (self.ident_head(x), self.guide_head(x))

def compute_loss(logits, labels, alpha=0.25, gamma=2.0, weights=None):
    """
    Compute the focal loss for multi-task classification.
    Args:
        logits: List of logits for each task.
        labels: Ground truth labels for each task.
        alpha: Balancing factor for focal loss.
        gamma: Focusing parameter for focal loss.
        weights: Task weights for balancing different tasks.
    Returns:
        Total focal loss across all tasks.
    """
    if weights is None:
        weights = [1.4271065569828554, 0.8690676699308719]  # Setting weights for all tasks

    loss_fn = nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to compute per-sample loss
    total = 0
    for i, log in enumerate(logits):
        ce_loss = loss_fn(log, labels[:, i].to(log.device))  # Compute cross-entropy loss
        pt = torch.exp(-ce_loss)  # Compute the probability of the correct class
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss  # Apply focal loss formula
        task_loss = focal_loss.mean()  # Take the mean of the focal loss
        total += weights[i] * task_loss  # Apply task weight
    return total

# 5. Epoch functions with progress bars - modified to accept focal loss parameters
def train_epoch(model, loader, optimizer, scheduler, device, alpha=0.25, gamma=2.0, weights=None):
    model.train(); total_loss=0
    loop=tqdm(loader, desc='Training', leave=False)
    for batch in loop:
        optimizer.zero_grad(); inputs={k:batch[k].to(device) for k in ['input_ids','attention_mask','labels']}
        logits=model(inputs['input_ids'], inputs['attention_mask'])
        loss=compute_loss(logits, inputs['labels'], alpha, gamma, weights);
        loss.backward(); optimizer.step(); scheduler.step()
        total_loss+=loss.item(); loop.set_postfix(loss=loss.item())
    return total_loss/len(loader)

def eval_epoch(model, loader, device, alpha=0.25, gamma=2.0, weights=None):
    model.eval(); total_loss=0
    loop=tqdm(loader, desc='Validation', leave=False)
    with torch.no_grad():
        for batch in loop:
            inputs={k:batch[k].to(device) for k in ['input_ids','attention_mask','labels']}
            logits=model(inputs['input_ids'], inputs['attention_mask'])
            loss=compute_loss(logits, inputs['labels'], alpha, gamma, weights);
            total_loss+=loss.item(); loop.set_postfix(loss=loss.item())
    return total_loss/len(loader)

# 6. Metric evaluation - modified to return average F1
def eval_model(model, loader, device):
    model.eval(); all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids, att = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels=batch['labels'].cpu().numpy()
            logits=model(input_ids=input_ids, attention_mask=att)
            preds=[torch.argmax(l,1).cpu().numpy() for l in logits]
            all_labels.append(labels); all_preds.append(np.stack(preds,1))
    all_labels=np.vstack(all_labels); all_preds=np.vstack(all_preds)
    metrics={}; tasks=['identification','guidance']
    for i,t in enumerate(tasks):
        y_true, y_pred = all_labels[:,i], all_preds[:,i]
        metrics[f"{t}_acc"]=accuracy_score(y_true,y_pred)
        metrics[f"{t}_f1"]=f1_score(y_true,y_pred,average='macro')
        # y_true_b, y_pred_b = (y_true!=0).astype(int),(y_pred!=0).astype(int)
        # metrics[f"{t}_lenient_acc"]=accuracy_score(y_true_b,y_pred_b)
        # metrics[f"{t}_lenient_f1"]=f1_score(y_true_b,y_pred_b,average='macro')

    # Calculate average F1 score across all dimensions
    metrics["avg_f1"] = np.mean([metrics[f"{t}_f1"] for t in tasks])

    return metrics

# 7. Helper for confusion matrices
def get_preds_labels(model, loader, device, lenient=False):
    model.eval(); all_lbl, all_pr=[],[]
    with torch.no_grad():
        for batch in loader:
            ids, att = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            lbls = batch['labels'].numpy()
            logits = model(ids, att)
            preds = np.stack([torch.argmax(l,1).cpu().numpy() for l in logits],1)
            if lenient:
                lbls=(lbls!=0).astype(int); preds=(preds!=0).astype(int)
            all_lbl.append(lbls); all_pr.append(preds)
    return np.vstack(all_lbl), np.vstack(all_pr)

# 8. Modified training loop with configurable hyperparameters
def train_loop(model, tr_loader, val_loader, device, epochs=25, lr=2.4352796960051026e-05,
                patience=5, alpha=0.25, gamma=2.0, weights=None, outdir=None, weight_decay=0.01):
    # Check token lengths
    check_token_lengths(train_df); check_token_lengths(val_df); check_token_lengths(test_df)

    # Create optimizer and scheduler
    opt=AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=0,num_training_steps=epochs*len(tr_loader))

    # Initialize tracking variables
    train_losses,val_losses=[],[]; best_f1,wait=0,0

    # Initialize dictionaries to store task-specific metrics
    tasks = ['identification', 'guidance']
    train_task_acc = {t: [] for t in tasks}
    val_task_acc = {t: [] for t in tasks}


    # Create output directory
    if outdir is None:
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir=f"output_BERT_RSQ_FL_HPFIND_{ts}"
    os.makedirs(outdir,exist_ok=True)

    # Save hyperparameters
    if hasattr(model, 'dropout'):
        dropout_rate = model.dropout.p
    else:
        dropout_rate = 0.1

    # Count frozen layers
    frozen_layers = 0
    for i, layer in enumerate(model.bert.encoder.layer):
        if not any(p.requires_grad for p in layer.parameters()):
            frozen_layers += 1

    # Get FC hidden size
    fc_hidden_size = model.fc1.out_features

    hyperparams = {
        "learning_rate": lr,
        "batch_size": tr_loader.batch_size,
        "epochs": epochs,
        "patience": patience,
        "alpha": alpha,
        "gamma": gamma,
        "weights": weights if weights else [1.427, 0.869],
        "dropout_rate": dropout_rate,
        "freeze_layers": frozen_layers,
        "fc_hidden_size": fc_hidden_size
        "weight_decay": weight_decay
    }

    with open(os.path.join(outdir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)

    tasks=['identification','guidance']

    # Training loop
    for e in range(1,epochs+1):
        st=time.time();
        t_loss=train_epoch(model, tr_loader, opt, sched, device, alpha, gamma, weights)
        v_loss=eval_epoch(model, val_loader, device, alpha, gamma, weights)
        elapsed=time.time()-st

        train_losses.append(t_loss); val_losses.append(v_loss)

        # Get training metrics
        train_metrics = eval_model(model, tr_loader, device)
        val_metrics = eval_model(model, val_loader, device)
        
        # Store accuracy for all tasks
        for t in tasks:
            train_task_acc[t].append(train_metrics[f'{t}_acc'])
            val_task_acc[t].append(val_metrics[f'{t}_acc'])

        # Validation metrics
        val_metrics=eval_model(model,val_loader,device)
        print(f"Epoch no: {e}, Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")

        for t in tasks:
            print(f"  {t}_acc: {val_metrics[f'{t}_acc']:.4f}")
            print(f"  {t}_f1: {val_metrics[f'{t}_f1']:.4f}")
            # print(f"  {t}_lenient_acc: {val_metrics[f'{t}_lenient_acc']:.4f}")
            # print(f"  {t}_lenient_f1: {val_metrics[f'{t}_lenient_f1']:.4f}")

        print(f"  avg_f1: {val_metrics['avg_f1']:.4f}")

        # Save model if improved
        if val_metrics['avg_f1'] > best_f1:
            best_f1 = val_metrics['avg_f1']
            wait = 0
            torch.save(model.state_dict(), os.path.join(outdir, 'best_model.pt'))
            print(f"  New best model saved (avg_f1: {best_f1:.4f})")
        else:
            wait += 1
            print(f"  No improvement {wait}/{patience}")

        # Early stopping
        if wait >= patience:
            print("Early stopping triggered.")
            break

    # Test metrics
    test_loader = DataLoader(TutorDataset(test_df), batch_size=tr_loader.batch_size)

    # Load best model
    model.load_state_dict(torch.load(os.path.join(outdir, 'best_model.pt')))
    test_metrics = eval_model(model, test_loader, device)

    # Save test metrics
    with open(os.path.join(outdir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print("Test Set Evaluation:")
    for t in tasks:
        print(f"  {t}_acc: {test_metrics[f'{t}_acc']:.4f}")
        print(f"  {t}_f1: {test_metrics[f'{t}_f1']:.4f}")
        # print(f"  {t}_lenient_acc: {test_metrics[f'{t}_lenient_acc']:.4f}")
        # print(f"  {t}_lenient_f1: {test_metrics[f'{t}_lenient_f1']:.4f}")
    print(f"  avg_f1: {test_metrics['avg_f1']:.4f}")

    # # Plot loss
    # plt.figure(); plt.plot(train_losses); plt.plot(val_losses)
    # plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(['Train','Val']); plt.savefig(os.path.join(outdir,'loss_curve.png')); plt.close()

    # # Confusion matrices - exact
    # y_true,y_pred=get_preds_labels(model,test_loader,device,lenient=False)
    # for i,t in enumerate(tasks):
    #     cm=confusion_matrix(y_true[:,i],y_pred[:,i]); plt.figure(); plt.imshow(cm); plt.title(f'CM_{t}_exact'); plt.colorbar(); plt.savefig(os.path.join(outdir,f'cm_{t}_exact.png')); plt.close()

    # # Confusion matrices - lenient
    # y_true_l,y_pred_l=get_preds_labels(model,test_loader,device,lenient=True)
    # for i,t in enumerate(tasks):
    #     cm=confusion_matrix(y_true_l[:,i],y_pred_l[:,i]); plt.figure(); plt.imshow(cm); plt.title(f'CM_{t}_lenient'); plt.colorbar(); plt.savefig(os.path.join(outdir,f'cm_{t}_lenient.png')); plt.close()

    # Make sure seaborn and numpy are imported
    import seaborn as sns
    import numpy as np

    # plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.plot(val_losses, 'r-', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(['Train', 'Val'], fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(outdir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot accuracy for each task
    for t in tasks:
        plt.figure(figsize=(10, 6))
        plt.plot(train_task_acc[t], 'b-', linewidth=2)
        plt.plot(val_task_acc[t], 'r-', linewidth=2)
        plt.title(f'{t.capitalize()} - Training and Validation Accuracy', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(['Train', 'Val'], fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(outdir, f'{t}_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot all tasks validation accuracy in one figure
    plt.figure(figsize=(12, 7))
    for t in tasks:
        plt.plot(val_task_acc[t], linewidth=2, label=t.capitalize())
    plt.title('Validation Accuracy - All Tasks', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(outdir, 'all_tasks_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==== CONFUSION MATRIX PLOTTING ====

    # Fixed label names
    exact_labels = ["No", "To some extent", "Yes"]
    #lenient_labels = ["No", "Yes"]

    # Exact Confusion Matrices
    y_true, y_pred = get_preds_labels(model, test_loader, device, lenient=False)
    for i, t in enumerate(tasks):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1, 2])  # 0=No, 1=To some extent, 2=Yes
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=exact_labels, yticklabels=exact_labels)

        plt.title(f'Confusion Matrix - {t.capitalize()} (Exact)', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'cm_{t}_exact.png'), dpi=300)
        plt.close()

    # Lenient Confusion Matrices
    # y_true_l, y_pred_l = get_preds_labels(model, test_loader, device, lenient=True)
    # for i, t in enumerate(tasks):
    #     cm = confusion_matrix(y_true_l[:, i], y_pred_l[:, i], labels=[0, 1])  # 0=No, 1=Yes
    #     plt.figure(figsize=(8, 6))

    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=lenient_labels, yticklabels=lenient_labels)

    #     plt.title(f'Confusion Matrix - {t.capitalize()} (Lenient)', fontsize=14)
    #     plt.xlabel('Predicted Label', fontsize=12)
    #     plt.ylabel('True Label', fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outdir, f'cm_{t}_lenient.png'), dpi=300)
    #     plt.close()


    print(f"Outputs saved to {outdir}")
    return best_f1, test_metrics['avg_f1']

# Define objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    trial_dir = f"optuna_trial/trial_{trial.number}"

    # Model architecture hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True) 
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3,0.4])
    freeze_layers = trial.suggest_categorical("freeze_layers", [0, 2, 4, 6, 8])
    fc_hidden_size = trial.suggest_categorical("fc_hidden_size", [512, 768, 1024])

    # Focal loss hyperparameters
    alpha = trial.suggest_float("alpha", 0.1, 0.5)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)

    # Task weighting hyperparameters
    tune_weights = trial.suggest_categorical("tune_weights", [True, False])

    if tune_weights:
        id_weight = trial.suggest_float("id_weight", 0.5, 2.0)
        
        guide_weight = trial.suggest_float("guide_weight", 0.5, 2.0)
        
        weights = [id_weight, guide_weight]
    else:
        # Use default weights from your previous optimization
        weights = [1.0, 1.0]

    # Create model with trial hyperparameters
    model = MultiTaskBert(
        dropout_rate=dropout_rate,
        freeze_layers=freeze_layers,
        fc_hidden_size=fc_hidden_size
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)    

    # Create dataloaders with trial batch size
    tr_dl = DataLoader(TutorDataset(train_df), batch_size=batch_size, shuffle=True)
    vl_dl = DataLoader(TutorDataset(val_df), batch_size=batch_size, shuffle=False)



    # Train with fewer epochs for hyperparameter tuning
    try:
        val_f1, _ = train_loop(
            model=model,
            tr_loader=tr_dl,
            val_loader=vl_dl,
            device=device,
            epochs=10,  # Reduced for faster tuning
            lr=lr,
            weight_decay=weight_decay,
            patience=5,  # Reduced for faster tuning
            alpha=alpha,
            gamma=gamma,
            weights=weights,
            outdir=trial_dir
        )
        return val_f1
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0  # Return a low score for failed trials

# 9. Main with hyperparameter tuning
if __name__=='__main__':
    start_time = datetime.now()

    train_df=map_labels(train_df); val_df=map_labels(val_df); test_df=map_labels(test_df)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using the device: {device}")

    # Hyperparameter tuning
    print("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    study.optimize(objective, n_trials=100)  # Adjust number of trials as needed

    # Print best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # # Load the best hyperparameters from the JSON file
    # with open('best_hyperparameters.json', 'r') as f:
    #     content = f.read()
    #     print(f"File content: {content}")  # This will print the raw content of the file
    #     hyper = json.loads(content)

    # Extract hyperparameters from the best trial
    best_lr = trial.params["learning_rate"]
    best_batch_size = trial.params["batch_size"]
    best_dropout_rate = trial.params["dropout_rate"]
    best_freeze_layers = trial.params["freeze_layers"]
    best_fc_hidden_size = trial.params["fc_hidden_size"]
    best_alpha = trial.params["alpha"]
    best_gamma = trial.params["gamma"]

    # Set task weights
    if trial.params.get("tune_weights", False):  # default to False if not present
        best_weights = [
            trial.params["id_weight"],
            
            trial.params["guide_weight"],
            
        ]
    else:
        best_weights = [1.4271065569828554, 0.8690676699308719]

    # Create final model and dataloaders
    final_model = MultiTaskBert(
        dropout_rate=best_dropout_rate,
        freeze_layers=best_freeze_layers,
        fc_hidden_size=best_fc_hidden_size
    )

    if torch.cuda.device_count( >0:
        print(f"Using {torch.cuda.device_count()} GPUs")
        final_model = nn.DataParallel(final_model)

    final_model.to(device)

    final_tr_dl = DataLoader(TutorDataset(train_df), batch_size=best_batch_size, shuffle=True)
    final_vl_dl = DataLoader(TutorDataset(val_df), batch_size=best_batch_size, shuffle=False)

    # Train final model with full settings
    final_outdir = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _, test_f1 = train_loop(
        model=final_model,
        tr_loader=final_tr_dl,
        val_loader=final_vl_dl,
        device=device,
        epochs=100,
        lr=best_lr,
        patience=10,
        alpha=best_alpha,
        gamma=best_gamma,
        weights=best_weights,
        outdir=final_outdir
    )

    print(f"Final model test average F1: {test_f1:.4f}")

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")
