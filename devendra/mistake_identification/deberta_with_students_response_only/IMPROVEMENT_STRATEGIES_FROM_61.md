# 🚀 Improvement Strategies: From F1-Macro 0.61 → 0.70+

## Current Status
✅ **Achieved**: F1-Macro = 0.61 on test set  
📈 **Improvement from baseline**: +110% (from 0.29)  
🎯 **Target**: 0.65-0.75

---

## 🔥 High-Impact Strategies (Ordered by Expected Gain)

### 1. **Model Ensemble** [Expected: +3-8% = 0.63-0.66]
**Why it works**: Different models learn different patterns; averaging reduces variance.

#### Quick Implementation (3-5 models):
```python
# Train multiple models with different seeds/architectures
seeds = [42, 123, 456, 789, 2024]
models = []

for seed in seeds:
    # In configuration cell:
    RANDOM_SEED = seed
    
    # Train model
    trainer.train()
    
    # Save predictions
    models.append(trainer)

# Average predictions
ensemble_probs = np.mean([model.predict(test_dataset).predictions 
                          for model in models], axis=0)
```

**Implementation Time**: 15-25 hours (train 5 models)  
**Difficulty**: Easy  
**Expected Gain**: +3-5%

#### Advanced Ensemble (Different Architectures):
```python
architectures = [
    "microsoft/deberta-v3-large",      # Current
    "roberta-large",                    # Different architecture
    "microsoft/deberta-v2-xlarge",     # Larger model
    "FacebookAI/xlm-roberta-large",    # Multilingual
]
```
**Expected Gain**: +5-8%

---

### 2. **Hyperparameter Optimization** [Expected: +2-5% = 0.62-0.64]
**Why it works**: Your current hyperparameters may not be optimal for this dataset.

#### Use Optuna for Automated Search:
```python
import optuna

def objective(trial):
    # Sample hyperparameters
    focal_gamma = trial.suggest_float('focal_gamma', 2.0, 4.0)
    learning_rate = trial.suggest_float('learning_rate', 5e-6, 2e-5, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.05, 0.2)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    balance_strategy = trial.suggest_categorical('balance_strategy', 
                                                  ['adaptive', 'oversample', 'smote'])
    
    # Train with these hyperparameters
    # ... (use your existing training code)
    
    # Return validation F1-macro
    return val_f1_macro

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best F1-Macro: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

**Key Parameters to Tune**:
1. `FOCAL_GAMMA`: Try 2.0, 2.5, 3.0, 3.5, 4.0
2. `LEARNING_RATE`: Try 5e-6, 7e-6, 1e-5, 1.5e-5
3. `BALANCE_STRATEGY`: adaptive vs oversample vs hybrid
4. `LABEL_SMOOTHING`: 0.0, 0.05, 0.1, 0.15
5. `WARMUP_RATIO`: 0.05, 0.1, 0.15, 0.2

**Implementation Time**: 2-4 days (50 trials × 3-5 hours each)  
**Difficulty**: Medium  
**Expected Gain**: +2-5%

---

### 3. **Advanced Data Augmentation** [Expected: +1-3% = 0.61-0.63]
**Why it works**: More diverse training data for minority classes.

#### Back-Translation Augmentation:
```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, lang='fr'):
    """Translate to French and back to English for paraphrasing."""
    # English -> French
    model_name = f'Helsinki-NLP/opus-mt-en-{lang}'
    translate_to = MarianMTModel.from_pretrained(model_name)
    tokenizer_to = MarianTokenizer.from_pretrained(model_name)
    
    # Translate
    inputs = tokenizer_to(text, return_tensors="pt", padding=True)
    translated = translate_to.generate(**inputs)
    french_text = tokenizer_to.decode(translated[0], skip_special_tokens=True)
    
    # French -> English
    model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
    translate_back = MarianMTModel.from_pretrained(model_name)
    tokenizer_back = MarianTokenizer.from_pretrained(model_name)
    
    inputs = tokenizer_back(french_text, return_tensors="pt", padding=True)
    back_translated = translate_back.generate(**inputs)
    return tokenizer_back.decode(back_translated[0], skip_special_tokens=True)

# Augment minority classes
def augment_minority_data(df, min_samples=500):
    augmented = []
    for label in ['No', 'To some extent']:
        class_df = df[df['label'] == label]
        
        while len(class_df) + len(augmented) < min_samples:
            # Sample and augment
            sample = class_df.sample(1).iloc[0]
            augmented_text = back_translate(sample['student_responses'])
            
            augmented.append({
                'student_responses': augmented_text,
                'answer_key': sample['answer_key'],
                'tutor_response': sample['tutor_response'],
                'label': label
            })
    
    return pd.concat([df, pd.DataFrame(augmented)])
```

**Implementation Time**: 1-2 days  
**Difficulty**: Medium  
**Expected Gain**: +1-3%

---

### 4. **Longer Training with Better Scheduling** [Expected: +1-2% = 0.62]
**Why it works**: You only trained for 15 epochs; the model may not have converged.

#### Current Issues:
- `NUM_EPOCHS = 15` (stopped early)
- Early stopping patience = 5 (might stop too soon)

#### Recommended Changes:
```python
# In configuration cell:
NUM_EPOCHS = 40                    # Train longer
EARLY_STOPPING_PATIENCE = 8        # More patience
LR_SCHEDULER = 'polynomial'         # Better for long training

# Add learning rate warmup and decay
WARMUP_STEPS = 500                 # Warm up first 500 steps
```

#### Polynomial Decay with Warmup:
```python
from transformers import get_polynomial_decay_schedule_with_warmup

scheduler = get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps,
    lr_end=1e-7,
    power=2.0
)
```

**Implementation Time**: 8-12 hours (longer training)  
**Difficulty**: Easy  
**Expected Gain**: +1-2%

---

### 5. **Feature Engineering: Better Input Representation** [Expected: +2-4% = 0.63]
**Why it works**: Current input may not capture all relevant information.

#### Current Input Format:
```python
text = f"{student_responses} [SEP] {answer_key} [SEP] {tutor_response}"
```

#### Enhanced Input Format:
```python
def create_enhanced_input(row):
    """Add explicit markers and reasoning signals."""
    
    # Calculate student correctness indicator
    is_correct = "CORRECT" if "correct" in row['tutor_response'].lower() else "INCORRECT"
    
    # Add explicit structure
    text = f"""
    [STUDENT_SOLUTION] {row['student_responses']}
    [ANSWER_KEY] {row['answer_key']}
    [CORRECTNESS] {is_correct}
    [TUTOR_FEEDBACK] {row['tutor_response']}
    [CLASSIFY_MISTAKE]
    """.strip()
    
    return text
```

#### Add Conversation-Level Features:
```python
def extract_conversation_features(conv_history):
    """Extract meta-features from conversation."""
    features = {
        'num_turns': conv_history.count('Student:'),
        'has_equation': bool(re.search(r'[=+\-*/]', conv_history)),
        'has_number': bool(re.search(r'\d+', conv_history)),
        'avg_response_length': len(conv_history.split()) / max(1, conv_history.count('Student:'))
    }
    return features
```

**Implementation Time**: 1-2 days  
**Difficulty**: Medium  
**Expected Gain**: +2-4%

---

### 6. **Cost-Sensitive Learning** [Expected: +1-2% = 0.62]
**Why it works**: Penalize minority class errors more heavily.

#### Modify Focal Loss with Class-Specific Costs:
```python
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, class_costs=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Cost for misclassifying each class
        self.costs = class_costs if class_costs else torch.ones(3)
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - class_probs) ** self.gamma
        
        # Apply class-specific costs
        costs = self.costs[targets]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        return (focal_weight * ce_loss * costs).mean()

# Define costs: penalize minority errors more
class_costs = torch.tensor([1.0, 3.0, 2.5])  # [Yes, To some extent, No]
focal_loss = WeightedFocalLoss(alpha=class_weights, gamma=2.5, class_costs=class_costs)
```

**Implementation Time**: 1-2 hours  
**Difficulty**: Easy  
**Expected Gain**: +1-2%

---

### 7. **Two-Stage Training** [Expected: +2-3% = 0.63]
**Why it works**: First learn general patterns, then fine-tune on hard examples.

#### Stage 1: Learn from all data
```python
# Train on full balanced dataset
NUM_EPOCHS_STAGE1 = 15
train_stage1(train_df_balanced)
```

#### Stage 2: Focus on hard examples
```python
# Get predictions on training set
train_preds = model.predict(train_dataset)
train_probs = torch.softmax(torch.tensor(train_preds.predictions), dim=1)

# Identify hard examples (low confidence or misclassified)
confidence = train_probs.max(dim=1).values
hard_examples = confidence < 0.7  # Low confidence examples

# Train more on hard examples
hard_df = train_df_balanced[hard_examples]
NUM_EPOCHS_STAGE2 = 10
LEARNING_RATE_STAGE2 = 5e-6  # Lower learning rate
train_stage2(hard_df)
```

**Implementation Time**: 6-10 hours  
**Difficulty**: Medium  
**Expected Gain**: +2-3%

---

### 8. **Pseudo-Labeling (Semi-Supervised)** [Expected: +1-2% = 0.62]
**Why it works**: Use high-confidence predictions on unlabeled data as training data.

#### Implementation:
```python
# 1. Get high-confidence predictions on test set
test_probs = model.predict(test_dataset)
confidence = test_probs.max(dim=1).values
high_conf_mask = confidence > 0.85

# 2. Add high-confidence test samples to training
pseudo_labeled = test_df[high_conf_mask].copy()
pseudo_labeled['label'] = [id2label[i] for i in test_probs.argmax(dim=1)[high_conf_mask]]

# 3. Retrain with combined data
combined_df = pd.concat([train_df_balanced, pseudo_labeled])
retrain(combined_df)
```

**Implementation Time**: 4-6 hours  
**Difficulty**: Medium  
**Expected Gain**: +1-2%

---

### 9. **Mixup/CutMix for Text** [Expected: +0.5-1.5% = 0.61-0.62]
**Why it works**: Creates synthetic training examples by interpolating.

#### Text Mixup:
```python
def text_mixup(batch, alpha=0.2):
    """Mix two text sequences at token level."""
    lam = np.random.beta(alpha, alpha)
    
    # Mix two random samples
    idx = torch.randperm(len(batch))
    mixed_input_ids = lam * batch['input_ids'] + (1 - lam) * batch['input_ids'][idx]
    mixed_labels = lam * batch['labels'] + (1 - lam) * batch['labels'][idx]
    
    return mixed_input_ids, mixed_labels
```

**Implementation Time**: 2-4 hours  
**Difficulty**: Hard  
**Expected Gain**: +0.5-1.5%

---

### 10. **Error Analysis and Targeted Fixes** [Expected: +1-3% = 0.62-0.64]
**Why it works**: Fix systematic errors in predictions.

#### Perform Error Analysis:
```python
# Get predictions and labels
val_preds = trainer.predict(val_dataset)
y_pred = val_preds.predictions.argmax(axis=1)
y_true = val_preds.label_ids

# Analyze errors
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred, target_names=['Yes', 'To some extent', 'No']))

# Identify most confused pairs
# Example: If "To some extent" is often confused with "Yes"
# -> Add more explicit features to distinguish them
```

#### Common Error Patterns to Fix:
1. **"To some extent" vs "Yes"**: Add confidence markers
2. **"No" vs "To some extent"**: Add explicit error indicators
3. **Short responses**: May need different handling

**Implementation Time**: 1-2 days  
**Difficulty**: Medium  
**Expected Gain**: +1-3%

---

## 📊 Implementation Priority

### **Quick Wins (1-2 days, +2-4%)**:
1. ✅ **Tune FOCAL_GAMMA** (try 3.0, 3.5) - 1 hour
2. ✅ **Train longer** (40 epochs) - 10 hours
3. ✅ **Better threshold optimization** - 1 hour
4. ✅ **Cost-sensitive learning** - 2 hours

**Expected Result**: 0.61 → **0.63-0.65**

### **Medium Effort (3-5 days, +3-6%)**:
1. ✅ **Hyperparameter tuning with Optuna** - 2-4 days
2. ✅ **Two-stage training** - 1 day
3. ✅ **Enhanced input format** - 1 day

**Expected Result**: 0.61 → **0.64-0.67**

### **High Effort (1-2 weeks, +5-10%)**:
1. ✅ **Model ensemble (5 models)** - 1 week
2. ✅ **Back-translation augmentation** - 2-3 days
3. ✅ **Full hyperparameter grid search** - 3-5 days

**Expected Result**: 0.61 → **0.66-0.71**

---

## 🎯 Recommended Action Plan

### **Phase 1: Quick Wins (This Week)**
```python
# 1. Increase focal gamma
FOCAL_GAMMA = 3.5

# 2. Train longer
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8

# 3. Add class costs
class_costs = torch.tensor([1.0, 3.0, 2.5])
```
**Expected**: 0.61 → 0.63-0.65 (1 week)

### **Phase 2: Model Ensemble (Next Week)**
```python
# Train 5 models with different seeds
seeds = [42, 123, 456, 789, 2024]
# Average predictions
```
**Expected**: 0.63-0.65 → 0.66-0.68 (1 week)

### **Phase 3: Hyperparameter Tuning (Week 3)**
```python
# Use Optuna for 50 trials
study.optimize(objective, n_trials=50)
```
**Expected**: 0.66-0.68 → 0.68-0.71 (3-5 days)

---

## 🔧 Ready-to-Use Code Templates

### Template 1: Quick Ensemble (5 Models)
See: `ensemble_training.py` (to be created)

### Template 2: Optuna Hyperparameter Search
See: `hyperparameter_tuning.py` (to be created)

### Template 3: Two-Stage Training
See: `two_stage_training.py` (to be created)

---

## 📈 Expected Timeline

| Week | Action | Expected F1-Macro |
|------|--------|-------------------|
| Current | Baseline | 0.61 |
| Week 1 | Quick wins | 0.63-0.65 |
| Week 2 | Ensemble | 0.66-0.68 |
| Week 3 | Hyperparameter tuning | 0.68-0.71 |
| Week 4 | Polish + Error analysis | 0.70-0.73 |

---

## 🎯 Target

**Final Goal**: F1-Macro ≥ 0.70 (within 3-4 weeks)

**Realistic Target**: 0.68-0.72

**Stretch Goal**: 0.73-0.75 (with all techniques)

---

## 💡 Pro Tips

1. **Always validate on held-out set**: Don't overfit to validation set
2. **Track all experiments**: Use MLflow or Weights & Biases
3. **Start simple**: Quick wins first, then complex methods
4. **Monitor per-class F1**: Ensure no class is ignored
5. **Save all checkpoints**: You can ensemble later

---

## 📚 References

1. **Focal Loss**: Lin et al., ICCV 2017
2. **Class-Balanced Loss**: Cui et al., CVPR 2019
3. **Mixup**: Zhang et al., ICLR 2018
4. **Pseudo-Labeling**: Lee, ICML 2013 Workshop

---

**Next Step**: Which strategy would you like to try first? I recommend starting with **Quick Wins** (Phase 1).
