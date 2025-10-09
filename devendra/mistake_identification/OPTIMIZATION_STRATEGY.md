# 🎯 Optimization Strategy: Push F1-Macro Above 0.70

## ✅ Changes Implemented

### 1. **Composite Metric (Primary Change)**
**What:** Changed from pure eval_loss to a balanced composite score
```python
composite_score = 0.6 * f1_macro + 0.4 * (1 - normalized_loss)
```

**Why This Works:**
- **60% F1-Macro:** Directly optimizes your target metric
- **40% Generalization:** Prevents overfitting by penalizing high loss
- **Balanced Approach:** Gets the best of both worlds

**Expected Impact:** +3-5% F1-Macro improvement while maintaining generalization

---

## 🚀 Additional Strategies to Push Beyond 0.70

### 2. **Data Augmentation** (High Impact - Recommended)

Add these techniques to expand your training data:

#### Back-Translation Augmentation
```python
# Install: pip install googletrans==4.0.0-rc1
from googletrans import Translator

def back_translate(text, intermediate_lang='fr'):
    """Translate to French and back to English for paraphrasing"""
    translator = Translator()
    intermediate = translator.translate(text, dest=intermediate_lang).text
    back = translator.translate(intermediate, dest='en').text
    return back

# Apply to minority classes only
for idx, row in train_df[train_df['label'] == 'minority_class'].iterrows():
    augmented = back_translate(row['text'])
    # Add to training set
```

#### Synonym Replacement
```python
# Install: pip install nlpaug
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(original_text)
```

**Expected Impact:** +4-7% F1-Macro (especially for minority classes)

---

### 3. **Ensemble Methods** (Very High Impact)

#### Strategy A: Train Multiple Models
```python
# Train 5 models with different seeds/hyperparameters
models = []
for seed in [42, 123, 456, 789, 1011]:
    model = train_with_seed(seed, best_params)
    models.append(model)

# Average predictions (soft voting)
predictions = np.mean([model.predict(X) for model in models], axis=0)
```

#### Strategy B: Different Architectures
```python
models = [
    'microsoft/deberta-v3-large',      # Current
    'roberta-large',                   # Alternative
    'microsoft/deberta-v2-xlarge',     # Bigger version
]

# Train each, then ensemble
final_pred = weighted_average([pred1, pred2, pred3], weights=[0.4, 0.3, 0.3])
```

**Expected Impact:** +5-10% F1-Macro (proven technique in competitions)

---

### 4. **Advanced Training Techniques**

#### A. Mixup Augmentation (In-Training)
```python
def mixup_data(x, y, alpha=0.2):
    """Mixup: Beyond Empirical Risk Minimization"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

#### B. Curriculum Learning
```python
# Train on easy examples first, then hard ones
def curriculum_training():
    # Phase 1: Train on high-confidence samples
    easy_samples = get_high_confidence_samples(train_data)
    train(model, easy_samples, epochs=5)
    
    # Phase 2: Add harder samples
    all_samples = train_data
    train(model, all_samples, epochs=10)
```

**Expected Impact:** +2-4% F1-Macro

---

### 5. **Model Architecture Modifications**

#### Add Multi-Sample Dropout
```python
class MultiSampleDropout(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_probs=[0.1, 0.2, 0.3, 0.4, 0.5]):
        super().__init__()
        self.dropouts = nn.ModuleList([
            nn.Dropout(p) for p in dropout_probs
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in dropout_probs
        ])
    
    def forward(self, pooled_output):
        logits = [
            classifier(dropout(pooled_output))
            for dropout, classifier in zip(self.dropouts, self.classifiers)
        ]
        return torch.mean(torch.stack(logits), dim=0)
```

**Expected Impact:** +1-3% F1-Macro (better regularization)

---

### 6. **Post-Processing Techniques**

#### Threshold Optimization
```python
from sklearn.metrics import f1_score

# Find optimal decision thresholds for each class
def optimize_thresholds(y_true, y_pred_proba):
    best_thresholds = []
    for class_idx in range(3):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.3, 0.7, 0.01):
            preds = (y_pred_proba[:, class_idx] > threshold).astype(int)
            f1 = f1_score(y_true == class_idx, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        best_thresholds.append(best_threshold)
    
    return best_thresholds
```

**Expected Impact:** +1-2% F1-Macro

---

### 7. **Hyperparameter Enhancements**

#### Add These Parameters to Your Search Space
```python
# In HyperparameterConfig class:

# Stochastic Weight Averaging (SWA)
USE_SWA = True
SWA_START_EPOCH = 10

# Gradient Checkpointing (train larger models)
GRADIENT_CHECKPOINTING = True

# Layer-wise Learning Rate Decay (already in your code)
LAYERWISE_LR_DECAY_MIN = 0.85
LAYERWISE_LR_DECAY_MAX = 0.95

# R-Drop (regularization dropout)
R_DROP_ALPHA = 0.5  # KL divergence weight

# Label Smoothing (already in your code - good!)
LABEL_SMOOTHING = [0.05, 0.1, 0.15]
```

---

## 📊 Priority Implementation Order

### **Phase 1: Quick Wins (1-2 days)**
1. ✅ **Composite Metric** (Already implemented)
2. 🔄 **Run optimization with 50-100 trials**
3. 🔄 **Test different label smoothing values** (0.05, 0.1, 0.15)

**Expected Result:** 0.65-0.68 F1-Macro

---

### **Phase 2: High-Impact Additions (3-5 days)**
1. **Data Augmentation**
   - Back-translation for minority classes
   - Synonym replacement
   - Target: 2x minority class samples

2. **Ensemble of 3-5 Models**
   - Different seeds: [42, 123, 456, 789, 1011]
   - Soft voting on predictions

**Expected Result:** 0.70-0.74 F1-Macro ✅ **TARGET ACHIEVED**

---

### **Phase 3: Fine-Tuning (If needed for >0.74)**
1. Multi-Sample Dropout architecture
2. Threshold optimization
3. Curriculum learning
4. Try different architectures (RoBERTa-large, DeBERTa-v2-xlarge)

**Expected Result:** 0.74-0.78 F1-Macro

---

## 🎓 Why Each Technique Works

| Technique | Addresses | Impact |
|-----------|-----------|--------|
| **Composite Metric** | Balances performance & generalization | High |
| **Data Augmentation** | Class imbalance, limited data | Very High |
| **Ensemble** | Model variance, uncertainty | Very High |
| **Multi-Sample Dropout** | Overfitting | Medium |
| **Threshold Optimization** | Class-specific decision boundaries | Medium |
| **Label Smoothing** | Overconfident predictions | Medium |
| **Curriculum Learning** | Hard example learning | Medium |

---

## 📈 Monitoring & Debugging

### Check These Metrics
```python
# During training, track:
1. Per-class F1 scores (identify weak classes)
2. Confusion matrix (find misclassification patterns)
3. Loss curves (check for overfitting)
4. Learning rate schedule (ensure proper warmup)

# After optimization:
print("Per-Class Performance:")
for i, label in enumerate(['no_mistake', 'has_mistake', 'partial_mistake']):
    precision = precision_score(y_true, y_pred, labels=[i], average=None)[0]
    recall = recall_score(y_true, y_pred, labels=[i], average=None)[0]
    f1 = f1_score(y_true, y_pred, labels=[i], average=None)[0]
    print(f"{label:20s}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
```

### Red Flags to Watch
- **F1-Macro < 0.60:** Check class weights, increase regularization
- **High train F1, low val F1:** Overfitting - add dropout, more augmentation
- **One class F1 << others:** Focus augmentation on that class

---

## 🔥 Quick Start: Next Steps

1. **Run the notebook now** with composite metric (already implemented)
   ```bash
   # The notebook is ready - just run all cells
   ```

2. **While it's running, implement data augmentation** (parallel work)
   ```bash
   # Create: augmentation_pipeline.py
   # Add back-translation and synonym replacement
   ```

3. **After first results, train ensemble**
   ```python
   # Use top 3 hyperparameter sets from Optuna
   # Train with seeds: 42, 123, 456
   # Average predictions
   ```

---

## 💡 Pro Tips

1. **Don't overtune:** If validation F1 > 0.70 but test F1 < 0.70, you're overfitting to validation
2. **Use stratified splits:** Ensure validation set has same class distribution as train
3. **Monitor minority class:** If "has_mistake" F1 < 0.60, focus on that class specifically
4. **Increase trials:** 50 trials is good, 100 is better, 200 is optimal (if time permits)
5. **Save everything:** Save all models, predictions, and metrics for ensemble later

---

## 📞 Need Help?

If F1-Macro is still below 0.70 after Phase 2:
1. Share confusion matrix
2. Share per-class F1 scores
3. Share training curves
4. Check if data quality issues exist

---

## 🎯 Summary

**Primary Change Made:**
- ✅ Composite metric: 60% F1-Macro + 40% generalization penalty

**Recommended Next Steps:**
1. Run optimization (50-100 trials)
2. Implement data augmentation
3. Train ensemble of 3-5 models

**Expected Final Result:** F1-Macro = 0.70-0.74 ✅

Good luck! 🚀
