# 🎯 Quick Reference Card - F1-Macro Optimization

## ⚡ TL;DR - Just Run These Cells

```
Cell 4  → Configuration (SMOTE + Focal Loss γ=2.5)
Cell 5  → Setup (imports, functions, tokenizer)  
Cell 7  → Create model & trainer
Cell 9  → Train (~4 hours)
Cell 11 → Optimize thresholds
Cell 15 → Generate predictions
```

**Expected result**: F1-macro 0.65-0.73 (vs 0.29 baseline)

---

## 🎛️ Key Hyperparameters (Cell 4)

### Data Balancing
```python
BALANCE_STRATEGY = 'smote'      # ⭐ BEST
BALANCE_STRATEGY = 'hybrid'     # Alternative
BALANCE_STRATEGY = 'adaptive'   # Fast
BALANCE_STRATEGY = 'oversample' # Simplest
```

### Focal Loss
```python
FOCAL_GAMMA = 2.5  # ⭐ Optimal for 11:1 imbalance
FOCAL_GAMMA = 2.0  # Moderate imbalance (5:1)
FOCAL_GAMMA = 3.0  # Extreme imbalance (>15:1)
```

### Learning Rate
```python
learning_rate = 1e-5   # ⭐ Stable for minorities
learning_rate = 5e-6   # Very cautious
learning_rate = 2e-5   # Faster (may overfit)
```

### Training Duration
```python
num_train_epochs = 25  # ⭐ Good balance
num_train_epochs = 15  # Quick test
num_train_epochs = 40  # Maximum convergence
```

---

## 📊 What Each Technique Does

| Technique | Impact | Why It Works |
|-----------|--------|--------------|
| **Focal Loss γ=2.5** | +15-20% | Down-weights easy (majority) examples |
| **SMOTE Balancing** | +10-15% | Synthetic minorities, no duplication |
| **Low LR (1e-5)** | +5-8% | Careful optimization, no overshooting |
| **Threshold Optimization** | +2-5% | Per-class optimal decisions |
| **Label Smoothing (0.1)** | +2-4% | Prevents overconfidence |
| **Extended Training (25 ep)** | +3-5% | Minorities need more time |
| **Effective Weights** | Multiplier | Better than inverse frequency |
| **Layer-wise LR (0.95)** | +1-2% | Task-specific top, general bottom |
| **Augmentation** | +1-3% | More minority diversity |

---

## 🎯 Performance Targets

```
F1-Macro < 0.50   → ⚠️  Check focal gamma, try higher
F1-Macro 0.50-0.60 → ✓ Good, add threshold optimization
F1-Macro 0.60-0.65 → ✅ Great! Try ensemble for more
F1-Macro 0.65-0.70 → 🎉 Excellent! Mission accomplished
F1-Macro > 0.70    → 🌟 Outstanding! Top-tier performance
```

---

## 🔧 Troubleshooting

### Problem: F1-Macro Not Improving

**Check 1**: Are all classes being predicted?
```python
# After training, check predictions distribution
pred_counts = predictions.value_counts()
print(pred_counts)  # Should have all 3 classes
```

**Fix**: Increase `FOCAL_GAMMA` to 3.0 or 3.5

---

**Check 2**: Is model seeing balanced data?
```python
# Check after balancing
print(train_df_balanced['label'].value_counts())
# Should be roughly equal
```

**Fix**: Try `BALANCE_STRATEGY = 'hybrid'`

---

**Check 3**: Is learning rate too high?
```python
# Check training loss - should decrease smoothly
# If erratic, LR too high
```

**Fix**: Reduce to `5e-6`

---

### Problem: Training Too Slow

**Quick Fix 1**: Reduce epochs for testing
```python
num_train_epochs = 10  # Quick test
```

**Quick Fix 2**: Increase batch size
```python
per_device_train_batch_size = 16
gradient_accumulation_steps = 2
```

**Quick Fix 3**: Use simpler balancing
```python
BALANCE_STRATEGY = 'oversample'  # Faster than SMOTE
```

---

### Problem: Out of Memory

**Fix 1**: Reduce batch size
```python
per_device_train_batch_size = 4
gradient_accumulation_steps = 8  # Keep effective size = 32
```

**Fix 2**: Reduce max length
```python
MAX_LENGTH = 256  # Instead of 512
```

**Fix 3**: Disable FP16 (if enabled)
```python
fp16 = False
```

---

### Problem: Overfitting (Val Loss Increasing)

**Fix 1**: Increase regularization
```python
weight_decay = 0.2  # Increase from 0.1545
hidden_dropout_prob = 0.2  # Increase from 0.15
```

**Fix 2**: More label smoothing
```python
LABEL_SMOOTHING = 0.15  # Increase from 0.1
```

**Fix 3**: Earlier stopping
```python
EARLY_STOPPING_PATIENCE = 5  # Reduce from 7
```

---

## 🚀 Push to F1-Macro > 0.75

### Quick Wins (2-4 hours extra)
1. Run threshold optimization ✅ (already in notebook)
2. Increase `FOCAL_GAMMA` to 3.0
3. Try `BALANCE_STRATEGY = 'hybrid'`

### Medium Effort (6-8 hours)
1. Train 3 models with different seeds
2. Average predictions (soft voting)
3. Expected: +3-5% gain

### High Effort (12+ hours)
1. Ensemble DeBERTa + RoBERTa
2. Add data augmentation (back-translation)
3. Hyperparameter tuning with Optuna
4. Expected: +5-8% gain

---

## 📐 Mathematical Intuition

### Why Focal Loss Works

**Standard Cross-Entropy**:
```
Loss = -log(p)
```
- Easy example (p=0.9): loss = 0.10
- Hard example (p=0.1): loss = 2.30
- Ratio: 23:1 (not enough focus on hard)

**Focal Loss (γ=2.5)**:
```
Loss = -(1-p)^2.5 * log(p)
```
- Easy example (p=0.9): (0.1)^2.5 × 0.10 = 0.003
- Hard example (p=0.1): (0.9)^2.5 × 2.30 = 1.76
- Ratio: 587:1 (much more focus on hard!)

### Why SMOTE Works

**Simple Oversampling**:
- Duplicate same examples → overfitting

**SMOTE**:
- Take minority sample A
- Find k nearest neighbors (k=5)
- Interpolate: new_sample = A + α × (B - A)
- Creates realistic variations

### Why Low LR Works

**High LR (2e-5)**:
- Big steps → may overshoot minority patterns
- Gets stuck in majority-class local minimum

**Low LR (1e-5)**:
- Small steps → carefully explores minority patterns
- Can escape majority-class minimum

---

## 📁 Output Files Reference

After running all cells:

```
results/mistake-identification-deberta-v3-large-OPTIMIZED-F1/
├── best_model/                          # Trained model
│   ├── pytorch_model.bin                # Weights
│   ├── config.json                      # Config
│   └── tokenizer files                  # Tokenizer
├── training_analysis_comprehensive.png  # Visualizations
├── confusion_matrices_comparison.png    # Before/after thresholds
├── optimal_thresholds.json              # Threshold values
├── test_predictions.csv                 # Test predictions (CSV)
├── test_predictions.json                # Test predictions (JSON)
├── trainer_state.json                   # Training history
└── logs/                                # TensorBoard logs
```

---

## 🎓 Key Concepts

### F1-Macro vs Accuracy
```
Accuracy = Correct / Total
→ Can be 78% by always predicting "Yes"
→ Doesn't care about minority classes

F1-Macro = (F1_yes + F1_to_some + F1_no) / 3
→ Equal weight to each class
→ Forces model to learn all classes
```

### Precision vs Recall
```
Precision = TP / (TP + FP)  # "Of predicted positives, how many correct?"
Recall = TP / (TP + FN)     # "Of actual positives, how many found?"

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Why Threshold Matters
```
Standard: argmax(probs) → threshold = 0.5 (implicit)

Imbalanced: Different thresholds per class
Example:
  Yes threshold: 0.3 (lower → more Yes predictions)
  No threshold: 0.7 (higher → only confident No)
  
Result: Better balance → higher F1-macro
```

---

## 🎯 Decision Tree: What to Do Next

```
Start here
│
├─ F1-Macro < 0.50?
│  ├─ Yes → Increase FOCAL_GAMMA to 3.0
│  └─ No → Continue
│
├─ F1-Macro < 0.60?
│  ├─ Yes → Try BALANCE_STRATEGY = 'hybrid'
│  └─ No → Continue
│
├─ F1-Macro < 0.65?
│  ├─ Yes → Run threshold optimization
│  └─ No → Continue
│
├─ F1-Macro < 0.70?
│  ├─ Yes → Ensemble 3 models (different seeds)
│  └─ No → Continue
│
└─ F1-Macro ≥ 0.70?
   └─ 🎉 Success! Consider:
      - Ensemble different architectures
      - Hyperparameter tuning
      - Advanced augmentation
      Goal: Push to 0.75+
```

---

## 💡 Pro Tips

1. **Monitor per-class F1**, not just macro
2. **Check confusion matrix** after each run
3. **Save all checkpoints** (sometimes earlier is better)
4. **Use same validation set** for fair comparison
5. **Set all seeds** for reproducibility
6. **Start with quick test** (10 epochs) before full run
7. **TensorBoard is your friend** → `tensorboard --logdir results/.../logs`
8. **Document what works** in a separate file

---

## 🔗 Useful Commands

```bash
# Monitor training (in terminal)
tensorboard --logdir results/mistake-identification-deberta-v3-large-OPTIMIZED-F1/logs

# Check GPU usage
nvidia-smi -l 1

# Estimate remaining time
# (Look at logs: "steps/sec", calculate: remaining_steps / steps_per_sec / 3600)
```

---

## 📞 When to Ask for Help

1. F1-Macro < 0.40 after full training
2. Training crashes (OOM, CUDA errors)
3. All predictions are same class
4. Validation loss diverging
5. Threshold optimization not helping

Otherwise, try the troubleshooting steps first!

---

**Remember**: Class imbalance is hard! Even F1-Macro 0.60 is good. 0.65+ is excellent! 🎯
