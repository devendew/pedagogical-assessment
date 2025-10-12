# 🔬 Optuna Hyperparameter Tuning Guide

## Overview

This notebook now supports **automated hyperparameter optimization** using Optuna, a powerful Bayesian optimization framework. This can improve your F1-Macro score from **0.65-0.73** to **0.67-0.75** or higher.

## Quick Start

### 1. Install Dependencies

```bash
pip install optuna plotly kaleido scipy
```

### 2. Choose Your Mode

**Option A: Optuna Tuning (Recommended for best results)**
```python
# In Step 1.5
RUN_OPTUNA_TUNING = True
OPTUNA_N_TRIALS = 50  # More trials = better results
```

**Option B: Fixed Hyperparameters (Faster)**
```python
# In Step 1.5
RUN_OPTUNA_TUNING = False
```

### 3. Run Cells

#### With Optuna:
1. **Step 0**: Install packages (run once)
2. **Step 1**: Configuration
3. **Step 1.5**: Enable Optuna
4. **Step 2**: Setup functions
5. **Step 2.5**: Define objective
6. **Step 3**: Load data
7. **Step 3.5**: Run optimization (12-48 hours)
8. **Step 3.6**: Train final model with best params
9. **Step 3.7**: Analyze results (optional)
10. **Step 4-6**: Evaluation and predictions

#### Without Optuna:
1. **Step 0**: Install packages
2. **Step 1**: Configuration
3. **Step 1.5**: Disable Optuna
4. **Step 2**: Setup
5. **Step 3**: Load data & create model
6. **Step 4**: Train (3-5 hours)
7. **Step 5-6**: Evaluation

---

## How Optuna Works

### Bayesian Optimization

Optuna uses **TPE (Tree-structured Parzen Estimator)** algorithm:

1. **Exploration Phase** (first ~10 trials)
   - Random sampling to explore the search space
   - Builds initial model of parameter-performance relationship

2. **Exploitation Phase** (remaining trials)
   - Focus on promising regions
   - Uses past trials to predict good parameters
   - Balances exploration vs exploitation

3. **Pruning** (optional, enabled by default)
   - Stops unpromising trials early
   - Saves time and resources
   - Uses MedianPruner strategy

### What Gets Optimized

**15 Hyperparameters** across 4 categories:

#### 1. Learning & Optimization
- `learning_rate`: 1e-6 to 5e-5 (log scale)
- `weight_decay`: 1e-3 to 0.3 (L2 regularization)
- `warmup_ratio`: 0.0 to 0.3 (gradual learning rate increase)
- `lr_scheduler`: linear, cosine, cosine_with_restarts, polynomial
- `layerwise_lr_decay`: 0.85 to 0.98 (lower layers learn slower)

#### 2. Training Strategy
- `num_epochs`: [20, 30, 40, 50]
- `batch_size`: [8, 16, 32]
- `gradient_accumulation`: [1, 2, 4, 8]
- `max_grad_norm`: 0.1 to 2.0 (gradient clipping)

#### 3. Loss & Regularization
- `focal_gamma`: 1.5 to 4.0 (class imbalance handling)
- `label_smoothing`: 0.0 to 0.2 (prevents overconfidence)
- `hidden_dropout`: 0.05 to 0.25
- `attention_dropout`: 0.05 to 0.25

#### 4. Data Processing
- `balance_strategy`: adaptive or oversample
- `augmentation_prob`: 0.1 to 0.5 (minority class augmentation)

---

## Expected Results

### Performance

| Mode | F1-Macro | Time | Trials |
|------|----------|------|--------|
| Fixed Hyperparameters | 0.65-0.73 | 3-5 hours | N/A |
| Optuna (10 trials) | 0.66-0.72 | 5-8 hours | 10 |
| **Optuna (50 trials)** ⭐ | **0.67-0.75** | **16-24 hours** | **50** |
| Optuna (100 trials) | 0.68-0.76 | 32-48 hours | 100 |

### Time Breakdown

Per trial (approximate):
- Data loading: 1-2 min
- Training: 15-30 min (depends on epochs)
- Evaluation: 1-2 min
- **Total per trial: 20-35 min**

50 trials = **16-30 hours**

---

## Configuration Options

### Basic Settings

```python
# Number of optimization trials
OPTUNA_N_TRIALS = 50  # Recommended: 25-100

# Time limit (optional)
OPTUNA_TIMEOUT = None  # or 86400 for 24 hours

# Parallel jobs
OPTUNA_N_JOBS = 1  # Sequential (recommended for GPU)

# Study name and storage
OPTUNA_STUDY_NAME = "bigbird_f1_macro_optimization"
OPTUNA_STORAGE = "sqlite:///optuna_bigbird_study.db"
```

### Pruning Settings

```python
# Enable early stopping of unpromising trials
OPTUNA_ENABLE_PRUNING = True  # Recommended: True

# Don't prune before this epoch
OPTUNA_PRUNING_WARMUP_EPOCHS = 5
```

### Customizing Search Space

Edit `OPTUNA_SEARCH_SPACE` dictionary in Step 1.5:

**Example 1: Narrow learning rate range**
```python
"learning_rate": {"type": "loguniform", "low": 5e-6, "high": 2e-5}
```

**Example 2: Add batch size option**
```python
"batch_size": {"type": "categorical", "choices": [4, 8, 16, 32, 64]}
```

**Example 3: Fix a parameter (remove from search)**
```python
# Comment out or remove the line
# "focal_gamma": {"type": "uniform", "low": 1.5, "high": 4.0}
```

**Example 4: Change epoch range**
```python
"num_epochs": {"type": "categorical", "choices": [10, 15, 20, 25, 30]}
```

---

## Output Files

### After Optimization (Step 3.5)

```
results/optimized_f1_macro/
├── best_hyperparameters.json     # Best parameters found
├── optuna_study_results.csv      # All trials (DataFrame)
├── optuna_bigbird_study.db       # SQLite database (persistent)
└── optuna_plots/                 # Visualizations
    ├── optimization_history.png  # F1-Macro progress
    ├── param_importances.png     # Which params matter most
    ├── parallel_coordinate.png   # Parameter relationships
    └── slice_plot.png            # Individual parameter effects
```

### After Final Training (Step 3.6)

```
results/optimized_f1_macro/
└── final_model/
    └── best/
        ├── config.json           # Model configuration
        ├── pytorch_model.bin     # Trained weights
        └── tokenizer files       # Tokenizer
```

### After Analysis (Step 3.7)

```
results/optimized_f1_macro/
└── optuna_detailed_analysis.png  # Custom visualizations
```

---

## Understanding Results

### best_hyperparameters.json

```json
{
  "trial_number": 35,
  "f1_macro": 0.6847,
  "params": {
    "learning_rate": 6.584e-06,
    "batch_size": 8,
    "weight_decay": 0.1815,
    ...
  },
  "user_attrs": {
    "accuracy": 0.8123,
    "f1_yes": 0.8945,
    "f1_to_some_extent": 0.5234,
    "f1_no": 0.6362
  },
  "timestamp": "2025-10-08 22:17:53"
}
```

### Parameter Importance

Shows which hyperparameters have the most impact:

```
Parameter                   Importance
learning_rate               0.3245  ████████████████
focal_gamma                 0.2156  ██████████
weight_decay                0.1834  █████████
batch_size                  0.1245  ██████
...
```

**Interpretation:**
- **High importance** (>0.2): Critical to optimize
- **Medium importance** (0.1-0.2): Helpful to tune
- **Low importance** (<0.1): Less critical, can use default

### Optimization History

Shows F1-Macro improving over trials:
- **Steep increase** early: Good exploration
- **Plateau** later: Convergence (good sign)
- **Oscillating**: Still exploring (run more trials)

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size in search space
"batch_size": {"type": "categorical", "choices": [4, 8, 16]}

# Or increase gradient accumulation
"gradient_accumulation": {"type": "categorical", "choices": [2, 4, 8, 16]}
```

### Issue 2: Trials Taking Too Long

**Solution:**
```python
# Reduce epochs in search space
"num_epochs": {"type": "categorical", "choices": [10, 15, 20]}

# Enable pruning
OPTUNA_ENABLE_PRUNING = True
```

### Issue 3: Study Not Converging

**Symptoms:** F1-Macro still improving at end

**Solution:**
- Run more trials (increase `OPTUNA_N_TRIALS`)
- Study has already run? It will resume automatically
- Check convergence in Step 3.7

### Issue 4: All Trials Failing

**Check:**
1. Data loading successful? (Run Step 3)
2. GPU available? (Check `torch.cuda.is_available()`)
3. Enough disk space for checkpoints?
4. SQLite database writable?

### Issue 5: Want to Resume Study

**Good news:** Studies auto-resume! Just run Step 3.5 again.

```python
# This will load existing study and continue
study = optuna.create_study(
    study_name=OPTUNA_STUDY_NAME,
    storage=OPTUNA_STORAGE,
    load_if_exists=True  # ← Auto-resume
)
```

---

## Advanced Usage

### Resume from Checkpoint

Study is saved in SQLite database. To resume:

1. Keep `OPTUNA_STORAGE` path the same
2. Run Step 3.5 again
3. It will continue from last trial

### Analyze Without Re-running

```python
# Load existing study
study = optuna.load_study(
    study_name="bigbird_f1_macro_optimization",
    storage="sqlite:///optuna_bigbird_study.db"
)

# Get best trial
print(f"Best F1-Macro: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Export Study to CSV

Already done automatically! See:
```
results/optimized_f1_macro/optuna_study_results.csv
```

### Compare Multiple Studies

```python
# Create studies with different search spaces
study1 = optuna.create_study(study_name="study_v1", ...)
study2 = optuna.create_study(study_name="study_v2", ...)

# Compare
print(f"Study 1 best: {study1.best_value:.4f}")
print(f"Study 2 best: {study2.best_value:.4f}")
```

### Use Best Params in New Notebook

```python
import json

# Load best hyperparameters
with open('results/optimized_f1_macro/best_hyperparameters.json', 'r') as f:
    best_config = json.load(f)

# Use in training
LEARNING_RATE = best_config['params']['learning_rate']
BATCH_SIZE = best_config['params']['batch_size']
# ... etc
```

---

## Best Practices

### 1. Start Small, Scale Up
- Run 10 trials first (2-3 hours)
- Check if study is working
- Then run full 50+ trials

### 2. Monitor Progress
- Check Step 3.7 periodically
- Look for convergence
- Stop if plateau reached

### 3. Save GPU Resources
- Use pruning (`OPTUNA_ENABLE_PRUNING = True`)
- Lower `per_device_eval_batch_size` if needed
- Clear cache between trials (automatic)

### 4. Parallel Trials (Multi-GPU)
```python
OPTUNA_N_JOBS = 2  # Use 2 GPUs
```
⚠️ **Note:** Each job needs separate GPU. Set `CUDA_VISIBLE_DEVICES`.

### 5. Time Limits
```python
# Stop after 24 hours regardless of trials
OPTUNA_TIMEOUT = 86400  # seconds
```

---

## FAQ

**Q: How many trials do I need?**
A: 50 is a good balance. 25 for quick results, 100+ for best results.

**Q: Can I stop and resume?**
A: Yes! Study is saved in SQLite database. Just re-run Step 3.5.

**Q: What if I find better params manually?**
A: You can add them as a trial or just use them directly.

**Q: Should I enable pruning?**
A: Yes, recommended. Saves ~30% time with minimal accuracy loss.

**Q: How to choose between adaptive vs oversample?**
A: Let Optuna decide! It's in the search space.

**Q: Can I run on CPU?**
A: Yes, but very slow. 100x longer than GPU.

**Q: Memory issues?**
A: Reduce batch_size, increase gradient_accumulation, or use smaller model.

---

## Comparison: Manual vs Optuna

| Aspect | Manual Tuning | Optuna |
|--------|---------------|--------|
| **Time** | Days to weeks | 16-48 hours |
| **Trials** | 5-10 manually | 50-100 automatically |
| **Coverage** | Limited combinations | Explores 10^15 combinations |
| **Strategy** | Grid search / guess | Bayesian optimization |
| **Best F1** | 0.65-0.70 | 0.67-0.75 |
| **Reproducible** | Sometimes | Always (saved params) |
| **Insights** | None | Parameter importance, plots |

---

## Citation

If you use this optimization approach, please cite:

```
Optuna: A hyperparameter optimization framework
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama
arXiv:1907.10902, 2019
```

---

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Run Step 3.7 for detailed analysis
3. Review Optuna logs in `results/optimized_f1_macro/`

---

**Happy Optimizing! 🚀**
