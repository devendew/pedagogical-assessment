# Hyperparameter Tuning Guide for Mistake Identification Model

## 🎯 Overview

This notebook implements **state-of-the-art hyperparameter optimization** using Optuna, featuring:

- **Bayesian Optimization** with TPE (Tree-structured Parzen Estimator)
- **Automatic Pruning** of unpromising trials using Hyperband
- **Multi-Objective Optimization** support (optional)
- **Comprehensive Visualization** and analysis
- **Production-Ready** best model training

## 📊 What Gets Tuned

### Core Hyperparameters

1. **Learning Rate** (1e-6 to 5e-5, log-scale)
   - Most critical hyperparameter
   - Log-scale search for efficiency
   - Typical optimal range: 1e-5 to 3e-5 for DeBERTa

2. **Batch Size** (4, 8, 16, 32)
   - Affects gradient stability and training speed
   - Larger batches = more stable but less regularization
   - GPU memory constraints apply

3. **Weight Decay** (0.0 to 0.3)
   - L2 regularization strength
   - Prevents overfitting
   - Typical optimal: 0.01-0.1

4. **Warmup Ratio** (0.0 to 0.2)
   - Gradual learning rate increase at start
   - Stabilizes early training
   - Usually 0.05-0.1 works well

5. **Number of Epochs** (3 to 15)
   - Training duration
   - Early stopping prevents overfitting
   - Pruning stops bad trials early

### Optimizer Settings

6. **LR Scheduler Type**
   - `linear`: Linear decay (most common)
   - `cosine`: Cosine annealing (smooth decay)
   - `cosine_with_restarts`: Periodic restarts
   - `polynomial`: Polynomial decay
   - `constant`: No decay

7. **Optimizer**
   - `adamw_torch`: PyTorch AdamW (default, recommended)
   - `adamw_hf`: HuggingFace AdamW
   - `adafactor`: Memory-efficient for large models

8. **Gradient Accumulation Steps** (1, 2, 4)
   - Simulates larger batch sizes
   - Useful when GPU memory is limited
   - Effective batch size = batch_size × grad_accum_steps

9. **Max Gradient Norm** (0.1 to 5.0)
   - Gradient clipping threshold
   - Prevents exploding gradients
   - Usually 1.0 works well

### Model-Specific

10. **Hidden Dropout** (0.1 to 0.3)
    - Dropout rate in hidden layers
    - Regularization technique
    - DeBERTa default: 0.1

11. **Attention Dropout** (0.1 to 0.3)
    - Dropout rate in attention layers
    - Prevents attention overfitting
    - Usually same as hidden dropout

12. **Layer-wise Learning Rate Decay** (0.8 to 1.0)
    - Different LR for different layers
    - Lower layers learn slower (closer to 0.8)
    - Higher layers learn faster (1.0)

## 🚀 How to Use

### Quick Start

```python
# 1. Set number of trials
N_TRIALS = 30  # More trials = better results but longer time

# 2. Run optimization
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# 3. Get best hyperparameters
best_params = study.best_trial.params

# 4. Train final model with best params
# (automatically done in the notebook)
```

### Configuration Options

#### Computational Budget

```python
# Number of trials to run
N_TRIALS = 30  # Default: 30 (adjust based on time/compute)

# Maximum time (in seconds)
TIMEOUT = None  # None = no limit, or set to e.g., 3600 for 1 hour

# Parallel jobs
N_JOBS = 1  # Keep at 1 for GPU to avoid OOM errors
```

#### Sampling Strategy

```python
# TPE Sampler (Recommended)
sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=10,  # Random trials before Bayesian optimization
    multivariate=True,    # Consider parameter interactions
)

# Alternative: Random Sampler (faster but less efficient)
# sampler = optuna.samplers.RandomSampler(seed=42)
```

#### Pruning Strategy

```python
# Hyperband Pruner (Recommended)
pruner = optuna.pruners.HyperbandPruner(
    min_resource=1,        # Minimum epochs before pruning
    max_resource=15,       # Maximum epochs
    reduction_factor=3,    # How aggressively to prune
)

# Alternative: Median Pruner
# pruner = optuna.pruners.MedianPruner(
#     n_startup_trials=5,
#     n_warmup_steps=0,
# )
```

### Multi-Objective Optimization

For optimizing multiple metrics simultaneously:

```python
# Enable multi-objective optimization
RUN_MULTI_OBJECTIVE = True

# This will optimize BOTH:
# - F1-Macro score (primary)
# - Accuracy (secondary)
# 
# Result: Pareto-optimal solutions balancing both metrics
```

## 📈 Understanding Results

### Visualization Dashboard

The notebook generates comprehensive visualizations:

1. **Optimization History**
   - Shows F1-score progression over trials
   - "Best So Far" line shows improvement

2. **Parameter Importance**
   - Ranks hyperparameters by impact
   - Focus tuning on top parameters

3. **Learning Rate vs Performance**
   - Scatter plot showing LR sensitivity
   - Log-scale x-axis

4. **Batch Size Impact**
   - Bar chart with mean ± std
   - Shows optimal batch size

5. **Weight Decay Impact**
   - Scatter plot
   - Shows regularization sweet spot

6. **Parallel Coordinates (Top 10)**
   - Shows parameter combinations of best trials
   - Helps identify patterns

### Trial Comparison Table

Top 10 trials ranked by F1-Macro:

```
Rank | Trial # | F1-Macro | Learning Rate | Batch Size | ...
-----|---------|----------|---------------|------------|----
  1  |   23    |  0.8542  |   2.34e-05   |     16     | ...
  2  |   17    |  0.8501  |   1.87e-05   |      8     | ...
...
```

### Output Files

| File | Description |
|------|-------------|
| `best_hyperparameters.json` | Best configuration found |
| `top_trials_comparison.csv` | Detailed comparison of top 10 trials |
| `hyperparameter_optimization_analysis.png` | Visualization dashboard |
| `final_model_evaluation.json` | Final model metrics |
| `best_model/mistake-identification-OPTIMIZED/` | Trained model with best hyperparameters |

## 🎓 Best Practices

### For Limited Compute (< 4 hours)

1. **Reduce trials**: `N_TRIALS = 10-15`
2. **Reduce epoch range**: `EPOCHS_MAX = 8`
3. **Smaller search space**: Focus on LR, batch size, weight decay
4. **Aggressive pruning**: Use Hyperband with `reduction_factor=3`

```python
# Minimal config for quick results
HyperparameterConfig.EPOCHS_MIN = 3
HyperparameterConfig.EPOCHS_MAX = 8
N_TRIALS = 10
```

### For Maximum Performance (> 12 hours)

1. **More trials**: `N_TRIALS = 50-100`
2. **Wider search space**: Include all hyperparameters
3. **Multi-objective**: Enable for balanced optimization
4. **Multiple seeds**: Run with different random seeds

```python
# Comprehensive config for best results
N_TRIALS = 50
RUN_MULTI_OBJECTIVE = True
```

### For Production Deployment

1. **Validate top 3-5 configurations**
   - Test on held-out test set
   - Check consistency across runs

2. **Ensemble models**
   - Combine predictions from top trials
   - Usually improves F1 by 1-3%

3. **Document hyperparameters**
   - Save `best_hyperparameters.json`
   - Include in model card

4. **Monitor for data drift**
   - Retune periodically on new data
   - Quick tuning: 10 trials around best config

## 🔬 Advanced Techniques

### 1. Warm Starting

Continue optimization from previous study:

```python
# Load previous study
study = optuna.load_study(
    study_name='deberta-mistake-identification',
    storage='sqlite:///optuna_study.db'
)

# Continue optimization
study.optimize(objective, n_trials=20)
```

### 2. Conditional Hyperparameters

Some hyperparameters depend on others:

```python
# Example: Only tune layerwise_lr_decay if using it
use_layerwise = trial.suggest_categorical('use_layerwise', [True, False])
if use_layerwise:
    layerwise_lr_decay = trial.suggest_float('layerwise_lr_decay', 0.8, 0.95)
else:
    layerwise_lr_decay = 1.0
```

### 3. Custom Pruning Logic

```python
# Prune if validation loss increases
if epoch > 3 and current_loss > previous_loss * 1.1:
    raise optuna.TrialPruned()
```

### 4. Learning Rate Range Test

Find optimal LR range before full tuning:

```python
# Quick LR range test (5 trials)
for lr in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]:
    # Train for 2 epochs
    # Check F1 score
    # Narrow range around best
```

## 📝 Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
1. Reduce batch size: `BATCH_SIZE_OPTIONS = [4, 8]`
2. Increase gradient accumulation: `GRAD_ACCUM_OPTIONS = [2, 4, 8]`
3. Enable FP16: Already enabled in code
4. Clear cache between trials: Already done with `torch.cuda.empty_cache()`

### Issue: All trials pruned

**Solution:**
1. Increase `min_resource` in pruner: `min_resource=2`
2. Reduce pruning aggressiveness: `reduction_factor=2`
3. Check if model is training (loss decreasing)

### Issue: No improvement over trials

**Solution:**
1. Widen search space (maybe optimal is outside current range)
2. Check if baseline model is already optimal
3. Try different sampler (Random → TPE or vice versa)
4. Increase `n_startup_trials` for more exploration

### Issue: Training too slow

**Solution:**
1. Reduce `EPOCHS_MAX`: Set to 8 or 10
2. Use smaller validation set during tuning
3. Increase `logging_steps` to reduce overhead
4. Disable extensive logging during trials

## 🎯 Expected Results

### Typical Improvements

- **Baseline** (default hyperparameters): F1-Macro ~0.75-0.80
- **After 10 trials**: F1-Macro ~0.80-0.83 (+3-5%)
- **After 30 trials**: F1-Macro ~0.82-0.86 (+5-8%)
- **After 50+ trials**: F1-Macro ~0.84-0.88 (+7-10%)

### Convergence

- First 5-10 trials: Exploration (random or quasi-random)
- Trials 10-20: Exploitation (focusing on promising regions)
- Trials 20+: Fine-tuning (small improvements)

Most gains come from first 20-30 trials. Beyond that, improvements are marginal.

## 📚 References

1. **Optuna**: A hyperparameter optimization framework
   - Paper: [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)
   - Docs: https://optuna.readthedocs.io/

2. **TPE**: Tree-structured Parzen Estimator
   - Paper: [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)

3. **Hyperband**: Successive Halving with restarts
   - Paper: [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560)

4. **DeBERTa**: Decoding-enhanced BERT with Disentangled Attention
   - Paper: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

---

**Created by**: Hyperparameter Tuning Pipeline
**Last Updated**: 2025
**Notebook**: `mistake_microsoft_deberta_v3_large_hyperparameter_tuning.ipynb`
