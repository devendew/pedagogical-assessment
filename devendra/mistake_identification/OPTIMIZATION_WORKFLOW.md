# Hyperparameter Optimization Workflow

## 🔄 Complete Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETER OPTIMIZATION                       │
│                     WITH OPTUNA + BAYESIAN SEARCH                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: SETUP & CONFIGURATION                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📋 Define Search Space                                              │
│  ├─ Learning Rate: [1e-6, 5e-5] (log scale)                        │
│  ├─ Batch Size: [4, 8, 16, 32]                                     │
│  ├─ Weight Decay: [0.0, 0.3]                                       │
│  ├─ Warmup Ratio: [0.0, 0.2]                                       │
│  ├─ Epochs: [3, 15]                                                │
│  └─ ... + 6 more parameters                                        │
│                                                                      │
│  ⚙️  Configure Sampler & Pruner                                      │
│  ├─ TPE Sampler (Bayesian optimization)                            │
│  └─ Hyperband Pruner (early stopping)                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: OPTIMIZATION LOOP (N_TRIALS iterations)                     │
└─────────────────────────────────────────────────────────────────────┘

  Trial 1 (Random)
  ┌──────────────────────────────────────────────────┐
  │ 1. Suggest hyperparameters                       │
  │    └─ First 10 trials: Random exploration        │
  │                                                   │
  │ 2. Initialize model with suggested params        │
  │    └─ Load DeBERTa-v3-large                      │
  │                                                   │
  │ 3. Train model                                   │
  │    ├─ Forward pass                               │
  │    ├─ Compute weighted loss                      │
  │    ├─ Backward pass                              │
  │    └─ Update weights                             │
  │                                                   │
  │ 4. Evaluate on validation set (each epoch)       │
  │    └─ Compute F1-Macro                           │
  │                                                   │
  │ 5. Report intermediate value                     │
  │    └─ Pruner decides: continue or stop?          │
  │                                                   │
  │ 6. Return final F1-Macro score                   │
  └──────────────────────────────────────────────────┘
                        ↓
  Trial 2 (Random)
  ┌──────────────────────────────────────────────────┐
  │ ... same process ...                             │
  │ F1-Macro: 0.7823                                 │
  └──────────────────────────────────────────────────┘
                        ↓
  ...
                        ↓
  Trial 10 (Random - Last startup trial)
  ┌──────────────────────────────────────────────────┐
  │ ... same process ...                             │
  │ F1-Macro: 0.8156                                 │
  └──────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ TPE SAMPLER KICKS IN          │
        │ (Bayesian Optimization)       │
        └───────────────────────────────┘
                        ↓
  Trial 11 (TPE-guided)
  ┌──────────────────────────────────────────────────┐
  │ 1. Analyze previous 10 trials                    │
  │    └─ Build probabilistic model                  │
  │                                                   │
  │ 2. Suggest promising hyperparameters             │
  │    └─ Focus on regions with high F1              │
  │                                                   │
  │ 3. Train & evaluate                              │
  │    └─ F1-Macro: 0.8234 ✓ Improvement!           │
  └──────────────────────────────────────────────────┘
                        ↓
  Trial 12 (TPE-guided)
  ┌──────────────────────────────────────────────────┐
  │ ... TPE suggests near Trial 11 ...              │
  │ F1-Macro: 0.8289 ✓ Better!                      │
  └──────────────────────────────────────────────────┘
                        ↓
  Trial 13 (TPE-guided)
  ┌──────────────────────────────────────────────────┐
  │ ... exploring different region ...               │
  │ Epoch 2/10: F1 = 0.72 (very low!)               │
  │     ↓                                             │
  │ ⚠️  HYPERBAND PRUNER STOPS TRIAL                 │
  │     (Not worth continuing)                        │
  └──────────────────────────────────────────────────┘
                        ↓
  ... Continue for N_TRIALS ...
                        ↓
  Trial 30 (Final)
  ┌──────────────────────────────────────────────────┐
  │ F1-Macro: 0.8456 ✓ Best so far!                 │
  └──────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: ANALYSIS & SELECTION                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📊 Analyze Results                                                  │
│  ├─ Best trial: #23 with F1 = 0.8542                               │
│  ├─ Parameter importance ranking                                    │
│  ├─ Optimization history visualization                              │
│  └─ Top 10 trials comparison                                        │
│                                                                      │
│  ✅ Select Best Hyperparameters                                      │
│  └─ Extract from best trial                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: FINAL MODEL TRAINING                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🎯 Train with Best Hyperparameters                                 │
│  ├─ Use full training data                                          │
│  ├─ Apply optimal configuration                                     │
│  ├─ Early stopping with patience                                    │
│  └─ Save best checkpoint                                            │
│                                                                      │
│  📈 Evaluate Final Model                                             │
│  ├─ Validation F1-Macro: 0.8542                                    │
│  ├─ Validation Accuracy: 0.8234                                    │
│  └─ Per-class metrics                                               │
│                                                                      │
│  💾 Save Everything                                                  │
│  ├─ Model weights                                                   │
│  ├─ Tokenizer                                                       │
│  ├─ Best hyperparameters (JSON)                                    │
│  ├─ Evaluation metrics (JSON)                                      │
│  └─ Training history                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: DEPLOYMENT & INFERENCE                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🚀 Load Optimized Model                                             │
│  └─ From: best_model/mistake-identification-OPTIMIZED/             │
│                                                                      │
│  🔮 Make Predictions                                                 │
│  ├─ On dev_testset.json                                            │
│  └─ On testset.json                                                │
│                                                                      │
│  📤 Export Results                                                   │
│  ├─ dev_predictions.csv                                            │
│  └─ test_predictions.csv                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🧠 How Bayesian Optimization Works

```
┌───────────────────────────────────────────────────────────────┐
│        TPE (Tree-structured Parzen Estimator) Process         │
└───────────────────────────────────────────────────────────────┘

After initial random trials (n=10):

1. Build Two Probabilistic Models
   ┌──────────────────────────────────────────────┐
   │ Good Trials (top 20%)                        │
   │ ├─ LR ~ LogNormal(μ=-10.5, σ=0.3)          │
   │ ├─ Batch Size ~ {8:0.4, 16:0.6}            │
   │ └─ Weight Decay ~ Normal(μ=0.08, σ=0.02)   │
   └──────────────────────────────────────────────┘
              ↕️
   ┌──────────────────────────────────────────────┐
   │ Bad Trials (bottom 80%)                      │
   │ ├─ LR ~ LogNormal(μ=-11.2, σ=0.8)          │
   │ ├─ Batch Size ~ {4:0.3, 8:0.3, 16:0.4}     │
   │ └─ Weight Decay ~ Uniform(0.0, 0.3)         │
   └──────────────────────────────────────────────┘

2. Compute Expected Improvement (EI)
   
   For each candidate hyperparameter set:
   
   EI = P(good) / P(bad)
   
   Higher EI = More likely to give good results

3. Select Hyperparameters with Highest EI
   
   Next trial uses: LR=2.3e-5, BS=16, WD=0.09
   (High probability under "good" distribution)

4. Update Models with New Results
   
   After trial completes:
   ├─ Add to good/bad group based on result
   └─ Re-fit probability distributions

5. Repeat
   
   Continuously refine understanding of
   which hyperparameters work best
```

## ⚡ How Hyperband Pruning Works

```
┌───────────────────────────────────────────────────────────────┐
│                    Hyperband Pruning Logic                     │
└───────────────────────────────────────────────────────────────┘

Trial starts training:

Epoch 1: F1 = 0.65
  ├─ Too early to judge
  └─ ✓ Continue

Epoch 2: F1 = 0.68
  ├─ Compare to other trials at epoch 2
  │  └─ Median F1 at epoch 2: 0.75
  ├─ This trial: 0.68 < 0.75 (below median)
  └─ ⚠️  Candidate for pruning

Epoch 3: F1 = 0.69
  ├─ Still below median (0.76)
  ├─ Not improving fast enough
  └─ ❌ PRUNE TRIAL (stop training)
     Save ~7 epochs of computation!

Good trial example:

Epoch 1: F1 = 0.72
Epoch 2: F1 = 0.78  ← Above median!
  └─ ✓ Promising, continue
Epoch 3: F1 = 0.81  ← Still above median
  └─ ✓ Definitely promising
...
Epoch 10: F1 = 0.85 ← Finish training
  └─ ✓ New best trial!
```

## 📊 Optimization Convergence

```
F1-Macro Score
    0.85 │                                    ●  ← Best: 0.8542
         │                              ●   ●
    0.84 │                        ●   ●
         │                      ●
    0.83 │          ●     ●   ●
         │        ●   ●   
    0.82 │    ●
         │  ●
    0.81 │●
         │
    0.80 │
         │
    0.79 │    ●
         │●
    0.78 │  ●       ●
         │    ●
    0.77 │      ●
         │
    0.76 │        ✗ ✗  ← Pruned trials
         │    ✗
    0.75 │  ✗
         └────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴──→ Trial #
              5    10    15    20    25    30

Phase 1 (Trials 1-10): Random Exploration
  └─ Discover general landscape

Phase 2 (Trials 11-20): Focused Exploitation  
  └─ TPE guides search to promising regions

Phase 3 (Trials 21-30): Fine-tuning
  └─ Small improvements around best configurations
```

## 🎯 Multi-Objective Optimization

```
┌───────────────────────────────────────────────────────────────┐
│          Multi-Objective: F1-Macro vs Accuracy                 │
└───────────────────────────────────────────────────────────────┘

Accuracy
  0.86 │                      ● C  ← Pareto optimal
       │                    ●
  0.85 │              ● B   ●     ← Pareto optimal
       │            ●
  0.84 │      ● A                 ← Pareto optimal
       │    ●   ●
  0.83 │  ●       ●
       │        ●     (dominated by B)
  0.82 │    ●
       │  ●
  0.81 │●
       └────┴─────┴─────┴─────┴─────┴─────→ F1-Macro
          0.78  0.80  0.82  0.84  0.86

Pareto Front: {A, B, C}
  └─ No single solution is better in BOTH objectives

Trade-off Selection:
  ├─ A: Balanced (F1=0.82, Acc=0.84)
  ├─ B: High F1 (F1=0.85, Acc=0.85)
  └─ C: High Accuracy (F1=0.84, Acc=0.86)

Choose based on your priority!
```

## 💡 Key Insights

### Why Bayesian Optimization?

| Method | Trials Needed | Intelligence | Efficiency |
|--------|--------------|--------------|------------|
| Grid Search | 1000+ | ❌ Blind | 5% |
| Random Search | 100+ | ❌ Blind | 30% |
| **Bayesian Opt** | **30-50** | **✅ Smart** | **80%** |

### Why Pruning?

Without pruning:
- 30 trials × 10 epochs = 300 epochs total
- Time: ~15 hours

With Hyperband pruning:
- 30 trials started
- 15 pruned early (avg 3 epochs each)
- 15 completed (10 epochs each)
- Total: 45 + 150 = 195 epochs
- Time: ~10 hours
- **Savings: 35%!**

### Parameter Importance (Typical)

```
Learning Rate         ████████████████████ 100% (most important)
Weight Decay          ████████████░░░░░░░░  60%
Batch Size            ██████████░░░░░░░░░░  50%
Warmup Ratio          ██████░░░░░░░░░░░░░░  30%
LR Scheduler          ████░░░░░░░░░░░░░░░░  20%
Epochs                ████░░░░░░░░░░░░░░░░  20%
Optimizer             ██░░░░░░░░░░░░░░░░░░  10%
Gradient Accum        ██░░░░░░░░░░░░░░░░░░  10%
Dropout               ░░░░░░░░░░░░░░░░░░░░   5%
Max Grad Norm         ░░░░░░░░░░░░░░░░░░░░   5%
```

**Focus on the top 3-4 for quick tuning!**

---

**See also**: `HYPERPARAMETER_TUNING_GUIDE.md` for detailed documentation.
