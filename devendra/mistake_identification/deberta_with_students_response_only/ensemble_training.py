#!/usr/bin/env python3
"""
Model Ensemble Training Script
Train multiple models and average predictions for better F1-Macro

Expected improvement: +3-5% (0.61 → 0.64-0.66)
Time required: 15-20 hours (5 models × 3-4 hours each)
"""

import json
import numpy as np
import torch
from pathlib import Path

# Configuration
SEEDS = [42, 123, 456, 789, 2024]
BASE_CONFIG = {
    'MODEL_NAME': "microsoft/deberta-v3-large",
    'NUM_EPOCHS': 25,
    'FOCAL_GAMMA': 3.0,
    'LEARNING_RATE': 1e-5,
    'BALANCE_STRATEGY': 'adaptive'
}

OUTPUT_BASE = "./results/ensemble"

def train_single_model(seed, config):
    """Train a single model with given seed."""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL WITH SEED {seed}")
    print(f"{'='*80}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Update output directory
    output_dir = f"{OUTPUT_BASE}/model_seed_{seed}"
    
    # TODO: Copy your training code here
    # This should include:
    # 1. Load data
    # 2. Create model
    # 3. Train
    # 4. Save model and predictions
    
    print(f"✅ Model {seed} trained and saved to {output_dir}")
    return output_dir

def load_predictions(model_dir, test_dataset):
    """Load predictions from a trained model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create trainer
    trainer = Trainer(model=model)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    return probs

def ensemble_predictions(model_dirs, test_dataset, method='average'):
    """Ensemble predictions from multiple models."""
    print(f"\n{'='*80}")
    print(f"CREATING ENSEMBLE ({method.upper()})")
    print(f"{'='*80}")
    
    # Load all predictions
    all_probs = []
    for model_dir in model_dirs:
        print(f"Loading predictions from: {model_dir}")
        probs = load_predictions(model_dir, test_dataset)
        all_probs.append(probs)
    
    # Ensemble methods
    if method == 'average':
        # Simple averaging
        ensemble_probs = np.mean(all_probs, axis=0)
    elif method == 'weighted':
        # Weighted by validation F1-macro (load from saved metrics)
        weights = []
        for model_dir in model_dirs:
            metrics_file = f"{model_dir}/trainer_state.json"
            with open(metrics_file, 'r') as f:
                state = json.load(f)
                # Get best F1-macro from logs
                best_f1 = max([log['eval_f1_macro'] for log in state['log_history'] 
                              if 'eval_f1_macro' in log])
                weights.append(best_f1)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        ensemble_probs = np.average(all_probs, axis=0, weights=weights)
    
    elif method == 'voting':
        # Hard voting
        predictions = [probs.argmax(axis=1) for probs in all_probs]
        from scipy import stats
        ensemble_preds = stats.mode(predictions, axis=0)[0]
        return ensemble_preds
    
    return ensemble_probs

def main():
    """Main ensemble training pipeline."""
    print("="*80)
    print("MODEL ENSEMBLE TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Number of models: {len(SEEDS)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Base config: {BASE_CONFIG}")
    print(f"\nEstimated time: {len(SEEDS) * 4} hours")
    print("="*80)
    
    # Step 1: Train all models
    model_dirs = []
    for seed in SEEDS:
        model_dir = train_single_model(seed, BASE_CONFIG)
        model_dirs.append(model_dir)
    
    print(f"\n✅ All {len(SEEDS)} models trained successfully!")
    
    # Step 2: Load test dataset
    # TODO: Load your test dataset here
    # test_dataset = ...
    
    # Step 3: Create ensemble predictions
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS")
    print("="*80)
    
    # Try different ensemble methods
    methods = ['average', 'weighted']
    
    for method in methods:
        print(f"\n📊 Method: {method}")
        ensemble_probs = ensemble_predictions(model_dirs, test_dataset, method=method)
        
        # Get final predictions
        ensemble_preds = ensemble_probs.argmax(axis=1)
        
        # Save predictions
        output_file = f"{OUTPUT_BASE}/ensemble_predictions_{method}.npy"
        np.save(output_file, ensemble_probs)
        print(f"  Saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✅ ENSEMBLE COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Load ensemble predictions from: {OUTPUT_BASE}/")
    print(f"2. Apply threshold optimization")
    print(f"3. Generate final predictions")
    print(f"\nExpected F1-Macro: 0.64-0.66")
    print("="*80)

if __name__ == "__main__":
    # Check if we want to train or just ensemble existing models
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--ensemble-only":
        print("Ensemble mode: Using existing models")
        # Load existing model directories
        model_dirs = [f"{OUTPUT_BASE}/model_seed_{seed}" for seed in SEEDS]
        # TODO: Create ensemble from existing models
    else:
        main()


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
## Basic Usage:

1. Train all models:
   python ensemble_training.py

2. Ensemble only (if models already trained):
   python ensemble_training.py --ensemble-only

## Integration with Existing Code:

Replace your single training loop with:

```python
for seed in SEEDS:
    # Set seed
    torch.manual_seed(seed)
    
    # Update output directory
    OUTPUT_DIR = f"./results/ensemble/model_seed_{seed}"
    
    # Run your existing training code
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
```

## Expected Results:

- Single model F1-Macro: 0.61
- 5-model ensemble F1-Macro: 0.64-0.66 (+3-5%)
- Training time: ~15-20 hours total

## Tips:

1. Train on separate GPUs if available (parallel training)
2. Use different FOCAL_GAMMA for each model (2.5, 3.0, 3.5, 4.0, 4.5)
3. Try different architectures:
   - Model 1-2: deberta-v3-large
   - Model 3-4: roberta-large
   - Model 5: deberta-v2-xlarge

4. Weight by validation F1-macro for better ensemble
"""
