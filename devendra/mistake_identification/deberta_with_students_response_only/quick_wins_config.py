#!/usr/bin/env python3
"""
Quick Wins Configuration for F1-Macro Improvement
Apply these changes to boost from 0.61 → 0.63-0.65

Usage:
1. Update your configuration cell with these values
2. Retrain the model
3. Expected improvement: +2-4% F1-macro
"""

# ============================================================================
# QUICK WIN #1: Higher Focal Gamma
# ============================================================================
# Rationale: More aggressive focus on hard/minority examples
# Current: 2.5 → New: 3.5
FOCAL_GAMMA = 3.5  # Try: 3.0, 3.5, 4.0

# ============================================================================
# QUICK WIN #2: Longer Training with More Patience
# ============================================================================
# Rationale: Model may not have fully converged at 15 epochs
NUM_EPOCHS = 40                     # Current: 15 → New: 40
EARLY_STOPPING_PATIENCE = 8         # Current: 5 → New: 8

# ============================================================================
# QUICK WIN #3: Better Learning Rate Schedule
# ============================================================================
# Rationale: Polynomial decay works better for long training
LR_SCHEDULER = 'polynomial'         # Current: cosine_with_restarts → New: polynomial
LEARNING_RATE = 1e-5                # Keep same
WARMUP_RATIO = 0.1                  # Current: 0.15 → New: 0.1

# ============================================================================
# QUICK WIN #4: Adjust Data Balancing
# ============================================================================
# Rationale: Adaptive balancing may work better than SMOTE
BALANCE_STRATEGY = 'adaptive'       # Current: smote → New: adaptive

# ============================================================================
# QUICK WIN #5: Slightly Increase Batch Size
# ============================================================================
# Rationale: Larger batch size = more stable gradients
BATCH_SIZE = 16                     # Current: 12 → New: 16
GRADIENT_ACCUMULATION = 6           # Adjust to keep effective batch = 96

# ============================================================================
# OPTIONAL: Class-Specific Costs
# ============================================================================
# Add this to your FocalLoss initialization:
"""
class_costs = torch.tensor([1.0, 3.0, 2.5])  # [Yes, To some extent, No]
focal_loss = WeightedFocalLoss(
    alpha=class_weights, 
    gamma=FOCAL_GAMMA,
    class_costs=class_costs
)
"""

# ============================================================================
# SUMMARY OF CHANGES
# ============================================================================
print("="*80)
print("QUICK WINS CONFIGURATION")
print("="*80)
print("\n📝 Changes from current setup:")
print(f"  1. FOCAL_GAMMA: 2.5 → 3.5")
print(f"  2. NUM_EPOCHS: 15 → 40")
print(f"  3. EARLY_STOPPING_PATIENCE: 5 → 8")
print(f"  4. LR_SCHEDULER: cosine_with_restarts → polynomial")
print(f"  5. BALANCE_STRATEGY: smote → adaptive")
print(f"  6. BATCH_SIZE: 12 → 16")
print(f"\n⏱️  Training time: ~12-15 hours")
print(f"📈 Expected F1-Macro: 0.63-0.65")
print(f"✨ Improvement: +2-4% from 0.61")
print("="*80)
