# F1 Macro Optimization Strategy

## Current Performance Analysis

### Current Metrics (10 samples)
- **Accuracy**: 71.43%
- **F1 Macro**: **0.4468** ⚠️
- **F1 Weighted**: 72.98%

### Per-Class F1 Scores
| Class | F1 Score | Status |
|-------|----------|--------|
| Yes | 0.8403 | ✅ Excellent |
| **To some extent** | **0.0000** | ❌ **CRITICAL BOTTLENECK** |
| No | 0.5000 | ⚠️ Moderate |

## Problem Diagnosis

### The "To some extent" Bottleneck

**Current State:**
- Actual: 7 samples (8.3%)
- Predicted: 2 samples (2.4%)
- **0 correct predictions** (all 7 cases misclassified)
- Confusion: 2 predicted as "Yes", 5 as "No"

**Impact on F1 Macro:**
- Current F1 Macro: 0.4468
- If "To some extent" reaches 0.50: **F1 Macro = 0.6134** (+37%)
- If "To some extent" reaches 0.60: **F1 Macro = 0.6468** (+45%)
- If "To some extent" reaches 0.70: **F1 Macro = 0.6801** (+52%)

## Root Causes

1. **Severe Class Imbalance**: 
   - Yes: 79.8%
   - No: 11.9%
   - To some extent: **8.3%** (very rare)

2. **Ambiguous Category**:
   - "To some extent" is inherently fuzzy
   - Sits between "Yes" and "No"
   - Harder for LLM to distinguish

3. **Model Bias**:
   - Model predicts "To some extent" only 2.4% of time
   - Under-predicting by 70% (should be 8.3%)

4. **Prompt Design**:
   - Current prompt may not adequately define "To some extent"
   - Few-shot examples may not cover partial identification well

## Optimization Strategies

### Strategy 1: Enhanced "To some extent" Detection (RECOMMENDED)

**Approach**: Improve prompt and rules to better identify partial identifications.

**Changes:**
1. Add more "To some extent" examples in few-shot prompt
2. Create specific heuristics for partial identification:
   - Single guiding question without explicit error mention
   - Implicit hints without direct correction
   - Acknowledgment + redirection without stating mistake
3. Adjust decision thresholds to be more sensitive to partial cases

**Expected Impact**: F1 Macro → 0.55-0.65

**Implementation Difficulty**: ⭐⭐ (Medium)

### Strategy 2: Weighted Sampling / SMOTE for Minority Class

**Approach**: Use class weighting or synthetic oversampling.

**Changes:**
1. Implement SMOTE or class weights during evaluation
2. Train a fine-tuned model with balanced classes
3. Use ensemble methods with class-specific models

**Expected Impact**: F1 Macro → 0.60-0.70

**Implementation Difficulty**: ⭐⭐⭐⭐ (High - requires model fine-tuning)

### Strategy 3: Hierarchical Classification

**Approach**: Two-stage classification
1. First: Identify if mistake was identified (Yes/No)
2. Second: Determine degree (Full/Partial)

**Changes:**
1. Split evaluation into two steps
2. Train separate models for each step
3. Combine predictions

**Expected Impact**: F1 Macro → 0.60-0.68

**Implementation Difficulty**: ⭐⭐⭐⭐ (High)

### Strategy 4: Threshold Optimization

**Approach**: Optimize decision boundaries specifically for "To some extent".

**Changes:**
1. Use softer thresholds for partial identification
2. Implement probabilistic scoring instead of hard classification
3. Optimize thresholds on validation set

**Expected Impact**: F1 Macro → 0.55-0.62

**Implementation Difficulty**: ⭐⭐⭐ (Medium-High)

### Strategy 5: Ensemble Methods

**Approach**: Combine multiple models/strategies.

**Changes:**
1. Create 3 specialized evaluators:
   - One optimized for "Yes"
   - One optimized for "No"
   - One optimized for "To some extent"
2. Use voting or weighted combination

**Expected Impact**: F1 Macro → 0.58-0.68

**Implementation Difficulty**: ⭐⭐⭐⭐ (High)

## Quick Wins (Implementable Now)

### Quick Win #1: Enhanced Prompt with "To some extent" Focus

Add specific examples and explicit decision criteria:

```python
Example for PARTIAL identification:
Student error: Wrong calculation
Tutor: "Can you check your work on that step?"
Answer: PARTIAL (Question hints at error but doesn't explicitly state it)

Example for PARTIAL identification:
Student error: Incorrect formula
Tutor: "Let's think about the formula we're using here"
Answer: PARTIAL (Redirects to formula without explicitly correcting)
```

### Quick Win #2: Adjusted Heuristics

Modify `_quick_heuristic()` to be more sensitive:

```python
# NEW: Partial identification patterns
partial_patterns = [
    "can you check", "let's look at", "let's think about",
    "what about", "consider", "try to", "let's reconsider",
    "hmm", "interesting", "almost", "not quite"
]

partial_count = sum(1 for p in partial_patterns if p in lower)

# If has guiding language but no explicit identification
if partial_count >= 1 and strong_count == 0:
    return "To some extent"
```

### Quick Win #3: Re-calibrate Decision Logic

Make the evaluator more willing to predict "To some extent":

```python
# Current logic is too strict for "To some extent"
# Change from: guided_q >= 2 → "To some extent"
# To: guided_q >= 1 AND no_strong_yes → "To some extent"

if guided_count >= 1 and strong_count == 0 and not has_affirmation:
    return "To some extent"
```

## Recommended Implementation Plan

### Phase 1: Immediate (Today)
1. ✅ Enhance prompt with 3-4 "To some extent" examples
2. ✅ Add partial identification patterns to heuristics
3. ✅ Lower threshold for "To some extent" predictions
4. ✅ Test on 10 samples, measure improvement

**Expected**: F1 Macro 0.50-0.58

### Phase 2: Short-term (1-2 days)
1. Fine-tune Llama 3.1 on labeled data with class weights
2. Implement threshold optimization
3. Test on full dataset

**Expected**: F1 Macro 0.58-0.65

### Phase 3: Medium-term (3-5 days)
1. Implement ensemble approach
2. Add hierarchical classification
3. Extensive validation

**Expected**: F1 Macro 0.65-0.72

## Monitoring Metrics

Track these after each change:

1. **F1 Macro** (primary metric)
2. **"To some extent" F1** (bottleneck metric)
3. **"To some extent" Recall** (are we finding them?)
4. **"To some extent" Precision** (are we correct when we predict it?)
5. **Overall Accuracy** (shouldn't drop significantly)

## Expected Trade-offs

⚠️ **Warning**: Optimizing for "To some extent" may:
- Slightly decrease "Yes" class performance
- Increase false positives for "To some extent"
- Reduce overall accuracy by 2-5%

**This is acceptable** because F1 Macro treats all classes equally, which is the goal.

## Success Criteria

- ✅ F1 Macro > 0.55 (Phase 1 target)
- ✅ F1 Macro > 0.60 (Phase 2 target)
- ✅ F1 Macro > 0.65 (Phase 3 target)
- ✅ "To some extent" F1 > 0.30
- ✅ All class F1 > 0.40

---

**Next Steps**: Implement Phase 1 enhancements immediately.
