# F1 Macro Optimization Summary

## Objective
Maximize F1 Macro score for 3-class mistake identification ("Yes", "To some extent", "No")

## Current Status

### Baseline Performance
- **F1 Macro:** 0.4468
- **F1 Weighted:** 0.7475
- **Accuracy:** 72.6%
- **Per-Class F1:**
  - Yes: 0.8403 (excellent)
  - To some extent: 0.0000 (bottleneck - only 2/7 correct)
  - No: 0.5000 (acceptable)
- **Yes Recall:** 76.1% (51/67 correct)

### Bottleneck Analysis
- "To some extent" class has 0% F1, severely impacting macro average
- Only 7% of dataset (7/84 samples) are "To some extent" cases
- Class imbalance makes detection difficult
- **If "To some extent" F1 reaches 0.50, F1 Macro would jump to 0.6134 (+37%)**

## Optimization Attempts

### Version 1: Aggressive Partial Detection
- **Strategy:** Enhanced prompt with explicit "To some extent" examples and 20+ partial patterns
- **Result:** FAILED - predicted "To some extent" for ALL cases (84/84)
- **F1 Macro:** N/A (unusable)
- **Learning:** Too aggressive, no threshold logic

### Version 2: Balanced Three-Way Classification
- **Strategy:** Clear YES/PARTIAL/NO distinction with strict criteria
- **Result:** FAILED - too conservative, over-predicted "No" (54/84)
- **F1 Macro:** 0.2834 (-37% vs baseline)
- **Yes Recall:** 37.3% (down from 76%)
- **Learning:** Strict criteria broke "Yes" detection

### Version 3: Two-Stage Approach
- **Strategy:**
  1. First check for partial guidance patterns (questions/hints without strong identification)
  2. Then use baseline logic for "Yes" vs "No"
- **Results:**
  - **F1 Macro:** 0.4432 (-0.8% vs baseline, almost equivalent!)
  - **F1 Weighted:** 0.6009 (down from 0.7475)
  - **Accuracy:** 54.8%
  - **"To some extent" F1:** 0.2727 (up from 0.0000!) ✅
    - Recall: 42.9% (3/7 correct)
    - Precision: 20.0%
  - **"Yes" Recall:** 52.2% (down from 76%)
  - **"No" Recall:** 80.0% (8/10 correct)
- **Confusion Matrix:**
  ```
  Actual \ Predicted   | Yes | Partial | No  |
  -------------------------------------------|
  Yes (67)             |  35 |    11   | 21  |
  To some extent (7)   |   2 |     3   |  2  |
  No (10)              |   1 |     1   |  8  |
  ```
- **Analysis:**
  - ✅ Successfully detected some "To some extent" cases
  - ✅ F1 Macro nearly maintained
  - ⚠️ Still too conservative on "Yes" (32/67 misclassified)
  - ⚠️ Over-predicting "To some extent" for "Yes" cases (11/67)

## Key Insights

1. **The Challenge:** Improving "To some extent" without sacrificing "Yes" performance
2. **Baseline Strength:** 76% recall on "Yes" class is hard to maintain
3. **V3 Progress:** First version to achieve non-zero "To some extent" F1 (0.2727)
4. **Trade-off:** V3 traded "Yes" recall (76% → 52%) for "To some extent" detection
5. **Current Best:** V3 achieves F1 Macro = 0.4432 (nearly equal to baseline)

## Next Steps (V4 Strategy)

### Proposed Approach: Baseline-First with Refinement
1. **Use baseline evaluator for initial prediction** (proven 76% Yes recall)
2. **If prediction is "Yes", perform secondary check:**
   - Look for strong partial guidance indicators:
     - Multiple questions without answers
     - Hints without explicit error statements
     - Guided discovery patterns
   - If found, downgrade to "To some extent"
3. **Otherwise, keep baseline prediction**

### Expected Outcome
- Maintain ~70% "Yes" recall (close to baseline)
- Detect 40-50% of "To some extent" cases
- Achieve F1 Macro ≈ 0.55-0.60 (target range)

### Implementation Notes
- Use conservative threshold for downgrading "Yes" → "Partial"
- Require at least 2 partial indicators + absence of strong identification
- Maintain baseline's robust "No" detection
