# Student Answer-Based Approach for Mistake Identification

## Overview

This document describes a **new evaluation approach** for assessing whether tutors correctly identify student mistakes in mathematical problem-solving conversations.

### Date Created
October 10, 2025

---

## Motivation

The original approach used the **full conversation history** (including all tutor-student exchanges) to evaluate mistake identification. However, this introduced several challenges:

- **Noise from intermediate exchanges**: Multiple back-and-forth questions made it hard to identify the core mistake
- **Context confusion**: The model had to parse through entire conversations to understand what went wrong
- **Implicit vs explicit guidance**: Difficult to distinguish between guiding questions and actual mistake identification

---

## New Approach: Student Answer-Based Evaluation

### Methodology

The new approach simplifies the evaluation by focusing on three key components:

1. **The Question**: The original math problem
2. **Student's Work Only**: Extract only the student's responses (excluding intermediate tutor questions)
3. **Correct Answer**: The verified solution as reference
4. **Tutor's Response**: The feedback to evaluate

### Input Processing

```python
def extract_student_parts_only(conversation_history: str) -> tuple[str, str]:
    """
    Extract question and student responses only from conversation history.
    Returns: (question, student_responses)
    """
```

This function:
- Extracts the original question
- Collects all student responses
- **Excludes** intermediate tutor questions and guidance
- Provides clean input for evaluation

### Evaluation Logic

The evaluator analyzes if the tutor's response identifies errors by:

1. **Understanding the student's mistake** (by comparing student work to correct answer)
2. **Analyzing the tutor's feedback** to see if it addresses that mistake
3. **Classifying the identification level**:
   - **YES**: Tutor explicitly stated the error OR provided correcting calculation OR asked pointed question about the specific mistake
   - **PARTIAL**: Tutor gave general hints/questions without pinpointing the specific error
   - **NO**: Tutor praised incorrect work, gave no error feedback, or moved to different topic

---

## Implementation

### Model Used
- **LLM**: Llama 3.1 8B Instruct
- **Hardware**: CUDA GPU
- **Temperature**: 0.05 (low for consistency)
- **Max tokens**: 10 (for classification output)

### Prompt Strategy

The prompt includes:
- Clear classification rules (YES/PARTIAL/NO)
- 6 diverse examples covering different scenarios
- Structured input format (Question → Student Work → Correct Answer → Tutor Feedback)
- Direct instruction to answer with only one word

---

## Results (10 Conversations, 84 Tutor Responses)

### Overall Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 52.38% |
| **Macro F1-Score** | 39.22% |
| **Weighted F1-Score** | 59.04% |

### Per-Class Performance

#### Class: "Yes" (Mistake Identified)
- **Precision**: 100.00% ✅ (When predicted "Yes", it's always correct)
- **Recall**: 50.75% (Identifies about half of actual "Yes" cases)
- **F1-Score**: 67.33%
- **Support**: 67 instances

#### Class: "To some extent" (Partial Identification)
- **Precision**: 25.00%
- **Recall**: 14.29% ⚠️ (Struggles to identify partial guidance)
- **F1-Score**: 18.18%
- **Support**: 7 instances

#### Class: "No" (No Identification)
- **Precision**: 19.57%
- **Recall**: 90.00% ✅ (Good at identifying when tutors don't address mistakes)
- **F1-Score**: 32.14%
- **Support**: 10 instances

### Confusion Matrix Analysis

```
Actual vs Predicted:
                    Predicted
           Yes   Partial   No
Actual:
Yes         34      2      31  (50.7% correct, 46.3% missed)
Partial      0      1       6  (14.3% correct, 85.7% missed as "No")
No           0      1       9  (90% correct)
```

### Key Observations

#### Strengths ✅
1. **Perfect precision on "Yes"**: When the model predicts a tutor identified a mistake, it's always correct
2. **High recall on "No"**: Accurately identifies when tutors don't address mistakes (90%)
3. **Conservative predictions**: Avoids false positives for mistake identification

#### Challenges ⚠️
1. **High false negative rate**: 46% of actual "Yes" cases are predicted as "No"
   - Model is too conservative in recognizing mistake identification
   - May be missing implicit or subtle forms of identification
   
2. **Poor "Partial" detection**: Only 14% recall on "To some extent"
   - Difficult to distinguish general hints from specific guidance
   - Most partial cases (86%) are classified as "No"

3. **Recall-precision trade-off**: High precision comes at cost of low recall for "Yes" class

---

## Example Cases

### ✅ Correctly Identified "Yes"

**Student's mistake**: Calculated 10 pounds of meat instead of 5 pounds

**Tutor response**: *"You're close, but I notice that you calculated the cost of 10 pounds of meat, when actually 5 pounds of meat are needed for 5 sandwiches, since each sandwich requires 1 pound of meat; can you recalculate the cost of the meat?"*

**Prediction**: Yes ✓  
**Actual**: Yes ✓

**Analysis**: Clear explicit identification with correction

---

### ❌ Missed "Yes" (False Negative)

**Student's mistake**: Wrong calculation in rice problem

**Tutor response**: *"No. What is 20% of 20?"*

**Prediction**: No ✗  
**Actual**: Yes

**Analysis**: Model missed that this pointed question identifies the calculation error. Too brief/implicit for the model.

---

### ✅ Correctly Identified "No"

**Student's mistake**: Wrong answer

**Tutor response**: *"To find the area of a rectangle, multiply its length by its width."*

**Prediction**: No ✓  
**Actual**: No ✓

**Analysis**: Tutor gave generic instruction, didn't address student's error

---

### ⚠️ Partial Identification Challenge

**Student's mistake**: Calculation error in harvest problem

**Tutor response**: *"I see where you're going with this, but let's take a closer look at how we calculate the total after both harvests, not just the second harvest alone."*

**Prediction**: To some extent ✓  
**Actual**: To some extent ✓

**Analysis**: Rare success in identifying partial guidance (only 14% recall overall)

---

## Advantages of This Approach

### 1. **Cleaner Signal**
- Removes noise from intermediate tutor-student exchanges
- Focuses on student's actual work vs correct solution
- Clearer view of what mistake exists

### 2. **Objective Reference Point**
- Correct answer provides clear comparison
- Easier to identify discrepancies
- Less ambiguity in evaluation

### 3. **Direct Evaluation**
- Focuses solely on whether tutor identified the mistake
- Not confused by pedagogical style or question strategies
- More aligned with task definition

### 4. **Better Interpretability**
- Clear what student did wrong
- Clear what tutor said
- Clear if they match

---

## Limitations & Areas for Improvement

### 1. **Conservative Bias**
The model is too conservative in predicting "Yes":
- **Solution**: Adjust thresholds or add more explicit identification examples
- **Alternative**: Fine-tune the model on annotated data

### 2. **Implicit Identification**
Struggles with indirect forms of mistake identification:
- Short questions like "What is 20% of 20?"
- Guided questioning without explicit statements
- **Solution**: Add examples of implicit identification patterns

### 3. **Partial Class Detection**
Poor performance on "To some extent" (14% recall):
- Small class size (only 7 instances)
- Ambiguous boundary between partial and no identification
- **Solution**: Better definition and more training examples

### 4. **Context Understanding**
May lack full context of the pedagogical situation:
- Doesn't know previous conversation flow
- Can't see if tutor is building up to identification
- **Trade-off**: More context vs cleaner signal

---

## Comparison with Previous Approaches

### Previous Approach (Full Conversation History)
- Used entire conversation history
- Achieved ~58-60% accuracy
- Better at capturing conversational context
- More complex prompt engineering

### Student Answer-Based Approach (This Work)
- Uses only student work + answer
- Achieved ~52% accuracy on 10 conversations
- Cleaner, more interpretable
- Lower recall but higher precision on "Yes"

### Trade-offs
- **Lost context**: Some pedagogical nuance lost
- **Gained clarity**: Easier to understand what model evaluates
- **Different strengths**: Better at identifying explicit mistakes, worse at implicit

---

## Recommendations

### For Immediate Use
1. **Best for**: Detecting explicit mistake identification
2. **Reliable when**: Model predicts "Yes" (100% precision)
3. **Caution when**: Model predicts "No" (might be false negative)

### For Future Improvements

#### Short-term (Engineering)
1. **Add more examples** of implicit identification
2. **Adjust classification thresholds** to balance precision/recall
3. **Enhance "Partial" detection** with clearer boundaries
4. **Test on larger dataset** (currently only 10 conversations)

#### Medium-term (Model)
1. **Fine-tune LLM** on annotated pedagogical data
2. **Ensemble approach**: Combine with full-context model
3. **Multi-step reasoning**: First identify mistake, then check tutor response
4. **Prompt optimization**: A/B test different prompt structures

#### Long-term (Architecture)
1. **Specialized model**: Train model specifically for pedagogical assessment
2. **Hybrid approach**: LLM + rule-based for different identification types
3. **Active learning**: Incrementally improve with human feedback
4. **Multi-task learning**: Joint training on related pedagogical tasks

---

## Files Generated

| File | Description |
|------|-------------|
| `trainset_student_based_evaluations.json` | Full dataset with predictions |
| `student_based_analysis.csv` | Analysis-ready tabular data |
| `confusion_matrix_student_based.png` | Confusion matrix visualization |
| `STUDENT_ANSWER_BASED_APPROACH.md` | This documentation |

---

## Code Structure

### Main Components

1. **`extract_student_parts_only()`**: Extract student work from conversation
2. **`StudentAnswerBasedEvaluator`**: Main evaluation class
3. **`evaluate_all_responses_student_based()`**: Batch evaluation
4. **`prepare_analysis_data_student_based()`**: Data preparation for analysis
5. **`plot_confusion_matrix_comparison()`**: Visualization

### Usage Example

```python
# Initialize evaluator
evaluator = StudentAnswerBasedEvaluator(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    use_local=True
)

# Evaluate
prediction = evaluator.evaluate_mistake_identification(
    question="[Math problem]",
    student_responses="[Student's work only]",
    correct_answer="[Correct solution]",
    tutor_response="[Tutor's feedback]"
)
```

---

## Conclusion

The **Student Answer-Based Approach** provides a cleaner, more interpretable method for evaluating mistake identification in tutoring conversations. While it achieves slightly lower overall accuracy than the full-context approach, it offers:

- **Perfect precision** when predicting "Yes" (100%)
- **Clearer evaluation logic** (student work vs correct answer)
- **Better interpretability** (easy to understand what's being evaluated)

The main limitation is **conservative predictions** (low recall on "Yes"), suggesting the model is too cautious. Future work should focus on:
1. Recognizing implicit forms of mistake identification
2. Better distinguishing partial from no identification
3. Balancing precision and recall

This approach shows promise as a complementary or alternative method, especially when explicit mistake identification is the primary concern.

---

## Contact & Acknowledgments

**Developed by**: Devendra  
**Institution**: IIIT Hyderabad (cs24resch11011)  
**Date**: October 10, 2025  
**Related Work**: LLM-based pedagogical assessment for mistake identification

For questions or suggestions, please refer to the main project repository.
