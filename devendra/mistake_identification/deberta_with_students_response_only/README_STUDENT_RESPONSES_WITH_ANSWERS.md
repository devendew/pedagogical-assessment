# DeBERTa Training with Student Responses and Answer Keys

## Overview
This notebook trains a DeBERTa model for mistake identification using:
1. **Student responses only** (extracted from conversation history)
2. **Answer keys** (correct solutions)
3. **Tutor responses** (to be classified)

## Key Changes from Original Approach

### 1. Data Source
- **Input**: `trainset_with_answers.json` (instead of `trainset.json`)
- **Additional field**: `answer_key` containing the correct solution

### 2. Data Processing
- **Extracts only student utterances** from conversation history
- **Removes tutor questions/prompts** to focus on student work
- **Includes answer key** as reference for correct solution

### 3. Model Input Format
```
[Student Response] [SEP] [Answer Key] [SEP] [Tutor Response]
```

**Example:**
```
Student Response: "To serve 20 people, Tyson needs to make 20/4 = 5 sandwiches..."
[SEP]
Answer Key: "1. First, we need to determine how many sandwiches... = $50.00"
[SEP]
Tutor Response: "Great, you've correctly identified the cost of the meat..."
```

## Why This Approach?

### Benefits
1. **Focused on student work**: Only student responses, not tutor prompts
2. **Ground truth available**: Answer key provides correct solution for reference
3. **Better context**: Model can compare student work against correct answer
4. **Reduced noise**: Eliminates potentially confusing tutor questions

### Comparison with Full Conversation
| Aspect | Full Conversation | Student Responses + Answers |
|--------|------------------|----------------------------|
| Input size | Larger (full history) | Smaller (student only) |
| Context | All interactions | Student work + solution |
| Noise | More (tutor prompts) | Less (focused content) |
| Answer key | Not included | Included |

## Files and Outputs

### Input Files
- `trainset_with_answers.json` - Training data with answer keys
- `../../../data/dev_testset.json` - Dev set (no labels)
- `../../../data/testset.json` - Test set (no labels)

### Generated Files
1. **Processed Data**
   - `trainset_student_responses_with_answers.json` - Cleaned training data

2. **Predictions**
   - `dev_predictions_student_responses_with_answers.csv` - Dev predictions
   - `test_predictions_student_responses_with_answers.csv` - Test predictions

3. **Visualizations**
   - `label_distribution.png` - Training label distribution
   - `confusion_matrix_validation.png` - Validation confusion matrix
   - `prediction_distributions_student_responses_with_answers.png` - Prediction distributions
   - `training_summary.png` - Training summary
   - `training_curves_comprehensive.png` - Detailed training metrics

4. **Model Artifacts**
   - `results/mistake-identification-deberta/` - Trained model
   - `training_history.json` - Training metrics history

## Data Processing Pipeline

### Step 1: Extract Student Responses
```python
def extract_student_responses_from_conversation(conv_history):
    """Extract only student utterances from conversation."""
    student_responses = []
    lines = conv_history.split('\n')
    
    for line in lines:
        if line.strip().startswith('Student:'):
            student_text = line.replace('Student:', '').strip()
            student_responses.append(student_text)
    
    return ' '.join(student_responses)
```

### Step 2: Create Training Samples
For each conversation:
- Extract student responses
- Get answer key
- For each tutor response model:
  - Create sample: `{student_responses, answer_key, tutor_response, label}`

### Step 3: Create Dataset
Combine components with `[SEP]` tokens for model input.

## Model Configuration

### Architecture
- **Base Model**: `microsoft/deberta-v3-large`
- **Task**: 3-class sequence classification
- **Labels**: Yes, To some extent, No

### Training Parameters
- **Epochs**: 20 (with early stopping)
- **Batch Size**: 8 (train), 16 (eval)
- **Learning Rate**: 2e-5
- **Max Length**: 512 tokens
- **Optimizer**: AdamW with warmup
- **Loss**: Weighted Cross-Entropy (class balanced)

### Evaluation Metrics
- Accuracy
- F1-Macro (primary metric)
- F1-Weighted
- Per-class Precision, Recall, F1

## Running the Notebook

### Prerequisites
```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn
```

### Execution
Run all cells sequentially. The notebook will:
1. Load and process training data
2. Extract student responses
3. Create processed trainset
4. Train DeBERTa model
5. Generate predictions on dev/test sets
6. Create visualizations

### Expected Runtime
- Data processing: ~1 minute
- Training: ~30-60 minutes (depending on GPU)
- Inference: ~5 minutes

## Results Interpretation

### CSV Output Format
Each prediction file contains:
- `conversation_id` - Unique conversation identifier
- `student_responses` - Extracted student utterances
- `tutor_response` - Tutor's response being classified
- `model` - Which tutor model generated the response
- `predicted_label` - Model's classification (Yes/To some extent/No)
- `prob_yes` - Confidence for "Yes" class
- `prob_to_some_extent` - Confidence for "To some extent" class
- `prob_no` - Confidence for "No" class

### Using Predictions
```python
import pandas as pd

# Load predictions
predictions = pd.read_csv('dev_predictions_student_responses_with_answers.csv')

# Filter high-confidence predictions
confident = predictions[predictions[['prob_yes', 'prob_to_some_extent', 'prob_no']].max(axis=1) > 0.8]

# Analyze by model
model_performance = predictions.groupby('model')['predicted_label'].value_counts()
```

## Comparison with Other Approaches

### Experiment Variations
1. **Full conversation** (original)
   - Input: Full conversation history + tutor response
   
2. **Student responses only** (this notebook)
   - Input: Student responses + answer key + tutor response
   
3. **Student responses without answers**
   - Input: Student responses + tutor response (no answer key)

Compare results across these variations to understand the impact of:
- Answer key inclusion
- Conversation history vs. student-only responses

## Next Steps

1. **Compare Results**: Evaluate against full conversation approach
2. **Analyze Answer Key Impact**: Test without answer keys
3. **Error Analysis**: Review misclassified samples
4. **Hyperparameter Tuning**: Optimize for better performance
5. **Deploy Model**: Use for real-time inference

## Troubleshooting

### Issue: Out of Memory
- Reduce batch size in TrainingArguments
- Decrease `MAX_LENGTH` from 512 to 256

### Issue: Poor Performance on Minority Class
- Adjust class weights in WeightedTrainer
- Try focal loss (set `USE_FOCAL_LOSS = True`)
- Oversample minority class (set `BALANCE_STRATEGY = 'oversample'`)

### Issue: Long Training Time
- Reduce number of epochs
- Use smaller model (e.g., `deberta-v3-base`)
- Enable mixed precision training (FP16)

## Citation

If you use this approach, please cite:
```
@misc{deberta-mistake-identification,
  title={Mistake Identification using Student Responses and Answer Keys},
  year={2025},
  note={DeBERTa-based approach for pedagogical assessment}
}
```
