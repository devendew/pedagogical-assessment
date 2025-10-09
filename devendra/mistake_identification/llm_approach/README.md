# LLM-based Mistake Identification Analysis

This directory contains a comprehensive notebook for evaluating mistake identification in pedagogical conversations using state-of-the-art LLMs.

## Overview

The notebook `mistake_llm.ipynb` implements a complete pipeline that:

1. **Loads the dataset** from `trainset.json`
2. **Generates answer keys** using Mathstral (Mistral's math-focused LLM)
3. **Evaluates mistake identification** using Llama 3.1 8B to assess whether tutor responses correctly identified student mistakes
4. **Analyzes results** with confusion matrices, performance metrics, and visualizations

## Workflow

### Phase 1: Answer Generation
- Extracts math questions from conversation histories
- Uses **Mathstral-7B-v0.1** to solve problems and generate reference answers
- Adds answer keys to each conversation in the dataset

### Phase 2: Mistake Identification Evaluation
- For each tutor response (from GPT-4, Llama, Expert, etc.):
  - Analyzes the conversation context and correct answer
  - Evaluates if the tutor correctly identified the student's mistake
  - Categorizes as: **"Yes"**, **"No"**, or **"To some extent"**
- Uses **Llama-3.1-8B-Instruct** (can upgrade to 70B)

### Phase 3: Analysis & Visualization
- Generates confusion matrices (overall and per-source)
- Calculates accuracy, F1 scores, and other metrics
- Creates comprehensive visualizations
- Performs error analysis on misclassifications

## Requirements

### Python Packages
```bash
pip install transformers torch accelerate tqdm pandas matplotlib seaborn scikit-learn
```

### Hardware Requirements
- **Recommended**: NVIDIA GPU with 32GB+ VRAM
- **Minimum**: 16GB VRAM (load models sequentially)
- **Fallback**: CPU inference (slower but functional)

### Model Access
Ensure you have access to:
- `mistralai/Mathstral-7B-v0.1`
- `meta-llama/Llama-3.1-8B-Instruct`

You may need to:
1. Accept license agreements on Hugging Face
2. Authenticate with `huggingface-cli login`

## Usage

1. **Open the notebook:**
   ```bash
   jupyter notebook mistake_llm.ipynb
   ```

2. **Run all cells sequentially** or customize:
   - Adjust `NUM_SAMPLES_TO_PROCESS` in Step 5 (set to `None` for full dataset)
   - Change model names if using alternatives
   - Modify evaluation prompts for different criteria

3. **Review outputs** in the generated files

## Output Files

The notebook generates:

### Data Files
- `trainset_with_answers.json` - Original data + answer keys
- `trainset_with_answers_and_evaluations.json` - Full dataset with predictions
- `performance_metrics_by_source.csv` - Metrics for each tutor source
- `error_analysis.csv` - Detailed error cases

### Visualizations
- `confusion_matrix_overall.png` - Overall confusion matrix
- `confusion_matrices_by_source.png` - Per-source confusion matrices
- `performance_analysis.png` - Comprehensive performance comparison

### Reports
- `summary_report.txt` - Complete analysis summary

## Key Features

### Robust Error Handling
- Fallback mechanisms when models can't load
- Graceful handling of edge cases
- Comprehensive error logging

### Scalability
- Process subset for testing (default: 10 samples)
- Easy switch to full dataset processing
- Batch processing with progress bars

### Comprehensive Analysis
- Multiple performance metrics (Accuracy, F1-macro, F1-weighted)
- Per-source and per-label breakdowns
- Agreement analysis and common error patterns

## Model Alternatives

### Math Solvers
If Mathstral is unavailable, try:
- `mistralai/Mistral-7B-Instruct-v0.2`
- `WizardLM/WizardMath-7B-V1.1`
- `Qwen/Qwen2.5-Math-7B-Instruct`

### Evaluators
For the evaluation model:
- `meta-llama/Llama-3.1-70B-Instruct` (better accuracy, needs more memory)
- `meta-llama/Llama-2-7b-chat-hf`
- Or implement API-based evaluation (GPT-4, Claude, etc.)

## Customization

### Modify Evaluation Criteria
Edit the prompt in `MistakeIdentificationEvaluator.evaluate_mistake_identification()` to change evaluation logic.

### Add New Metrics
Extend the `calculate_metrics()` function to compute additional performance measures.

### Change Visualization Style
Modify the plotting functions to adjust colors, layouts, or add new visualizations.

## Troubleshooting

### Out of Memory Error
- Reduce `NUM_SAMPLES_TO_PROCESS`
- Load models sequentially (unload one before loading another)
- Use quantized models (4-bit or 8-bit)
- Enable CPU offloading

### Model Loading Issues
- Verify Hugging Face authentication
- Check model availability and licenses
- Try alternative models
- Use fallback rule-based methods

### Slow Inference
- Enable GPU acceleration
- Use smaller models
- Process smaller batches
- Consider API-based inference

## Results Interpretation

### Confusion Matrix
- **Diagonal elements**: Correct predictions
- **Off-diagonal**: Misclassifications
- Percentages show row-wise distribution

### F1 Scores
- **Macro**: Unweighted average across classes
- **Weighted**: Accounts for class imbalance
- Per-class scores show category-specific performance

### Agreement Rate
- Percentage of predictions matching actual labels
- Higher is better, but consider class distribution

## Citation

If you use this code in your research, please cite the pedagogical assessment project.

## License

See LICENSE file in the repository root.

## Contact

For questions or issues, please open an issue in the repository or contact the maintainers.
