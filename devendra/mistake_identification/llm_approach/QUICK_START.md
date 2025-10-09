# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Open the Notebook
```bash
cd /DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach
jupyter notebook mistake_llm.ipynb
```

### Step 2: Run All Cells
- Click **Cell** → **Run All** (or use Shift+Enter for each cell)
- Default: Processes 10 samples (~5-10 minutes)

### Step 3: Check Results
Look for these output files in the same directory:
- `trainset_with_answers.json` ✅
- `confusion_matrix_overall.png` 📊
- `performance_analysis.png` 📈
- `summary_report.txt` 📝

## 📋 What the Notebook Does

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Load Data                                          │
│  • Reads trainset.json from data/ folder                   │
│  • Shows conversation structure and statistics             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Generate Answers (Using Mathstral)                │
│  • Extracts math questions from conversations              │
│  • Solves each problem using Mathstral-7B LLM             │
│  • Adds answer_key to each conversation                    │
│  • Saves: trainset_with_answers.json                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Evaluate Mistake Identification (Using Llama 3.1) │
│  • For each tutor response (GPT-4, Llama, Expert, etc.):  │
│    - Analyzes conversation + answer + response            │
│    - Determines if mistake was identified                 │
│    - Categorizes: "Yes" / "No" / "To some extent"        │
│  • Adds predictions to annotations                         │
│  • Saves: trainset_with_answers_and_evaluations.json      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Analyze & Visualize Results                       │
│  • Generates confusion matrices (overall + per-source)     │
│  • Calculates accuracy, F1 scores, metrics                │
│  • Creates performance comparison plots                    │
│  • Performs error analysis                                 │
│  • Outputs: Multiple PNG charts + CSV + report            │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **State-of-Art Models** | Mathstral for math + Llama 3.1 for evaluation |
| **Comprehensive** | Analyzes all tutor sources (GPT-4, Expert, etc.) |
| **Visual** | Beautiful confusion matrices and performance plots |
| **Detailed** | Metrics, error analysis, sample comparisons |
| **Scalable** | Test on 10 samples, then scale to full dataset |

## ⚙️ Configuration

### Default Settings (Good for Testing)
```python
NUM_SAMPLES_TO_PROCESS = 10  # Quick test
```

### Full Dataset Processing
```python
NUM_SAMPLES_TO_PROCESS = None  # Process all
```

### Model Selection
```python
# Math Solver (in Step 4)
math_solver = MathSolver(model_name="mistralai/Mathstral-7B-v0.1")

# Evaluator (in Step 7) 
evaluator = MistakeIdentificationEvaluator(
    model_name="meta-llama/Llama-3.1-8B-Instruct"  # or 70B
)
```

## 📊 Expected Output

### Files Created
```
llm_approach/
├── trainset_with_answers.json              [Data with answers]
├── trainset_with_answers_and_evaluations.json  [Full results]
├── confusion_matrix_overall.png            [Overall CM]
├── confusion_matrices_by_source.png        [Per-source CMs]
├── performance_analysis.png                [4-panel metrics plot]
├── performance_metrics_by_source.csv       [Metrics table]
├── error_analysis.csv                      [Error details]
└── summary_report.txt                      [Text summary]
```

### Sample Confusion Matrix Output
```
              Predicted
              Yes  To some extent  No
Actual  Yes   [80%]    [15%]      [5%]
        TSE   [20%]    [70%]      [10%]
        No    [10%]    [20%]      [70%]
```

### Sample Metrics Output
```
Source          Samples  Accuracy  F1_Macro  F1_Weighted
Expert          10       0.900     0.887     0.895
GPT4            10       0.850     0.832     0.845
Llama31405B     10       0.800     0.782     0.795
...
```

## 🔧 Troubleshooting

### ❌ "CUDA out of memory"
**Fix**: Reduce `NUM_SAMPLES_TO_PROCESS` or load models one at a time

### ❌ "Model not found"
**Fix**: Run `huggingface-cli login` and accept model licenses

### ❌ "Package not found"
**Fix**: All packages should be installed. If not:
```bash
pip install transformers torch pandas matplotlib seaborn scikit-learn
```

### ⚠️ Slow performance
**Check**: GPU is being used (first cell shows "CUDA available: True")

## 📖 Documentation

| File | Purpose |
|------|---------|
| `README.md` | Full documentation and user guide |
| `IMPLEMENTATION_SUMMARY.md` | What was built and technical details |
| `CHECKLIST.md` | Pre-flight checklist and verification |
| `QUICK_START.md` | This file - quick reference |

## 💡 Tips

1. **Start Small**: Run with 10 samples first to verify everything works
2. **Monitor Progress**: Progress bars show processing status
3. **Check Logs**: Look for errors in cell outputs
4. **GPU Usage**: First cell shows if GPU is available
5. **Save Often**: Results are automatically saved after each major step

## 🎓 Understanding the Results

### Confusion Matrix
- **Diagonal = Good**: High values on diagonal mean accurate predictions
- **Off-diagonal = Errors**: Shows what predictions were confused

### F1 Scores
- **Close to 1.0 = Excellent**
- **Macro**: Treats all classes equally
- **Weighted**: Accounts for class size

### Agreement Rate
- **% of predictions matching actual labels**
- Higher = better model performance

## ⏱️ Time Estimates

| Dataset Size | With GPU | Without GPU |
|--------------|----------|-------------|
| 10 samples   | 5-10 min | 30-60 min   |
| 100 samples  | 30-60 min| 4-6 hours   |
| Full dataset | 2-4 hours| 1-2 days    |

## 🚦 Status Indicators in Notebook

Watch for these in output:
- ✅ Green checkmarks = Success
- ⚠️ Warnings = Non-critical issues
- ❌ Errors = Need attention
- Progress bars = Current processing status

## 📞 Need Help?

1. Check the cell outputs for error messages
2. Review `README.md` for detailed explanations
3. Look at `IMPLEMENTATION_SUMMARY.md` for technical details
4. Verify environment with `./setup.sh`

---

**Ready to start?** Just open the notebook and run all cells! 🚀
