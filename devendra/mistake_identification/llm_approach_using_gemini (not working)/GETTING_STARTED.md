# Quick Start Guide - Pedagogical Assessment Pipeline

This guide will help you get started with the Pedagogical Assessment Pipeline in minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- An API key from either:
  - OpenAI (GPT-4o recommended)
  - Anthropic (Claude 3.5 Sonnet recommended)

## 🚀 Quick Start (5 minutes)

### Step 1: Setup

```bash
cd /DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach

# Run the setup script
bash setup.sh
```

### Step 2: Set API Key

```bash
# For OpenAI
export OPENAI_API_KEY='your-openai-api-key-here'

# OR for Anthropic
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
```

To make this permanent, add to your `~/.bashrc`:
```bash
echo "export OPENAI_API_KEY='your-key-here'" >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Test the Pipeline

Run a quick test with 5 samples:

```bash
python test_pipeline.py
```

This will:
- ✅ Solve 5 math problems
- ✅ Generate answer keys
- ✅ Evaluate tutor responses
- ✅ Create train/val/test splits
- ⏱️ Takes about 2-3 minutes

### Step 4: Check Results

```bash
cd test_output
ls -lh

# View the statistics
cat dataset_statistics.json | python -m json.tool
```

## 📊 Understanding the Output

### Generated Files

1. **`trainset_with_answers.json`**
   - Original conversations + correct answers
   - Math problems extracted and solved
   - Solution steps included

2. **`trainset_fully_evaluated.json`**
   - Everything from above
   - + Evaluations of each tutor response
   - + Predictions for mistake identification
   - + Confidence scores

3. **`dataset_statistics.json`**
   - Summary statistics
   - Agreement rates with human annotations
   - Distribution of categories

4. **`train_split.json`, `validation_split.json`, `test_split.json`**
   - 70% training / 15% validation / 15% test
   - Ready for model training

## 🎯 Next Steps

### Option 1: Process More Data

Process 50 conversations:
```bash
python pedagogical_assessment_pipeline.py --limit 50 --rate-limit-delay 2.0
```

Process entire dataset (⚠️ this will take hours and cost $$$):
```bash
python pedagogical_assessment_pipeline.py --rate-limit-delay 2.0
```

### Option 2: Use Jupyter Notebook

For interactive exploration:
```bash
jupyter notebook pedagogical_pipeline_notebook.ipynb
```

The notebook includes:
- 📊 Visualizations
- 🔍 Interactive exploration
- 📈 Statistics and plots
- Step-by-step execution

### Option 3: Customize Settings

Use different models:
```bash
# Use Claude 3.5 Sonnet
python pedagogical_assessment_pipeline.py \
    --provider anthropic \
    --solver-model claude-3-5-sonnet-20241022 \
    --evaluator-model claude-3-5-sonnet-20241022 \
    --limit 10

# Use different models for solving vs evaluating
python pedagogical_assessment_pipeline.py \
    --solver-model gpt-4o \
    --evaluator-model gpt-4o-mini \
    --limit 10
```

## 💡 Common Use Cases

### Use Case 1: Just Generate Answers

```bash
python pedagogical_assessment_pipeline.py \
    --limit 100 \
    --skip-evaluation \
    --skip-splits
```

This generates `trainset_with_answers.json` only.

### Use Case 2: Evaluate Existing Answers

If you already have answers:
```bash
python pedagogical_assessment_pipeline.py \
    --skip-solving \
    --limit 100
```

### Use Case 3: Just Create Splits

If you have everything processed:
```bash
python pedagogical_assessment_pipeline.py \
    --skip-solving \
    --skip-evaluation
```

## 🔍 Examining Results

### View a Sample Answer

```python
import json

with open('test_output/trainset_with_answers.json') as f:
    data = json.load(f)

# Look at first conversation
sample = data[0]
print("Problem:", sample['math_problem'])
print("\nAnswer:", sample['answer_key']['answer'])
print("\nSolution:", sample['answer_key']['solution_steps'])
```

### View Evaluations

```python
import json
import pandas as pd

with open('test_output/trainset_fully_evaluated.json') as f:
    data = json.load(f)

# Extract all evaluations
evals = []
for conv in data:
    for tutor, resp in conv['tutor_responses'].items():
        if resp and 'evaluation' in resp:
            evals.append({
                'tutor': tutor,
                'predicted_mi': resp['evaluation']['predicted_mistake_identification'],
                'human_mi': resp['annotation']['Mistake_Identification'],
                'agrees': resp['evaluation']['agrees_with_human']
            })

df = pd.DataFrame(evals)
print(df.groupby('tutor').agg({'agrees': 'sum'}))
```

## 🐛 Troubleshooting

### Issue: "No API key found"

**Solution:**
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY='sk-...'
```

### Issue: "Rate limit exceeded"

**Solution:** Increase the delay:
```bash
python pedagogical_assessment_pipeline.py --rate-limit-delay 5.0
```

### Issue: "Module not found"

**Solution:** Install dependencies:
```bash
pip install openai anthropic tqdm
```

### Issue: "Out of memory"

**Solution:** Process in smaller batches:
```bash
# Process 10 at a time
python pedagogical_assessment_pipeline.py --limit 10
```

## 💰 Cost Estimation

Approximate costs per conversation (as of Oct 2024):

**Using GPT-4o:**
- Problem solving: ~$0.02 per problem
- Evaluation: ~$0.03 per tutor response (8 tutors avg)
- **Total per conversation: ~$0.26**

**Using Claude 3.5 Sonnet:**
- Problem solving: ~$0.015 per problem
- Evaluation: ~$0.025 per tutor response
- **Total per conversation: ~$0.215**

**For full dataset (~1000 conversations):**
- GPT-4o: ~$260
- Claude 3.5: ~$215

💡 **Tip:** Start with `--limit 10` to test before running on full dataset!

## 📚 Documentation

- **Detailed README**: See `README_PIPELINE.md`
- **Code Documentation**: See docstrings in `pedagogical_assessment_pipeline.py`
- **Interactive Tutorial**: See `pedagogical_pipeline_notebook.ipynb`

## 🎓 Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Setup (one-time)
bash setup.sh
export OPENAI_API_KEY='your-key'

# 2. Test with 5 samples
python test_pipeline.py

# 3. Check results look good
cat test_output/dataset_statistics.json

# 4. Process first 100 conversations
python pedagogical_assessment_pipeline.py --limit 100 --output-dir output

# 5. Analyze results in Jupyter
jupyter notebook pedagogical_pipeline_notebook.ipynb

# 6. Process full dataset (when ready)
python pedagogical_assessment_pipeline.py --output-dir full_output
```

## 🤝 Need Help?

- Check the README: `README_PIPELINE.md`
- Look at the notebook: `pedagogical_pipeline_notebook.ipynb`
- Review the code: `pedagogical_assessment_pipeline.py`

## ✅ Checklist

Before running on full dataset:

- [ ] API key is set
- [ ] Tested with small sample (`--limit 5`)
- [ ] Reviewed sample outputs
- [ ] Checked cost estimates
- [ ] Have sufficient API credits
- [ ] Understand the output format

Ready to go? 🚀

```bash
python pedagogical_assessment_pipeline.py
```
