# 📂 Project Structure - LLM Approach

```
llm_approach/
│
├── 📜 Core Pipeline Scripts
│   ├── pedagogical_assessment_pipeline.py  [27KB] ⭐ Main pipeline implementation
│   ├── test_pipeline.py                    [2.5KB] Quick test script (5 samples)
│   └── setup.sh                            [1.9KB] One-command setup script
│
├── 📓 Interactive Notebook
│   └── pedagogical_pipeline_notebook.ipynb [20KB] Jupyter notebook with visualizations
│
├── 📚 Documentation
│   ├── IMPLEMENTATION_SUMMARY.md           [9.7KB] ⭐ What was built & how to use
│   ├── GETTING_STARTED.md                  [6.8KB] Quick start guide
│   ├── README_PIPELINE.md                  [6.7KB] Detailed documentation
│   └── PROJECT_STRUCTURE.md                [this file]
│
├── 📦 Configuration
│   └── requirements_pipeline.txt           [219B]  Dependencies
│
└── 📁 Output Directories (created by setup.sh)
    ├── output/              → Full pipeline outputs
    ├── test_output/         → Test run outputs (5 samples)
    └── notebook_output/     → Jupyter notebook outputs
```

## 🎯 Where to Start

### 1️⃣ First Time? Start Here!
```
IMPLEMENTATION_SUMMARY.md  → Read this first to understand what was built
    ↓
GETTING_STARTED.md         → Follow the quick start guide
    ↓
setup.sh                   → Run this to set everything up
    ↓
test_pipeline.py           → Test with 5 samples
```

### 2️⃣ Ready to Process Data?
```
pedagogical_assessment_pipeline.py
    ↓
    Use with various options:
    --limit 10              → Process 10 samples
    --limit 100             → Process 100 samples
    (no limit)              → Process all samples
```

### 3️⃣ Want Interactive Exploration?
```
pedagogical_pipeline_notebook.ipynb
    ↓
    Open in Jupyter for:
    - Step-by-step execution
    - Visualizations
    - Analysis tools
```

## 📊 Output Structure

When you run the pipeline, it creates:

```
output/  (or test_output/ for tests)
│
├── trainset_with_answers.json         [Original + Answer Keys]
│   └── Contains: conversation + math_problem + answer_key
│
├── trainset_fully_evaluated.json      [Everything + Evaluations]
│   └── Contains: all above + evaluation per tutor response
│
├── dataset_statistics.json            [Summary Stats]
│   └── Contains: counts, distributions, agreement rates
│
├── train_split.json                   [70% for training]
├── validation_split.json              [15% for validation]
└── test_split.json                    [15% for testing]
```

## 🔧 Pipeline Components

### Class: MathProblemSolver
```python
Responsibilities:
- Extract math problems from conversations
- Call LLM to solve problems
- Generate answer keys with solution steps
- Support multiple LLM providers
```

### Class: MistakeIdentificationEvaluator
```python
Responsibilities:
- Evaluate tutor responses
- Predict "Mistake_Identification" category
- Predict "Providing_Guidance" category
- Compare with human annotations
- Generate confidence scores
```

### Class: PedagogicalAssessmentPipeline
```python
Responsibilities:
- Orchestrate full pipeline
- Load and save datasets
- Generate statistics
- Create train/val/test splits
- Command-line interface
```

## 🎓 Usage Patterns

### Pattern 1: Quick Test
```bash
python test_pipeline.py
# → Processes 5 samples
# → Output in test_output/
# → Takes ~3 minutes
```

### Pattern 2: Incremental Processing
```bash
# Step 1: Solve problems only
python pedagogical_assessment_pipeline.py \
    --limit 50 \
    --skip-evaluation \
    --skip-splits

# Step 2: Add evaluations
python pedagogical_assessment_pipeline.py \
    --skip-solving \
    --skip-splits

# Step 3: Create splits
python pedagogical_assessment_pipeline.py \
    --skip-solving \
    --skip-evaluation
```

### Pattern 3: Full Pipeline
```bash
# One command, processes everything
python pedagogical_assessment_pipeline.py \
    --rate-limit-delay 2.0 \
    --output-dir full_output
```

### Pattern 4: Interactive Exploration
```bash
jupyter notebook pedagogical_pipeline_notebook.ipynb
# → Run cells step by step
# → See visualizations
# → Analyze results
```

## 📝 File Descriptions

| File | Size | Purpose | When to Use |
|------|------|---------|-------------|
| `pedagogical_assessment_pipeline.py` | 27KB | Complete pipeline | Production runs |
| `test_pipeline.py` | 2.5KB | Quick test | Before full run |
| `setup.sh` | 1.9KB | Setup script | First time setup |
| `pedagogical_pipeline_notebook.ipynb` | 20KB | Interactive | Exploration |
| `IMPLEMENTATION_SUMMARY.md` | 9.7KB | Overview | Start here |
| `GETTING_STARTED.md` | 6.8KB | Quick guide | For quick start |
| `README_PIPELINE.md` | 6.7KB | Full docs | Reference |
| `requirements_pipeline.txt` | 219B | Dependencies | Installation |

## 🚀 Typical Workflow

```
Day 1: Setup & Test
├── Read IMPLEMENTATION_SUMMARY.md
├── Run setup.sh
├── Set API key
├── Run test_pipeline.py
└── Review test_output/

Day 2: Process Sample
├── Process 50 samples
├── Review outputs
├── Check statistics
└── Verify quality

Day 3: Full Processing
├── Run full pipeline
├── Monitor progress
├── Handle any errors
└── Generate final splits

Day 4+: Analysis & Modeling
├── Load train/val/test splits
├── Train models
├── Validate performance
└── Final evaluation
```

## 💡 Quick Commands Reference

### Setup
```bash
bash setup.sh
export OPENAI_API_KEY='your-key'
```

### Test (5 samples)
```bash
python test_pipeline.py
```

### Process 10 samples
```bash
python pedagogical_assessment_pipeline.py --limit 10
```

### Process 100 samples
```bash
python pedagogical_assessment_pipeline.py --limit 100 --rate-limit-delay 2.0
```

### Use Claude instead of GPT
```bash
export ANTHROPIC_API_KEY='your-key'
python pedagogical_assessment_pipeline.py \
    --provider anthropic \
    --solver-model claude-3-5-sonnet-20241022 \
    --limit 10
```

### Interactive exploration
```bash
jupyter notebook pedagogical_pipeline_notebook.ipynb
```

### View results
```bash
cat output/dataset_statistics.json | python -m json.tool
```

## 📈 Expected Outputs

### After test_pipeline.py (5 samples)
```
test_output/
├── trainset_with_answers.json       [~50KB]
├── trainset_fully_evaluated.json    [~100KB]
├── dataset_statistics.json          [~2KB]
├── train_split.json                 [~70KB]
├── validation_split.json            [~15KB]
└── test_split.json                  [~15KB]
```

### After full pipeline (~1000 samples)
```
output/
├── trainset_with_answers.json       [~10MB]
├── trainset_fully_evaluated.json    [~20MB]
├── dataset_statistics.json          [~5KB]
├── train_split.json                 [~14MB]
├── validation_split.json            [~3MB]
└── test_split.json                  [~3MB]
```

## 🎯 Next Steps

1. ✅ Read `IMPLEMENTATION_SUMMARY.md`
2. ✅ Follow `GETTING_STARTED.md`
3. ✅ Run `test_pipeline.py`
4. ✅ Review outputs
5. ✅ Scale up processing
6. ✅ Train your models!

---

**Legend:**
- ⭐ = Essential files
- [Size] = Approximate file size
- → = Leads to
- ✅ = Recommended action
