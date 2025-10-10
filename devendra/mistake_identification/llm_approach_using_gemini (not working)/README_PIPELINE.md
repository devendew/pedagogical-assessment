# Pedagogical Assessment Pipeline

This pipeline processes pedagogical assessment data to:
1. Extract and solve math problems using LLMs
2. Generate answer keys for conversations
3. Evaluate tutor responses for mistake identification quality
4. Create train/validation/test splits

## Features

- **Math Problem Solving**: Uses state-of-the-art LLMs (GPT-4, Claude) to solve math problems
- **Automated Evaluation**: Evaluates whether tutor responses correctly identify student mistakes
- **Flexible Configuration**: Support for OpenAI and Anthropic models
- **Rate Limiting**: Built-in delays to respect API limits
- **Statistics Generation**: Comprehensive statistics about the dataset

## Installation

```bash
# Install dependencies
pip install -r requirements_pipeline.txt

# Set up API keys
export OPENAI_API_KEY="your-openai-api-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Quick Start

### Basic Usage (with default settings)

```bash
# Run full pipeline with GPT-4o
python pedagogical_assessment_pipeline.py

# Test with first 5 samples only
python pedagogical_assessment_pipeline.py --limit 5
```

### Using Anthropic Claude

```bash
python pedagogical_assessment_pipeline.py \
    --provider anthropic \
    --solver-model claude-3-5-sonnet-20241022 \
    --evaluator-model claude-3-5-sonnet-20241022
```

### Advanced Usage

```bash
# Custom directories and models
python pedagogical_assessment_pipeline.py \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --solver-model gpt-4o \
    --evaluator-model gpt-4o \
    --rate-limit-delay 2.0 \
    --limit 10
```

### Incremental Processing

If you've already completed some steps, you can skip them:

```bash
# Skip solving (use existing answers)
python pedagogical_assessment_pipeline.py --skip-solving

# Skip evaluation (use existing evaluations)
python pedagogical_assessment_pipeline.py --skip-evaluation

# Only create splits from existing data
python pedagogical_assessment_pipeline.py --skip-solving --skip-evaluation
```

## Output Files

The pipeline generates the following files in the output directory:

1. **`trainset_with_answers.json`**: Original dataset enriched with answer keys
   - Includes: `math_problem`, `answer_key` with solution steps

2. **`trainset_fully_evaluated.json`**: Dataset with all evaluations
   - Includes: All tutor responses evaluated for mistake identification

3. **`dataset_statistics.json`**: Statistics about the dataset
   - Agreement rates, distribution of categories, etc.

4. **`train_split.json`**: Training set (70%)
5. **`validation_split.json`**: Validation set (15%)
6. **`test_split.json`**: Test set (15%)

## Data Structure

### Input Format (trainset.json)
```json
{
  "conversation_id": "...",
  "conversation_history": "Tutor: ... Student: ...",
  "tutor_responses": {
    "GPT4": {
      "response": "...",
      "annotation": {
        "Mistake_Identification": "Yes",
        "Providing_Guidance": "Yes"
      }
    }
  }
}
```

### Output Format (with answers and evaluations)
```json
{
  "conversation_id": "...",
  "conversation_history": "...",
  "math_problem": "Tyson decided to make...",
  "answer_key": {
    "answer": "50",
    "solution_steps": "Step 1: ...",
    "explanation": "...",
    "model_used": "gpt-4o"
  },
  "tutor_responses": {
    "GPT4": {
      "response": "...",
      "annotation": {...},
      "evaluation": {
        "predicted_mistake_identification": "Yes",
        "predicted_providing_guidance": "Yes",
        "reasoning": "The tutor correctly identifies...",
        "confidence": 0.85,
        "agrees_with_human": true,
        "model_used": "gpt-4o"
      }
    }
  }
}
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `../../../data` | Input data directory |
| `--output-dir` | `./output` | Output directory |
| `--solver-model` | `gpt-4o` | Model for solving problems |
| `--evaluator-model` | `gpt-4o` | Model for evaluation |
| `--provider` | `openai` | LLM provider (openai/anthropic) |
| `--limit` | `None` | Max samples to process |
| `--rate-limit-delay` | `1.0` | Delay between API calls (sec) |
| `--skip-solving` | `False` | Skip problem solving |
| `--skip-evaluation` | `False` | Skip response evaluation |
| `--skip-splits` | `False` | Skip creating splits |

## Evaluation Criteria

The evaluator assesses tutor responses on two dimensions:

### Mistake Identification
- **Yes**: Clearly identifies the specific mistake(s)
- **No**: Fails to identify mistakes or identifies wrong issues
- **To some extent**: Partially identifies mistakes but is incomplete/vague

### Providing Guidance
- **Yes**: Provides clear, actionable guidance
- **No**: Provides no or unhelpful guidance
- **To some extent**: Provides some but incomplete/unclear guidance

## Tips

1. **Start Small**: Test with `--limit 5` before running on full dataset
2. **Rate Limiting**: Adjust `--rate-limit-delay` based on your API tier
3. **Cost Management**: Each sample requires multiple API calls; monitor costs
4. **Incremental Processing**: Use skip flags to resume from failures
5. **Model Selection**: 
   - GPT-4o: Fast, cost-effective, great accuracy
   - Claude 3.5 Sonnet: Excellent reasoning, slightly slower
   - GPT-4 Turbo: Good balance of speed and quality

## Troubleshooting

### API Key Issues
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set temporarily
export OPENAI_API_KEY="sk-..."
```

### Rate Limit Errors
- Increase `--rate-limit-delay` to 2.0 or higher
- Consider upgrading API tier

### Out of Memory
- Process in batches using `--limit`
- Reduce the number of samples

### Parsing Errors
- Some math problems may not be extracted correctly
- Check the `math_problem` field in output

## Example Workflow

```bash
# Step 1: Test with small sample
python pedagogical_assessment_pipeline.py --limit 5

# Step 2: Process first 100 samples
python pedagogical_assessment_pipeline.py --limit 100

# Step 3: Run full dataset (this will take time!)
python pedagogical_assessment_pipeline.py --rate-limit-delay 2.0

# Step 4: Analyze results
cat output/dataset_statistics.json | jq '.'
```

## Advanced: Using as a Library

```python
from pedagogical_assessment_pipeline import PedagogicalAssessmentPipeline

# Initialize
pipeline = PedagogicalAssessmentPipeline(
    data_dir="/path/to/data",
    output_dir="/path/to/output",
    solver_model="gpt-4o",
    evaluator_model="gpt-4o",
    provider="openai"
)

# Run individual steps
enriched_data = pipeline.process_trainset_with_answers(limit=10)
evaluated_data = pipeline.evaluate_all_responses(enriched_data)
train, val, test = pipeline.create_train_val_test_splits(evaluated_data)
```

## License

See main repository LICENSE file.

## Contact

For issues or questions, please open an issue in the repository.
