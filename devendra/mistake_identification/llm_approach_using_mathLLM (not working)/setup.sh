#!/bin/bash
# Quick start script for LLM-based Mistake Identification Analysis

echo "================================================"
echo "LLM Mistake Identification Analysis - Setup"
echo "================================================"
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment is active."
    echo "Please activate your environment first:"
    echo "  conda activate culture"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Check CUDA availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPU detected')"
echo ""

# Check if required packages are installed
echo "Checking required packages..."
REQUIRED_PACKAGES=(
    "transformers"
    "torch"
    "accelerate"
    "tqdm"
    "pandas"
    "matplotlib"
    "seaborn"
    "scikit-learn"
)

MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    python -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo "✓ All required packages are installed"
else
    echo "✗ Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Install missing packages with:"
    echo "  pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi
echo ""

# Check Hugging Face authentication
echo "Checking Hugging Face authentication..."
if [ -f "$HOME/.cache/huggingface/token" ] || [ -n "$HF_TOKEN" ]; then
    echo "✓ Hugging Face token found"
else
    echo "⚠ Hugging Face token not found"
    echo "  You may need to authenticate to access Llama models:"
    echo "  huggingface-cli login"
fi
echo ""

# Check data files
echo "Checking data files..."
DATA_DIR="../../../data"
if [ -f "$DATA_DIR/trainset.json" ]; then
    NUM_CONVERSATIONS=$(python -c "import json; print(len(json.load(open('$DATA_DIR/trainset.json'))))")
    echo "✓ trainset.json found ($NUM_CONVERSATIONS conversations)"
else
    echo "✗ trainset.json not found in $DATA_DIR"
    exit 1
fi
echo ""

# Create output directory
echo "Creating output directory..."
mkdir -p results
echo "✓ Output directory ready"
echo ""

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To start the analysis:"
echo "  1. Open the notebook: jupyter notebook mistake_llm.ipynb"
echo "  2. Run all cells or customize parameters"
echo "  3. Check results in this directory"
echo ""
echo "For testing, the notebook processes 10 samples by default."
echo "To process all data, set NUM_SAMPLES_TO_PROCESS = None in Step 5."
echo ""
echo "Estimated time:"
echo "  - 10 samples: ~5-10 minutes (with GPU)"
echo "  - Full dataset: ~2-4 hours (depends on GPU and dataset size)"
echo ""
