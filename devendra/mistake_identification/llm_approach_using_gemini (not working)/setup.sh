#!/bin/bash
# Setup script for Pedagogical Assessment Pipeline

echo "=========================================="
echo "Pedagogical Assessment Pipeline Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p llm_approach/output
mkdir -p llm_approach/test_output
mkdir -p llm_approach/notebook_output
echo "✓ Directories created"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install openai anthropic tqdm matplotlib seaborn pandas -q
echo "✓ Dependencies installed"
echo ""

# Check for API keys
echo "Checking for API keys..."
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OPENAI_API_KEY is set"
else
    echo "⚠  OPENAI_API_KEY is not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key-here'"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✓ ANTHROPIC_API_KEY is set"
else
    echo "⚠  ANTHROPIC_API_KEY is not set"
    echo "   Set it with: export ANTHROPIC_API_KEY='your-key-here'"
fi
echo ""

# Make scripts executable
echo "Making scripts executable..."
chmod +x test_pipeline.py
chmod +x pedagogical_assessment_pipeline.py
echo "✓ Scripts are now executable"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set your API key (if not already set):"
echo "   export OPENAI_API_KEY='your-key-here'"
echo ""
echo "2. Run a quick test (processes 5 samples):"
echo "   python test_pipeline.py"
echo ""
echo "3. Or run the full pipeline:"
echo "   python pedagogical_assessment_pipeline.py --limit 10"
echo ""
echo "4. Or use the Jupyter notebook for interactive exploration:"
echo "   jupyter notebook pedagogical_pipeline_notebook.ipynb"
echo ""
