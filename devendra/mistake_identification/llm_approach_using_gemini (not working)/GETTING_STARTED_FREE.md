# FREE LLM Pipeline - Quick Start Guide

This version uses **completely FREE** or **FREE-TIER** LLMs instead of paid APIs!

## 🆓 Free LLM Options

### Option 1: Ollama (Recommended - Completely Free, Local)

**Best for**: Complete privacy, no API limits, no costs

```bash
# Install Ollama
# Visit: https://ollama.ai

# Pull models (one-time)
ollama pull llama3.1:8b        # 8B model (fast, good)
ollama pull llama3.1:70b       # 70B model (slower, excellent)
ollama pull qwen2.5:7b         # Alternative
ollama pull mistral:7b         # Alternative

# Run pipeline
python pedagogical_assessment_pipeline_free.py --provider ollama --limit 5
```

**Pros**: 
- ✅ Completely free
- ✅ No API keys needed
- ✅ No rate limits
- ✅ Runs locally (private)
- ✅ No internet needed after download

**Cons**:
- ⚠️ Requires decent hardware (8GB+ RAM for 8B models)
- ⚠️ Slower than cloud APIs

---

### Option 2: Google Gemini (Free Tier - Best Quality)

**Best for**: High quality results, no local hardware needed

```bash
# Get free API key
# Visit: https://makersuite.google.com/app/apikey

# Set environment variable
export GEMINI_API_KEY='your-key-here'

# Install package
pip install google-generativeai

# Run pipeline
python pedagogical_assessment_pipeline_free.py \
    --provider gemini \
    --solver-model gemini-1.5-flash \
    --evaluator-model gemini-1.5-flash \
    --limit 5
```

**Pros**:
- ✅ Free tier: 15 requests/min
- ✅ Excellent quality
- ✅ Fast responses
- ✅ No local hardware needed

**Cons**:
- ⚠️ Rate limits (15 req/min)
- ⚠️ Requires API key (but free)

---

### Option 3: Groq (Free Tier - Fastest)

**Best for**: Speed, no local hardware needed

```bash
# Get free API key
# Visit: https://console.groq.com

# Set environment variable
export GROQ_API_KEY='your-key-here'

# Install package
pip install groq

# Run pipeline
python pedagogical_assessment_pipeline_free.py \
    --provider groq \
    --solver-model llama-3.1-70b-versatile \
    --evaluator-model llama-3.1-70b-versatile \
    --limit 5
```

**Pros**:
- ✅ Free tier available
- ✅ Extremely fast (fastest of all options)
- ✅ Good quality (Llama 3.1 70B)
- ✅ No local hardware needed

**Cons**:
- ⚠️ Rate limits
- ⚠️ Requires API key (but free)

---

### Option 4: HuggingFace Inference API

**Best for**: Access to many open models

```bash
# Get free API key
# Visit: https://huggingface.co/settings/tokens

# Set environment variable
export HUGGINGFACE_API_KEY='your-key-here'

# Run pipeline
python pedagogical_assessment_pipeline_free.py \
    --provider huggingface \
    --solver-model meta-llama/Meta-Llama-3-8B-Instruct \
    --evaluator-model meta-llama/Meta-Llama-3-8B-Instruct \
    --limit 5
```

**Pros**:
- ✅ Free tier available
- ✅ Many model choices
- ✅ No local hardware needed

**Cons**:
- ⚠️ Rate limits
- ⚠️ Can be slow (queue times)
- ⚠️ Requires API key (but free)

---

## 🚀 Quick Start

### Method 1: Ollama (Recommended)

```bash
# 1. Install Ollama from https://ollama.ai

# 2. Pull a model
ollama pull llama3.1:8b

# 3. Run pipeline
cd /DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach
python pedagogical_assessment_pipeline_free.py --limit 5
```

### Method 2: Gemini (Best Quality)

```bash
# 1. Get API key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY='your-key'

# 2. Install package
pip install google-generativeai

# 3. Run pipeline
python pedagogical_assessment_pipeline_free.py \
    --provider gemini \
    --solver-model gemini-1.5-flash \
    --limit 5
```

---

## 📊 Comparison

| Provider | Cost | Speed | Quality | Hardware | Rate Limits |
|----------|------|-------|---------|----------|-------------|
| **Ollama** | FREE | Medium | Good | Local GPU/CPU | None |
| **Gemini** | FREE | Fast | Excellent | None | 15/min |
| **Groq** | FREE | Very Fast | Good | None | Limited |
| **HuggingFace** | FREE | Slow | Varies | None | Limited |

---

## 💡 Recommendations

**For Testing (5-10 samples):**
- Use **Gemini** - fastest to set up, excellent quality

**For Small Datasets (< 100 samples):**
- Use **Gemini** or **Groq** - fast and free

**For Large Datasets (> 100 samples):**
- Use **Ollama** - no rate limits, runs 24/7

**Best Quality:**
- **Gemini 1.5 Pro** or **Ollama llama3.1:70b**

**Best Speed:**
- **Groq** (cloud) or **Ollama** (local with GPU)

---

## 📝 Complete Examples

### Example 1: Test with Ollama (Local)

```bash
# Install and setup Ollama
ollama pull llama3.1:8b

# Test with 5 samples
python pedagogical_assessment_pipeline_free.py \
    --provider ollama \
    --solver-model llama3.1:8b \
    --evaluator-model llama3.1:8b \
    --limit 5 \
    --output-dir output_test
```

### Example 2: Process 50 with Gemini

```bash
export GEMINI_API_KEY='your-key'
pip install google-generativeai

python pedagogical_assessment_pipeline_free.py \
    --provider gemini \
    --solver-model gemini-1.5-flash \
    --evaluator-model gemini-1.5-flash \
    --limit 50 \
    --rate-limit-delay 4.5 \
    --output-dir output_gemini
```

### Example 3: Full Dataset with Ollama

```bash
# No rate limits with Ollama!
python pedagogical_assessment_pipeline_free.py \
    --provider ollama \
    --solver-model llama3.1:70b \
    --evaluator-model llama3.1:70b \
    --rate-limit-delay 0.1 \
    --output-dir output_full
```

---

## 🛠️ Troubleshooting

### Ollama: "Connection refused"

```bash
# Make sure Ollama is running
ollama list

# If not running, start it
ollama serve
```

### Ollama: "Model not found"

```bash
# Pull the model first
ollama pull llama3.1:8b

# Check available models
ollama list
```

### Gemini: "API key invalid"

```bash
# Get new key from https://makersuite.google.com/app/apikey
export GEMINI_API_KEY='your-new-key'

# Test
python -c "import google.generativeai as genai; genai.configure(api_key='$GEMINI_API_KEY'); print('OK')"
```

### Groq: Rate limit exceeded

```bash
# Increase delay
python pedagogical_assessment_pipeline_free.py \
    --provider groq \
    --rate-limit-delay 2.0
```

---

## 💰 Cost Comparison

| Provider | 1000 samples | Notes |
|----------|--------------|-------|
| **Ollama** | $0 | Electricity only |
| **Gemini Free** | $0 | Up to 15 req/min |
| **Groq Free** | $0 | Rate limits apply |
| **GPT-4 (paid)** | $250-300 | From original pipeline |

**Savings: $250-300 per 1000 samples!**

---

## 🎓 Recommended Workflow

```bash
# Day 1: Test with Gemini (fast setup)
export GEMINI_API_KEY='your-key'
pip install google-generativeai
python pedagogical_assessment_pipeline_free.py --provider gemini --limit 5

# Day 2: Install Ollama for unlimited processing
# Download from https://ollama.ai
ollama pull llama3.1:70b

# Day 3: Process full dataset with Ollama (no limits!)
python pedagogical_assessment_pipeline_free.py \
    --provider ollama \
    --solver-model llama3.1:70b \
    --evaluator-model llama3.1:70b \
    --rate-limit-delay 0.1
```

---

## 🔥 Pro Tips

1. **Start with Gemini** for quick testing (free, easy setup)
2. **Move to Ollama** for large-scale processing (no limits)
3. **Use Groq** for speed when you have rate limits
4. **Use 70B models** for best quality (if hardware allows)
5. **Monitor quality** - free models are good but not perfect

---

## ✅ Ready to Start!

Pick your option and run:

```bash
# Ollama (local, free, unlimited)
python pedagogical_assessment_pipeline_free.py --limit 5

# Gemini (cloud, free tier, excellent)
python pedagogical_assessment_pipeline_free.py --provider gemini --limit 5

# Groq (cloud, free tier, fastest)
python pedagogical_assessment_pipeline_free.py --provider groq --limit 5
```

**No more API costs! 🎉**
