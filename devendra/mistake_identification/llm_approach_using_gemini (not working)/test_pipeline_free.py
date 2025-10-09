#!/usr/bin/env python3
"""
Quick test script for FREE LLM Pipeline
Tests with 5 conversations to verify everything works
"""

import os
import sys

def check_ollama():
    """Check if Ollama is available."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return True, models
    except:
        pass
    return False, []

def main():
    print("=" * 80)
    print("QUICK TEST - FREE LLM PIPELINE")
    print("=" * 80)
    print("\nThis script will process 5 sample conversations using FREE LLMs.")
    print()
    
    # Check what's available
    print("Checking available FREE LLM options...")
    print()
    
    # Check Ollama
    ollama_available, ollama_models = check_ollama()
    if ollama_available:
        print("✓ Ollama is running!")
        print(f"  Available models: {', '.join(ollama_models[:3])}")
        provider = "ollama"
        model = ollama_models[0] if ollama_models else "llama3.1:8b"
    else:
        print("✗ Ollama not found")
        print("  Install from: https://ollama.ai")
        print("  Then: ollama pull llama3.1:8b")
    
    # Check Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("✓ Gemini API key found!")
        if not ollama_available:
            provider = "gemini"
            model = "gemini-1.5-flash"
    else:
        print("✗ Gemini API key not found")
        print("  Get free key: https://makersuite.google.com/app/apikey")
        print("  Then: export GEMINI_API_KEY='your-key'")
    
    # Check Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✓ Groq API key found!")
        if not ollama_available and not gemini_key:
            provider = "groq"
            model = "llama-3.1-70b-versatile"
    else:
        print("✗ Groq API key not found")
        print("  Get free key: https://console.groq.com")
        print("  Then: export GROQ_API_KEY='your-key'")
    
    print()
    
    # Decide what to use
    if not ollama_available and not gemini_key and not groq_key:
        print("❌ ERROR: No FREE LLM option available!")
        print()
        print("Please set up at least one option:")
        print("1. Install Ollama (recommended): https://ollama.ai")
        print("2. Get Gemini key: https://makersuite.google.com/app/apikey")
        print("3. Get Groq key: https://console.groq.com")
        sys.exit(1)
    
    print(f"Using: {provider} with model {model}")
    print()
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    # Run the pipeline
    from pedagogical_assessment_pipeline_free import FreePedagogicalAssessmentPipeline
    
    output_dir = "/DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach/test_output_free"
    
    pipeline = FreePedagogicalAssessmentPipeline(
        data_dir="/DATA/cs24resch11011/repos/pedagogical-assessment/data",
        output_dir=output_dir,
        solver_model=model,
        evaluator_model=model,
        provider=provider,
        api_key=gemini_key or groq_key
    )
    
    # Run with limit of 5
    pipeline.run_full_pipeline(
        solve_problems=True,
        evaluate_responses=True,
        create_splits=True,
        limit=5,
        rate_limit_delay=0.5 if ollama_available else 4.5
    )
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print(f"\nCheck the output directory: {output_dir}")
    print()
    print("Files generated:")
    print("  - trainset_with_answers.json")
    print("  - trainset_fully_evaluated.json")
    print("  - dataset_statistics.json")
    print("  - train_split.json")
    print("  - validation_split.json")
    print("  - test_split.json")
    print()
    print("💰 Total cost: $0 (FREE!)")
    print()

if __name__ == "__main__":
    main()
