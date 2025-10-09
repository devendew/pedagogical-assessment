#!/usr/bin/env python3
"""
Quick test script for the Pedagogical Assessment Pipeline
Tests with a small sample (5 conversations) to verify everything works
"""

import os
import sys
from pedagogical_assessment_pipeline import PedagogicalAssessmentPipeline

def main():
    print("=" * 80)
    print("QUICK TEST - PEDAGOGICAL ASSESSMENT PIPELINE")
    print("=" * 80)
    print("\nThis script will process 5 sample conversations to test the pipeline.")
    print("Make sure you have set your API key:")
    print("  export OPENAI_API_KEY='your-key-here'")
    print("  OR")
    print("  export ANTHROPIC_API_KEY='your-key-here'")
    print()
    
    # Check for API keys
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None
    
    if not has_openai and not has_anthropic:
        print("ERROR: No API key found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)
    
    # Choose provider
    provider = "openai" if has_openai else "anthropic"
    model = "gpt-4o" if provider == "openai" else "claude-3-5-sonnet-20241022"
    
    print(f"Using provider: {provider}")
    print(f"Using model: {model}")
    print()
    
    input("Press Enter to continue or Ctrl+C to cancel...")
    
    # Initialize pipeline
    pipeline = PedagogicalAssessmentPipeline(
        data_dir="/DATA/cs24resch11011/repos/pedagogical-assessment/data",
        output_dir="/DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach/test_output",
        solver_model=model,
        evaluator_model=model,
        provider=provider
    )
    
    # Run with limit of 5
    pipeline.run_full_pipeline(
        solve_problems=True,
        evaluate_responses=True,
        create_splits=True,
        limit=5,
        rate_limit_delay=1.0
    )
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print("\nCheck the output directory for results:")
    print("  /DATA/cs24resch11011/repos/pedagogical-assessment/devendra/mistake_identification/llm_approach/test_output")
    print()
    print("Files generated:")
    print("  - trainset_with_answers.json")
    print("  - trainset_fully_evaluated.json")
    print("  - dataset_statistics.json")
    print("  - train_split.json")
    print("  - validation_split.json")
    print("  - test_split.json")
    print()

if __name__ == "__main__":
    main()
