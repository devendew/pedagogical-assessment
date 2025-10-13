#!/usr/bin/env python3
"""
Merge predictions from test_predictions_optimized.json with testset.json
This script matches conversation_id and model name, then adds the Mistake_Identification
prediction to each tutor response.
"""

import json
from pathlib import Path
from collections import defaultdict

# File paths
PREDICTIONS_FILE = "../../../data/test_predictions_optimized.json"
TESTSET_FILE = "../../../data/testset.json"
OUTPUT_FILE = "../../../data/testset_with_predictions.json"

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    print("="*80)
    print("MERGING PREDICTIONS WITH TEST SET")
    print("="*80)
    
    # Load files
    print("\n[1/4] Loading predictions...")
    predictions = load_json(PREDICTIONS_FILE)
    print(f"  Loaded {len(predictions)} predictions")
    
    print("\n[2/4] Loading test set...")
    testset = load_json(TESTSET_FILE)
    print(f"  Loaded {len(testset)} conversations")
    
    # Create lookup dictionary: (conversation_id, model) -> prediction
    print("\n[3/4] Creating prediction lookup...")
    pred_lookup = {}
    for pred in predictions:
        key = (pred['conversation_id'], pred['model'])
        pred_lookup[key] = {
            'Mistake_Identification': pred['Mistake_Identification'],
            'confidence': pred['confidence']
        }
    print(f"  Created lookup with {len(pred_lookup)} entries")
    
    # Merge predictions into testset
    print("\n[4/4] Merging predictions into test set...")
    matched_count = 0
    missing_count = 0
    
    for conversation in testset:
        conv_id = conversation['conversation_id']
        
        for model_name, tutor_data in conversation['tutor_responses'].items():
            key = (conv_id, model_name)
            
            if key in pred_lookup:
                # Add prediction to tutor response in annotation format
                tutor_data['annotation'] = {
                    'Mistake_Identification': pred_lookup[key]['Mistake_Identification']
                }
                # Optionally add confidence outside annotation
                # tutor_data['confidence'] = pred_lookup[key]['confidence']
                matched_count += 1
            else:
                print(f"  ⚠️  Missing prediction for: {conv_id} / {model_name}")
                missing_count += 1
    
    # Save merged data
    print(f"\n  Matched: {matched_count} predictions")
    print(f"  Missing: {missing_count} predictions")
    
    save_json(testset, OUTPUT_FILE)
    
    print(f"\n✅ Merged test set saved to: {OUTPUT_FILE}")
    print("="*80)
    
    # Show example
    print("\n📊 EXAMPLE OUTPUT:")
    print("-"*80)
    if testset and testset[0]['tutor_responses']:
        conv = testset[0]
        model_name = list(conv['tutor_responses'].keys())[0]
        tutor_resp = conv['tutor_responses'][model_name]
        
        print(f"Conversation ID: {conv['conversation_id']}")
        print(f"Model: {model_name}")
        print(f"Response: {tutor_resp['response'][:100]}...")
        if 'annotation' in tutor_resp:
            print(f"Annotation: {tutor_resp['annotation']}")
        print(f"Confidence: {tutor_resp.get('confidence', 'N/A')}")
    print("-"*80)

if __name__ == "__main__":
    main()
