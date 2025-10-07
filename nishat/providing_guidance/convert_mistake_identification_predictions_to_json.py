#!/usr/bin/env python3
"""
Convert CSV predictions back to the original JSON format.
"""

import json
import csv
from collections import defaultdict, Counter

def convert_predictions_to_json(csv_file, json_template_file, output_file):
    """
    Convert CSV predictions back to JSON format matching the input structure.
    
    Args:
        csv_file: Path to the predictions CSV file
        json_template_file: Path to the original JSON file (for structure reference)
        output_file: Path to save the output JSON with predictions
    """
    # Load predictions CSV
    print(f"Loading predictions from: {csv_file}")
    predictions_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions_data.append(row)
    print(f"  Loaded {len(predictions_data)} predictions")
    
    # Load original JSON template
    print(f"\nLoading original data structure from: {json_template_file}")
    with open(json_template_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    print(f"  Loaded {len(original_data)} conversations")
    
    # Group predictions by conversation_id and model
    print("\nGrouping predictions by conversation and model...")
    predictions_grouped = defaultdict(dict)
    
    for row in predictions_data:
        conv_id = row['conversation_id']
        model = row['model']
        predicted_label = row['predicted_label']
        
        predictions_grouped[conv_id][model] = {
            'predicted_label': predicted_label,
            'prob_yes': float(row['prob_yes']),
            'prob_to_some_extent': float(row['prob_to_some_extent']),
            'prob_no': float(row['prob_no'])
        }
    
    # Create output data with predictions
    print("\nAdding predictions to original structure...")
    output_data = []
    
    for conversation in original_data:
        conv_id = conversation['conversation_id']
        conv_history = conversation['conversation_history']
        
        # Create new conversation entry
        new_conversation = {
            'conversation_id': conv_id,
            'conversation_history': conv_history,
            'tutor_responses': {}
        }
        
        # Add each model's response with prediction
        for model_name, response_data in conversation['tutor_responses'].items():
            new_conversation['tutor_responses'][model_name] = {
                'response': response_data['response']
            }
            
            # Add prediction if available
            if conv_id in predictions_grouped and model_name in predictions_grouped[conv_id]:
                pred = predictions_grouped[conv_id][model_name]
                new_conversation['tutor_responses'][model_name]['annotation'] = {
                    'Mistake_Identification': pred['predicted_label']
                }
                # new_conversation['tutor_responses'][model_name]['prediction_confidence'] = {
                #     'prob_yes': pred['prob_yes'],
                #     'prob_to_some_extent': pred['prob_to_some_extent'],
                #     'prob_no': pred['prob_no']
                # }
            else:
                print(f"  Warning: No prediction found for {conv_id} - {model_name}")
        
        output_data.append(new_conversation)
    
    # Save output JSON
    print(f"\nSaving predictions to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully saved {len(output_data)} conversations with predictions!")
    
    # Print statistics
    total_predictions = sum(len(conv['tutor_responses']) for conv in output_data)
    print(f"\nStatistics:")
    print(f"  Total conversations: {len(output_data)}")
    print(f"  Total predictions: {total_predictions}")
    
    # Count prediction distribution
    all_predictions = []
    for conv in output_data:
        for model, data in conv['tutor_responses'].items():
            if 'annotation' in data:
                all_predictions.append(data['annotation']['Mistake_Identification'])
    
    if all_predictions:
        pred_counts = Counter(all_predictions)
        print(f"\nPrediction Distribution:")
        for label, count in sorted(pred_counts.items()):
            percentage = (count / len(all_predictions)) * 100
            print(f"  {label:20s}: {count:4d} ({percentage:5.1f}%)")

if __name__ == "__main__":
    import sys
    
    # Convert dev predictions
    print("="*80)
    print("CONVERTING DEV PREDICTIONS TO JSON FORMAT")
    print("="*80)
    convert_predictions_to_json(
        csv_file='dev_predictions.csv',
        json_template_file='../../data/dev_testset.json',
        output_file='dev_predictions.json'
    )
    
    print("\n" + "="*80)
    print("CONVERTING TEST PREDICTIONS TO JSON FORMAT")
    print("="*80)
    convert_predictions_to_json(
        csv_file='test_predictions.csv',
        json_template_file='../../data/testset.json',
        output_file='test_predictions.json'
    )
    
    print("\n" + "="*80)
    print("✓ ALL CONVERSIONS COMPLETE!")
    print("="*80)
    print("\nOutput files created:")
    print("  - dev_predictions.json")
    print("  - test_predictions.json")
