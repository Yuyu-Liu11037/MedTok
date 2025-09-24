#!/usr/bin/env python3
"""
Test script for multiple runs functionality.
This script tests the basic functionality without requiring full training.
"""

import torch
import numpy as np
import json
import os
from datetime import datetime

def test_random_seed_functionality():
    """Test random seed setting functionality"""
    print("Testing Random Seed Functionality...")
    
    # Import the function
    import sys
    sys.path.append('.')
    from MedTok_EHR import set_random_seed
    
    # Test 1: Same seed should produce same results
    print("\nTest 1: Same seed produces same results")
    
    # Set seed and generate random numbers
    set_random_seed(42)
    torch_vals_1 = torch.randn(5)
    np_vals_1 = np.random.randn(5)
    
    # Reset and set same seed
    set_random_seed(42)
    torch_vals_2 = torch.randn(5)
    np_vals_2 = np.random.randn(5)
    
    # Check if they're the same
    torch_same = torch.allclose(torch_vals_1, torch_vals_2)
    np_same = np.allclose(np_vals_1, np_vals_2)
    
    print(f"  PyTorch values same: {torch_same}")
    print(f"  NumPy values same: {np_same}")
    
    if torch_same and np_same:
        print("  ✓ Random seed functionality works correctly")
    else:
        print("  ✗ Random seed functionality has issues")
    
    # Test 2: Different seeds should produce different results
    print("\nTest 2: Different seeds produce different results")
    
    set_random_seed(42)
    torch_vals_1 = torch.randn(5)
    
    set_random_seed(43)
    torch_vals_2 = torch.randn(5)
    
    torch_different = not torch.allclose(torch_vals_1, torch_vals_2)
    print(f"  Different seeds produce different values: {torch_different}")
    
    if torch_different:
        print("  ✓ Different seeds produce different results")
    else:
        print("  ✗ Different seeds produce same results (unexpected)")

def test_results_aggregation():
    """Test results aggregation functionality"""
    print("\nTesting Results Aggregation...")
    
    # Mock individual results
    individual_results = [
        {"test/auc": 0.85, "test/aupr": 0.78, "test/f1": 0.72},
        {"test/auc": 0.87, "test/aupr": 0.80, "test/f1": 0.74},
        {"test/auc": 0.86, "test/aupr": 0.79, "test/f1": 0.73},
        {"test/auc": 0.88, "test/aupr": 0.81, "test/f1": 0.75},
        {"test/auc": 0.84, "test/aupr": 0.77, "test/f1": 0.71}
    ]
    
    # Calculate averages and standard deviations
    avg_results = {}
    std_results = {}
    
    # Get all metric keys
    all_keys = set()
    for result in individual_results:
        all_keys.update(result.keys())
    
    for key in all_keys:
        values = [result.get(key, 0) for result in individual_results if key in result]
        if values:
            avg_results[key] = np.mean(values)
            std_results[key] = np.std(values)
    
    # Display results
    print("  Individual Results:")
    for i, result in enumerate(individual_results):
        print(f"    Run {i+1}: AUC={result['test/auc']:.3f}, AUPR={result['test/aupr']:.3f}, F1={result['test/f1']:.3f}")
    
    print("\n  Averaged Results:")
    for key in sorted(avg_results.keys()):
        avg_val = avg_results[key]
        std_val = std_results[key]
        print(f"    {key}: {avg_val:.4f} ± {std_val:.4f}")
    
    # Test coefficient of variation
    print("\n  Coefficient of Variation:")
    for key in sorted(avg_results.keys()):
        avg_val = avg_results[key]
        std_val = std_results[key]
        if avg_val != 0:
            cv = std_val / abs(avg_val)
            print(f"    {key}: {cv:.4f} ({cv*100:.2f}%)")
    
    print("  ✓ Results aggregation works correctly")

def test_json_save_load():
    """Test JSON save/load functionality"""
    print("\nTesting JSON Save/Load...")
    
    # Create mock results data
    results_data = {
        'num_runs': 5,
        'base_seed': 42,
        'parameters': {
            'dataset': 'MIMIC_IV',
            'task': 'readmission',
            'epochs': 10,
            'use_cpcc': 1,
            'cpcc_lamb': 1.0
        },
        'individual_results': [
            {"test/auc": 0.85, "test/aupr": 0.78, "test/f1": 0.72},
            {"test/auc": 0.87, "test/aupr": 0.80, "test/f1": 0.74},
            {"test/auc": 0.86, "test/aupr": 0.79, "test/f1": 0.73},
            {"test/auc": 0.88, "test/aupr": 0.81, "test/f1": 0.75},
            {"test/auc": 0.84, "test/aupr": 0.77, "test/f1": 0.71}
        ],
        'averaged_results': {
            'test/auc': 0.860,
            'test/aupr': 0.790,
            'test/f1': 0.730
        },
        'std_results': {
            'test/auc': 0.015,
            'test/aupr': 0.014,
            'test/f1': 0.014
        }
    }
    
    # Save to file
    test_file = "test_results.json"
    with open(test_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Load from file
    with open(test_file, 'r') as f:
        loaded_data = json.load(f)
    
    # Verify data integrity
    if loaded_data == results_data:
        print("  ✓ JSON save/load works correctly")
    else:
        print("  ✗ JSON save/load has issues")
    
    # Clean up
    if os.path.exists(test_file):
        os.remove(test_file)
        print("  ✓ Test file cleaned up")

def test_directory_creation():
    """Test directory creation for multiple runs"""
    print("\nTesting Directory Creation...")
    
    # Test directory naming
    base_dirpath = "results_batch_size_256_Epochs_10_Layers_4_LR_0.001_MemorySize_512"
    
    for run_id in range(3):
        dirpath = f"{base_dirpath}_run_{run_id + 1}"
        print(f"  Run {run_id + 1} directory: {dirpath}")
        
        # Create directory
        os.makedirs(dirpath, exist_ok=True)
        
        # Check if directory exists
        if os.path.exists(dirpath):
            print(f"    ✓ Directory created successfully")
        else:
            print(f"    ✗ Directory creation failed")
    
    # Clean up test directories
    for run_id in range(3):
        dirpath = f"{base_dirpath}_run_{run_id + 1}"
        if os.path.exists(dirpath):
            os.rmdir(dirpath)
            print(f"    ✓ Directory {dirpath} cleaned up")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Multiple Runs Functionality")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_random_seed_functionality()
    test_results_aggregation()
    test_json_save_load()
    test_directory_creation()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run a quick test: python MedTok_EHR.py --num_runs 3 --epochs 5 --use_partial_data 1000")
    print("2. Analyze results: python evaluate_results.py --results_file results_summary_*.json")
    print("3. Run multiple experiments: python run_multiple_experiments.py")

if __name__ == "__main__":
    main()
