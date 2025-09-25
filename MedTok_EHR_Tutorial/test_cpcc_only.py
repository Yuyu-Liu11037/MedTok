#!/usr/bin/env python3
"""
Test script for CPCC-only loss experiments.
This script runs experiments with only CPCC loss (no base loss) to debug the loss function.
"""

import subprocess
import os
import time
from datetime import datetime

def run_cpcc_only_experiment(cpcc_lamb, experiment_name, epochs=5, num_runs=2):
    """Run a CPCC-only experiment with given lambda value"""
    print(f"\n{'='*60}")
    print(f"Starting CPCC-Only Experiment: {experiment_name}")
    print(f"CPCC Lambda: {cpcc_lamb}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        "python", "MedTok_EHR.py",
        "--dataset", "MIMIC_IV",
        "--task", "readmission",
        "--epochs", str(epochs),
        "--batch_size", "256",
        "--lr", "1e-3",
        "--num_layers", "4",
        "--hidden_dim", "256",
        "--output_dim", "64",
        "--num_heads", "4",
        "--dropout", "0.5",
        "--num_runs", str(num_runs),
        "--base_seed", "42",
        "--use_partial_data", "1000",  # Use partial data for faster testing
        "--use_cpcc", "1",
        "--cpcc_lamb", str(cpcc_lamb),
        "--cpcc_distance_type", "l2",
        "--cpcc_center", "0",
        "--cpcc_only", "1"  # Only use CPCC loss
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"Experiment '{experiment_name}' completed successfully!")
            print(f"Duration: {end_time - start_time:.2f} seconds")
            return True
        else:
            print(f"Experiment '{experiment_name}' failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Experiment '{experiment_name}' timed out!")
        return False
    except Exception as e:
        print(f"Experiment '{experiment_name}' failed with exception: {e}")
        return False

def main():
    """Run CPCC-only experiments with different lambda values"""
    print("CPCC-Only Loss Debug Experiments")
    print("=" * 60)
    
    # Test different CPCC lambda values
    experiments = [
        {"lambda": 0.1, "name": "CPCC-Only Lambda 0.1"},
        {"lambda": 1.0, "name": "CPCC-Only Lambda 1.0"},
        {"lambda": 5.0, "name": "CPCC-Only Lambda 5.0"},
        {"lambda": 10.0, "name": "CPCC-Only Lambda 10.0"},
    ]
    
    results = []
    start_time = datetime.now()
    
    for exp in experiments:
        success = run_cpcc_only_experiment(
            cpcc_lamb=exp["lambda"],
            experiment_name=exp["name"],
            epochs=3,  # Short epochs for debugging
            num_runs=2  # Few runs for debugging
        )
        results.append({
            'name': exp['name'],
            'lambda': exp['lambda'],
            'success': success
        })
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("CPCC-ONLY EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    print("\nResults:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['name']} (λ={result['lambda']})")
    
    # Analysis
    successful_experiments = [r for r in results if r['success']]
    if successful_experiments:
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print(f"{'='*60}")
        print("Check the output logs to see:")
        print("1. train/cpcc_loss values - should be different for different lambda values")
        print("2. train/base_loss values - should be 0.0 for all experiments")
        print("3. test/loss values - should be different for different lambda values")
        print("4. Other metrics (AUC, AUPR, F1) - may be similar or different")
        
        print("\nExpected behavior:")
        print("- Higher lambda values should lead to higher total loss")
        print("- CPCC loss should dominate the training")
        print("- Model should still learn some patterns even with only CPCC loss")

if __name__ == "__main__":
    main()
