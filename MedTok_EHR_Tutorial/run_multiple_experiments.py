#!/usr/bin/env python3
"""
Script to run multiple experiments with different configurations.
This script demonstrates how to run experiments with and without CPCC loss.
"""

import subprocess
import os
import time
from datetime import datetime

def run_experiment(config, experiment_name):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = ["python", "MedTok_EHR.py"]
    for key, value in config.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
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
    """Run multiple experiments"""
    print("Multiple Experiments Runner")
    print("=" * 60)
    
    # Base configuration
    base_config = {
        'dataset': 'MIMIC_IV',
        'task': 'readmission',
        'epochs': 10,
        'batch_size': 256,
        'lr': 1e-3,
        'num_layers': 4,
        'hidden_dim': 256,
        'output_dim': 64,
        'num_heads': 4,
        'dropout': 0.5,
        'num_runs': 3,  # Reduced for demo
        'base_seed': 42,
        'use_partial_data': 1000  # Use partial data for faster demo
    }
    
    # Experiment configurations
    experiments = [
        {
            'name': 'Baseline (No CPCC)',
            'config': {**base_config, 'use_cpcc': 0}
        },
        {
            'name': 'CPCC L2 Distance',
            'config': {**base_config, 'use_cpcc': 1, 'cpcc_lamb': 1.0, 'cpcc_distance_type': 'l2'}
        },
        {
            'name': 'CPCC Cosine Distance',
            'config': {**base_config, 'use_cpcc': 1, 'cpcc_lamb': 1.0, 'cpcc_distance_type': 'cosine'}
        },
        {
            'name': 'CPCC L2 with Centering',
            'config': {**base_config, 'use_cpcc': 1, 'cpcc_lamb': 1.0, 'cpcc_distance_type': 'l2', 'cpcc_center': 1}
        },
        {
            'name': 'CPCC High Lambda',
            'config': {**base_config, 'use_cpcc': 1, 'cpcc_lamb': 2.0, 'cpcc_distance_type': 'l2'}
        }
    ]
    
    # Run experiments
    results = []
    start_time = datetime.now()
    
    for exp in experiments:
        success = run_experiment(exp['config'], exp['name'])
        results.append({
            'name': exp['name'],
            'success': success,
            'config': exp['config']
        })
        
        if success:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed")
    
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total duration: {total_duration}")
    print(f"Successful experiments: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    print("\nResults:")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {result['name']}")
    
    # Analyze results if any succeeded
    successful_experiments = [r for r in results if r['success']]
    if successful_experiments:
        print(f"\n{'='*60}")
        print("ANALYZING RESULTS")
        print(f"{'='*60}")
        
        # Look for result files
        result_files = []
        for exp in successful_experiments:
            # Expected result file name
            config = exp['config']
            expected_file = f"results_summary_{config['dataset']}_{config['task']}_Transformer_runs_{config['num_runs']}.json"
            if os.path.exists(expected_file):
                result_files.append(expected_file)
        
        if result_files:
            print(f"Found {len(result_files)} result files:")
            for file in result_files:
                print(f"  - {file}")
            
            # Run evaluation script
            if len(result_files) > 1:
                print("\nRunning comparison analysis...")
                cmd = ["python", "evaluate_results.py", "--results_file", result_files[0]]
                for file in result_files[1:]:
                    cmd.extend(["--compare_files", file])
                
                try:
                    subprocess.run(cmd, check=True)
                    print("Comparison analysis completed!")
                except subprocess.CalledProcessError as e:
                    print(f"Comparison analysis failed: {e}")
            else:
                print("\nRunning single experiment analysis...")
                cmd = ["python", "evaluate_results.py", "--results_file", result_files[0], "--generate_report"]
                try:
                    subprocess.run(cmd, check=True)
                    print("Analysis completed!")
                except subprocess.CalledProcessError as e:
                    print(f"Analysis failed: {e}")
        else:
            print("No result files found for analysis")

if __name__ == "__main__":
    main()
