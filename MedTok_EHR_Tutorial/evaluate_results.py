#!/usr/bin/env python3
"""
Evaluation script for analyzing multiple run results.
This script helps analyze and compare results from multiple training runs.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os

def load_results(results_file):
    """Load results from JSON file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_results(results_data):
    """Analyze and display results statistics"""
    print("=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    num_runs = results_data['num_runs']
    individual_results = results_data['individual_results']
    averaged_results = results_data['averaged_results']
    std_results = results_data['std_results']
    
    print(f"Number of runs: {num_runs}")
    print(f"Base seed: {results_data['base_seed']}")
    
    # Display parameters
    print("\nExperiment Parameters:")
    params = results_data['parameters']
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Display individual run results
    print(f"\nIndividual Run Results:")
    print("-" * 40)
    for i, result in enumerate(individual_results):
        print(f"Run {i+1}:")
        for key, value in result.items():
            if key.startswith('test/'):
                print(f"  {key}: {value:.4f}")
    
    # Display averaged results
    print(f"\nAveraged Results (Mean ± Std):")
    print("-" * 40)
    for key in sorted(averaged_results.keys()):
        if key.startswith('test/'):
            mean_val = averaged_results[key]
            std_val = std_results[key]
            print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Calculate coefficient of variation
    print(f"\nCoefficient of Variation (CV = Std/Mean):")
    print("-" * 40)
    for key in sorted(averaged_results.keys()):
        if key.startswith('test/'):
            mean_val = averaged_results[key]
            std_val = std_results[key]
            if mean_val != 0:
                cv = std_val / abs(mean_val)
                print(f"  {key}: {cv:.4f} ({cv*100:.2f}%)")
    
    return averaged_results, std_results

def create_visualizations(results_data, output_dir="plots"):
    """Create visualization plots for the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    individual_results = results_data['individual_results']
    averaged_results = results_data['averaged_results']
    std_results = results_data['std_results']
    
    # Extract test metrics
    test_metrics = {}
    for key in averaged_results.keys():
        if key.startswith('test/'):
            test_metrics[key] = {
                'mean': averaged_results[key],
                'std': std_results[key]
            }
    
    if not test_metrics:
        print("No test metrics found for visualization")
        return
    
    # Create bar plot with error bars
    plt.figure(figsize=(12, 8))
    metrics = list(test_metrics.keys())
    means = [test_metrics[m]['mean'] for m in metrics]
    stds = [test_metrics[m]['std'] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Test Results (Mean ± Std)')
    plt.xticks(x_pos, [m.replace('test/', '') for m in metrics], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/test_metrics_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual run comparison
    plt.figure(figsize=(15, 10))
    num_runs = len(individual_results)
    num_metrics = len(test_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, _) in enumerate(test_metrics.items()):
        if i >= 4:  # Limit to 4 subplots
            break
            
        ax = axes[i]
        values = [result.get(metric, 0) for result in individual_results]
        runs = list(range(1, num_runs + 1))
        
        ax.plot(runs, values, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Run Number')
        ax.set_ylabel(metric.replace('test/', ''))
        ax.set_title(f'{metric.replace("test/", "")} Across Runs')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(runs)
    
    # Hide unused subplots
    for i in range(len(test_metrics), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/individual_runs_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create box plot for all metrics
    plt.figure(figsize=(12, 8))
    data_for_box = []
    labels_for_box = []
    
    for metric in test_metrics.keys():
        values = [result.get(metric, 0) for result in individual_results]
        data_for_box.append(values)
        labels_for_box.append(metric.replace('test/', ''))
    
    plt.boxplot(data_for_box, labels=labels_for_box)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Distribution of Test Results Across Runs')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/box_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def compare_experiments(results_files):
    """Compare results from multiple experiments"""
    print("=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    
    comparison_data = []
    
    for results_file in results_files:
        if os.path.exists(results_file):
            results = load_results(results_file)
            exp_name = Path(results_file).stem
            
            # Extract key metrics
            avg_results = results['averaged_results']
            key_metrics = ['test/auc', 'test/aupr', 'test/f1']
            
            exp_data = {'experiment': exp_name}
            for metric in key_metrics:
                if metric in avg_results:
                    exp_data[metric] = avg_results[metric]
            
            comparison_data.append(exp_data)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        print("\nComparison Table:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        metrics = ['test/auc', 'test/aupr', 'test/f1']
        x_pos = np.arange(len(comparison_data))
        
        for i, metric in enumerate(metrics):
            values = [exp_data.get(metric, 0) for exp_data in comparison_data]
            plt.plot(x_pos, values, 'o-', label=metric.replace('test/', ''), linewidth=2, markersize=8)
        
        plt.xlabel('Experiments')
        plt.ylabel('Values')
        plt.title('Experiment Comparison')
        plt.xticks(x_pos, [exp_data['experiment'] for exp_data in comparison_data], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('experiment_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nComparison plot saved as 'experiment_comparison.png'")

def generate_report(results_file, output_file="evaluation_report.txt"):
    """Generate a comprehensive evaluation report"""
    results_data = load_results(results_file)
    
    with open(output_file, 'w') as f:
        f.write("EHR Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Experiment Configuration:\n")
        f.write(f"  Number of runs: {results_data['num_runs']}\n")
        f.write(f"  Base seed: {results_data['base_seed']}\n")
        f.write(f"  Parameters: {json.dumps(results_data['parameters'], indent=2)}\n\n")
        
        f.write("Individual Run Results:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(results_data['individual_results']):
            f.write(f"Run {i+1}:\n")
            for key, value in result.items():
                if key.startswith('test/'):
                    f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
        
        f.write("Averaged Results:\n")
        f.write("-" * 20 + "\n")
        avg_results = results_data['averaged_results']
        std_results = results_data['std_results']
        
        for key in sorted(avg_results.keys()):
            if key.startswith('test/'):
                mean_val = avg_results[key]
                std_val = std_results[key]
                f.write(f"  {key}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write("\nStatistical Analysis:\n")
        f.write("-" * 20 + "\n")
        for key in sorted(avg_results.keys()):
            if key.startswith('test/'):
                mean_val = avg_results[key]
                std_val = std_results[key]
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    f.write(f"  {key} CV: {cv:.4f} ({cv*100:.2f}%)\n")
    
    print(f"Evaluation report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple run results')
    parser.add_argument('--results_file', type=str, required=True, help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--compare_files', nargs='+', help='Additional result files to compare')
    parser.add_argument('--generate_report', action='store_true', help='Generate text report')
    
    args = parser.parse_args()
    
    # Load and analyze results
    results_data = load_results(args.results_file)
    analyze_results(results_data)
    
    # Create visualizations
    create_visualizations(results_data, args.output_dir)
    
    # Compare experiments if provided
    if args.compare_files:
        compare_experiments([args.results_file] + args.compare_files)
    
    # Generate report if requested
    if args.generate_report:
        generate_report(args.results_file)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
