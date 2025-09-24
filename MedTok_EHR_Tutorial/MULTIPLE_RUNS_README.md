# Multiple Runs for Stable Results

This document describes how to use the multiple runs functionality to get more stable and reliable results by averaging over multiple training runs.

## Overview

To address the randomness in training results, the system now supports running multiple experiments with different random seeds and averaging the results. This provides more reliable and statistically significant results.

## New Parameters

### Multiple Runs Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--num_runs` | int | 5 | Number of runs for averaging results |
| `--base_seed` | int | 42 | Base seed for random number generation |

### Random Seed Control

The system now includes comprehensive random seed control:

- **Python random**: `random.seed(seed)`
- **NumPy random**: `np.random.seed(seed)`
- **PyTorch random**: `torch.manual_seed(seed)`
- **CUDA random**: `torch.cuda.manual_seed_all(seed)`
- **Deterministic behavior**: `torch.backends.cudnn.deterministic = True`
- **Environment variables**: `PYTHONHASHSEED`

## Usage

### Basic Multiple Runs

```bash
# Run 5 experiments with different seeds and average results
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 --num_runs 5
```

### Custom Number of Runs

```bash
# Run 10 experiments for more stable results
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 --num_runs 10
```

### Custom Base Seed

```bash
# Use different base seed
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 --num_runs 5 --base_seed 123
```

### With CPCC Loss

```bash
# Multiple runs with CPCC loss
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 --num_runs 5 \
    --use_cpcc 1 --cpcc_lamb 1.0 --cpcc_distance_type l2
```

## Output Structure

### Directory Structure

Each run creates its own directory:
```
results_batch_size_256_Epochs_10_Layers_4_LR_0.001_MemorySize_512_run_1/
results_batch_size_256_Epochs_10_Layers_4_LR_0.001_MemorySize_512_run_2/
results_batch_size_256_Epochs_10_Layers_4_LR_0.001_MemorySize_512_run_3/
...
```

### Results Summary

A comprehensive results summary is saved as JSON:
```json
{
  "num_runs": 5,
  "base_seed": 42,
  "parameters": {...},
  "individual_results": [
    {"test/auc": 0.85, "test/aupr": 0.78, "test/f1": 0.72},
    {"test/auc": 0.87, "test/aupr": 0.80, "test/f1": 0.74},
    ...
  ],
  "averaged_results": {
    "test/auc": 0.856,
    "test/aupr": 0.792,
    "test/f1": 0.732
  },
  "std_results": {
    "test/auc": 0.012,
    "test/aupr": 0.015,
    "test/f1": 0.008
  }
}
```

## Console Output

### During Training

```
Starting 5 runs for averaging results...

============================================================
Starting Run 1/5
============================================================
Run 1/5: Random seed set to 42
...
Run 1 completed!

============================================================
Starting Run 2/5
============================================================
Run 2/5: Random seed set to 43
...
Run 2 completed!

...

============================================================
FINAL AVERAGED RESULTS
============================================================
Results averaged over 5 runs:
  test/auc: 0.8560 ± 0.0120
  test/aupr: 0.7920 ± 0.0150
  test/f1: 0.7320 ± 0.0080

Detailed results saved to: results_summary_MIMIC_IV_readmission_Transformer_runs_5.json
```

## Result Analysis

### Using the Evaluation Script

```bash
# Analyze results from a single experiment
python evaluate_results.py --results_file results_summary_MIMIC_IV_readmission_Transformer_runs_5.json

# Generate comprehensive report
python evaluate_results.py --results_file results_summary_MIMIC_IV_readmission_Transformer_runs_5.json --generate_report

# Compare multiple experiments
python evaluate_results.py --results_file results_summary_MIMIC_IV_readmission_Transformer_runs_5.json \
    --compare_files results_summary_MIMIC_IV_mortality_Transformer_runs_5.json
```

### Evaluation Output

The evaluation script provides:

1. **Statistical Analysis**: Mean, standard deviation, coefficient of variation
2. **Visualizations**: Bar plots, individual run comparisons, box plots
3. **Comparison Tables**: Side-by-side comparison of experiments
4. **Text Reports**: Comprehensive evaluation reports

## Automated Experiment Runner

### Run Multiple Configurations

```bash
# Run predefined experiments comparing different configurations
python run_multiple_experiments.py
```

This script automatically runs:
- Baseline (no CPCC)
- CPCC with L2 distance
- CPCC with cosine distance
- CPCC with centering regularization
- CPCC with high lambda

### Custom Experiment Runner

```python
# Example: Run custom experiments
experiments = [
    {
        'name': 'Baseline',
        'config': {'use_cpcc': 0, 'num_runs': 3}
    },
    {
        'name': 'CPCC L2',
        'config': {'use_cpcc': 1, 'cpcc_distance_type': 'l2', 'num_runs': 3}
    }
]

for exp in experiments:
    run_experiment(exp['config'], exp['name'])
```

## Statistical Significance

### Recommended Number of Runs

- **Development/Testing**: 3-5 runs
- **Final Results**: 5-10 runs
- **Publication**: 10+ runs

### Interpreting Results

- **Mean**: Average performance across runs
- **Standard Deviation**: Variability in results
- **Coefficient of Variation**: Relative variability (CV = std/mean)
- **Lower CV**: More stable results

### Example Interpretation

```
test/auc: 0.8560 ± 0.0120 (CV: 1.4%)
test/aupr: 0.7920 ± 0.0150 (CV: 1.9%)
test/f1: 0.7320 ± 0.0080 (CV: 1.1%)
```

- All metrics have low CV (< 2%), indicating stable results
- AUC is most stable (CV: 1.4%)
- F1 score is least variable (CV: 1.1%)

## Best Practices

### 1. Seed Management

- Use different base seeds for different experiments
- Document seed values for reproducibility
- Use consistent seed ranges across experiments

### 2. Result Analysis

- Always report mean ± standard deviation
- Include coefficient of variation for stability assessment
- Use statistical tests for significance when comparing methods

### 3. Resource Management

- Start with fewer runs during development
- Increase runs for final results
- Consider computational cost vs. statistical significance

### 4. Documentation

- Save all individual run results
- Document experimental parameters
- Include statistical analysis in reports

## Troubleshooting

### Common Issues

1. **High Variability**: Increase number of runs
2. **Memory Issues**: Reduce batch size or use fewer runs
3. **Long Training Time**: Use partial data for testing
4. **Inconsistent Results**: Check random seed implementation

### Performance Tips

1. **Parallel Runs**: Run experiments on different GPUs
2. **Reduced Data**: Use `--use_partial_data` for faster testing
3. **Early Stopping**: Use validation-based early stopping
4. **Checkpointing**: Save checkpoints for each run

## Example Workflow

### 1. Quick Test (3 runs)

```bash
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 5 --num_runs 3 --use_partial_data 1000
```

### 2. Full Experiment (5 runs)

```bash
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 20 --num_runs 5
```

### 3. Analyze Results

```bash
python evaluate_results.py --results_file results_summary_MIMIC_IV_readmission_Transformer_runs_5.json --generate_report
```

### 4. Compare Methods

```bash
python run_multiple_experiments.py
```

This workflow ensures reliable, statistically significant results for your EHR model experiments.
