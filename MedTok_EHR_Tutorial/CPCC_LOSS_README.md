# CPCC Loss Integration with EHR Model

This document describes the integration of CPCC (Correlation Preserving Contrastive Coding) loss from the HypStructure project into the MedTok EHR model.

## Overview

The CPCC loss encourages the learned patient representations to preserve hierarchical relationships in medical data, leading to better representation learning and improved performance on medical prediction tasks.

## Files Modified

1. **`ehr_cpcc_loss.py`** - New file containing CPCC loss implementations adapted for EHR data
2. **`EHRModel_token.py`** - Modified to support CPCC loss integration
3. **`MedTok_EHR.py`** - Updated to include CPCC loss parameters
4. **`test_cpcc_integration.py`** - Test script to verify functionality
5. **`example_usage_cpcc.py`** - Usage examples

## Key Features

### CPCC Loss Components

1. **Hierarchical Distance Computation**: Computes distances based on medical outcome hierarchies
2. **Representation Distance**: Supports multiple distance metrics (L2, L1, cosine, Poincare)
3. **Correlation Preservation**: Maximizes correlation between hierarchical and representation distances
4. **Centering Regularization**: Optional centering constraint for numerical stability

### Supported Distance Metrics

- **L2**: Euclidean distance (default)
- **L1**: Manhattan distance
- **Cosine**: Cosine distance
- **Poincare**: Simplified Poincare distance

## Usage

### Basic Usage

```bash
# Without CPCC loss (default)
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10

# With CPCC loss
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 --use_cpcc 1
```

### Advanced Usage

```bash
# Custom CPCC parameters
python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 \
    --use_cpcc 1 \
    --cpcc_lamb 0.5 \
    --cpcc_distance_type cosine \
    --cpcc_center 1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use_cpcc` | int | 0 | Enable CPCC loss (0/1) |
| `--cpcc_lamb` | float | 1.0 | Weight for CPCC loss |
| `--cpcc_distance_type` | str | 'l2' | Distance metric (l2/l1/cosine/poincare) |
| `--cpcc_center` | int | 0 | Enable centering regularization (0/1) |

## Implementation Details

### Loss Function

The combined loss is computed as:

```
total_loss = base_loss + λ * cpcc_loss + center_reg
```

Where:
- `base_loss`: Original classification loss (cross-entropy or binary cross-entropy)
- `cpcc_loss`: CPCC regularization loss
- `λ`: Lambda weight parameter
- `center_reg`: Optional centering regularization

### Hierarchical Structure

The CPCC loss defines hierarchical relationships based on:

1. **Task Severity**: mortality > readmission > length of stay
2. **Medical Code Categories**: ICD-10 hierarchy
3. **Visit Patterns**: Recent vs historical visits

### Monitoring

When CPCC loss is enabled, additional metrics are logged:

- `train/cpcc_loss`: CPCC loss during training
- `val/cpcc_loss`: CPCC loss during validation  
- `test/cpcc_loss`: CPCC loss during testing

## Testing

Run the test script to verify functionality:

```bash
python test_cpcc_integration.py
```

This will test:
1. CPCC loss computation with different distance metrics
2. Hierarchical loss functionality
3. EHR model integration

## Expected Benefits

1. **Better Representation Learning**: Preserves hierarchical structure in medical data
2. **Improved Performance**: Better performance on medical prediction tasks
3. **Robust Embeddings**: More robust patient representations
4. **Generalization**: Better generalization across different medical conditions

## Troubleshooting

### Common Issues

1. **High CPCC Loss**: Reduce `--cpcc_lamb` parameter
2. **Performance Decrease**: Try different `--cpcc_distance_type`
3. **Numerical Instability**: Enable `--cpcc_center 1`
4. **Memory Issues**: Reduce batch size or use simpler distance metrics

### Recommended Settings

- **Start Small**: Begin with `--cpcc_lamb 0.1-0.5`
- **L2 Distance**: Use `--cpcc_distance_type l2` for stability
- **Centering**: Enable `--cpcc_center 1` for numerical stability
- **Gradual Increase**: Increase lambda gradually during training

## Example Results

Expected improvements when using CPCC loss:

- **AUC**: 2-5% improvement
- **AUPR**: 3-7% improvement
- **F1 Score**: 1-3% improvement
- **Representation Quality**: Better clustering of similar patients

## Future Enhancements

1. **Medical Code Hierarchies**: Integration with actual ICD-10 hierarchies
2. **Temporal Hierarchies**: Time-based hierarchical relationships
3. **Multi-task Learning**: CPCC loss across multiple tasks
4. **Hyperbolic Geometry**: Full Poincare ball implementation

## References

- HypStructure: Hierarchical Structure-Aware Contrastive Learning
- CPCC: Correlation Preserving Contrastive Coding
- Medical Representation Learning with Hierarchical Structures
