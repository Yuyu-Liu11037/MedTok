#!/usr/bin/env python3
"""
Example usage of CPCC loss with EHR model.
This script shows how to use the new CPCC loss functionality.
"""

import argparse
import os

def main():
    """Example usage of CPCC loss with EHR model"""
    
    print("=" * 60)
    print("Example: Using CPCC Loss with EHR Model")
    print("=" * 60)
    
    print("\n1. Basic usage with CPCC loss disabled (default):")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10")
    
    print("\n2. Using CPCC loss with default parameters:")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 \\")
    print("    --use_cpcc 1")
    
    print("\n3. Using CPCC loss with custom parameters:")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 \\")
    print("    --use_cpcc 1 --cpcc_lamb 0.5 --cpcc_distance_type cosine --cpcc_center 1")
    
    print("\n4. Using CPCC loss with different distance metrics:")
    print("# L2 distance (default)")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task mortality --epochs 10 \\")
    print("    --use_cpcc 1 --cpcc_distance_type l2")
    
    print("\n# L1 distance")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task mortality --epochs 10 \\")
    print("    --use_cpcc 1 --cpcc_distance_type l1")
    
    print("\n# Cosine distance")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task mortality --epochs 10 \\")
    print("    --use_cpcc 1 --cpcc_distance_type cosine")
    
    print("\n# Poincare distance")
    print("python MedTok_EHR.py --dataset MIMIC_IV --task mortality --epochs 10 \\")
    print("    --use_cpcc 1 --cpcc_distance_type poincare")
    
    print("\n5. Hyperparameter tuning for CPCC loss:")
    print("# Different lambda values")
    for lamb in [0.1, 0.5, 1.0, 2.0]:
        print(f"python MedTok_EHR.py --dataset MIMIC_IV --task readmission --epochs 10 \\")
        print(f"    --use_cpcc 1 --cpcc_lamb {lamb}")
    
    print("\n6. Testing with different tasks:")
    tasks = ['mortality', 'readmission', 'lenofstay', 'drugrec', 'phenotype']
    for task in tasks:
        print(f"python MedTok_EHR.py --dataset MIMIC_IV --task {task} --epochs 10 \\")
        print(f"    --use_cpcc 1 --cpcc_lamb 1.0")
    
    print("\n" + "=" * 60)
    print("CPCC Loss Parameters:")
    print("=" * 60)
    print("--use_cpcc: Enable/disable CPCC loss (0/1, default: 0)")
    print("--cpcc_lamb: Weight for CPCC loss (float, default: 1.0)")
    print("--cpcc_distance_type: Distance metric (l2/l1/cosine/poincare, default: l2)")
    print("--cpcc_center: Enable centering regularization (0/1, default: 0)")
    
    print("\n" + "=" * 60)
    print("Expected Benefits:")
    print("=" * 60)
    print("1. Better representation learning through hierarchical structure preservation")
    print("2. Improved performance on medical prediction tasks")
    print("3. More robust patient embeddings")
    print("4. Better generalization across different medical conditions")
    
    print("\n" + "=" * 60)
    print("Monitoring:")
    print("=" * 60)
    print("When CPCC loss is enabled, you'll see additional metrics in the logs:")
    print("- train/cpcc_loss: CPCC loss value during training")
    print("- val/cpcc_loss: CPCC loss value during validation")
    print("- test/cpcc_loss: CPCC loss value during testing")
    
    print("\n" + "=" * 60)
    print("Troubleshooting:")
    print("=" * 60)
    print("1. If CPCC loss is too high, try reducing --cpcc_lamb")
    print("2. If model performance decreases, try different --cpcc_distance_type")
    print("3. For numerical stability, use --cpcc_center 1")
    print("4. Start with small --cpcc_lamb values (0.1-0.5) and increase gradually")

if __name__ == "__main__":
    main()
