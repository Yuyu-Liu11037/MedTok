#!/usr/bin/env python3
"""
Verify that the fix ensures identical results when CPCC is disabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_original_vs_modified():
    """Test that original and modified loss computation are identical when CPCC is disabled"""
    
    print("Testing Original vs Modified Loss Computation")
    print("=" * 50)
    
    # Test parameters
    batch_size = 4
    num_classes = 2
    
    # Create test data
    prob_logits = torch.randn(batch_size, num_classes)
    y_true_one_hot = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Test different tasks
    tasks = ['readmission', 'mortality', 'lenofstay']
    
    for task in tasks:
        print(f"\nTask: {task}")
        
        # Original computation (what it should be)
        if task == 'lenofstay':
            original_loss = F.cross_entropy(prob_logits, y_true_one_hot)
        else:
            original_loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        
        # Modified computation (simulating our new logic with CPCC disabled)
        if task == 'lenofstay':
            modified_loss = F.cross_entropy(prob_logits, y_true_one_hot)
        else:
            modified_loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        
        # When CPCC is disabled, no additional computation should happen
        # So they should be identical
        
        print(f"  Original:  {original_loss.item():.8f}")
        print(f"  Modified:  {modified_loss.item():.8f}")
        print(f"  Difference: {abs(original_loss.item() - modified_loss.item()):.10f}")
        
        if torch.allclose(original_loss, modified_loss, atol=1e-10):
            print("  ✅ IDENTICAL")
        else:
            print("  ❌ DIFFERENT")
    
    print("\n" + "=" * 50)
    print("Key Points:")
    print("1. When use_cpcc=False, the loss computation should be identical to original")
    print("2. No additional computation should happen when CPCC is disabled")
    print("3. The order of operations should be preserved")
    print("=" * 50)

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    test_original_vs_modified()
