#!/usr/bin/env python3
"""
Diagnose why results might be different between original and modified versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_step_by_step():
    """Test each step of the loss computation"""
    
    print("Step-by-Step Loss Computation Analysis")
    print("=" * 50)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    batch_size = 4
    num_classes = 2
    feature_dim = 64
    
    # Create test data
    prob_logits = torch.randn(batch_size, num_classes, requires_grad=True)
    y_true_one_hot = torch.randint(0, 2, (batch_size, num_classes)).float()
    patient_embedding = torch.randn(batch_size, feature_dim, requires_grad=True)
    sample_labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Input shapes:")
    print(f"  prob_logits: {prob_logits.shape}")
    print(f"  y_true_one_hot: {y_true_one_hot.shape}")
    print(f"  patient_embedding: {patient_embedding.shape}")
    print(f"  sample_labels: {sample_labels.shape}")
    
    # Test different tasks
    tasks = ['readmission', 'mortality', 'lenofstay']
    
    for task in tasks:
        print(f"\n{'='*20} Task: {task} {'='*20}")
        
        # Step 1: Compute base loss
        if task == 'lenofstay':
            loss = F.cross_entropy(prob_logits, y_true_one_hot)
        else:
            loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        
        print(f"Step 1 - Base loss: {loss.item():.8f}")
        print(f"  Loss requires_grad: {loss.requires_grad}")
        
        # Step 2: Check if CPCC would be added (simulate use_cpcc=False)
        use_cpcc = False
        if use_cpcc:
            print("Step 2 - CPCC would be added (but disabled)")
        else:
            print("Step 2 - CPCC disabled, no additional loss")
        
        # Step 3: Convert logits to probabilities
        if task == 'lenofstay' or task == 'readmission' or task == 'mortality':
            prob_logits_converted = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits_converted = torch.sigmoid(prob_logits)
        
        print(f"Step 3 - Converted logits shape: {prob_logits_converted.shape}")
        print(f"  Original logits sum: {prob_logits.sum().item():.8f}")
        print(f"  Converted logits sum: {prob_logits_converted.sum().item():.8f}")
        
        # Step 4: Compute metrics (simulate)
        print("Step 4 - Metrics computation (simulated)")
        
        # Check gradients
        if loss.requires_grad:
            loss.backward(retain_graph=True)
            print(f"  prob_logits grad norm: {prob_logits.grad.norm().item():.8f}")
            print(f"  patient_embedding grad norm: {patient_embedding.grad.norm().item():.8f}")
        
        print(f"Final loss: {loss.item():.8f}")

def test_cpcc_impact():
    """Test the impact of CPCC when enabled"""
    
    print(f"\n{'='*50}")
    print("CPCC Impact Analysis")
    print("=" * 50)
    
    try:
        from ehr_cpcc_loss import EHRCPCCLoss
        
        # Test parameters
        batch_size = 4
        feature_dim = 64
        num_classes = 2
        
        # Create test data
        prob_logits = torch.randn(batch_size, num_classes, requires_grad=True)
        y_true_one_hot = torch.randint(0, 2, (batch_size, num_classes)).float()
        patient_embedding = torch.randn(batch_size, feature_dim, requires_grad=True)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Base loss
        base_loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        print(f"Base loss: {base_loss.item():.8f}")
        
        # CPCC loss
        cpcc_loss = EHRCPCCLoss(distance_type='l2', lamb=1.0, center=False)
        cpcc_loss_value = cpcc_loss(patient_embedding, labels, 'readmission')
        print(f"CPCC loss: {cpcc_loss_value.item():.8f}")
        
        # Combined loss
        total_loss = base_loss + 1.0 * cpcc_loss_value
        print(f"Total loss: {total_loss.item():.8f}")
        
        # Check gradients
        total_loss.backward()
        print(f"prob_logits grad norm: {prob_logits.grad.norm().item():.8f}")
        print(f"patient_embedding grad norm: {patient_embedding.grad.norm().item():.8f}")
        
    except Exception as e:
        print(f"Error testing CPCC: {e}")

def main():
    """Run all diagnostic tests"""
    
    print("Diagnosing Loss Computation Differences")
    print("=" * 60)
    
    test_step_by_step()
    test_cpcc_impact()
    
    print(f"\n{'='*60}")
    print("Diagnosis Summary:")
    print("1. Check if the loss computation order is preserved")
    print("2. Verify that no additional computation happens when CPCC is disabled")
    print("3. Ensure gradients are computed correctly")
    print("4. Check for any side effects from imports or initialization")
    print("=" * 60)

if __name__ == "__main__":
    main()
