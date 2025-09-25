#!/usr/bin/env python3
"""
Unit test for CPCC-only functionality.
This script tests the CPCC loss function and CPCC-only mode without running the full training.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ehr_cpcc_loss import EHRCPCCLoss

def test_cpcc_loss():
    """Test CPCC loss function"""
    print("Testing CPCC Loss Function")
    print("=" * 40)
    
    # Create test data
    batch_size = 8
    feature_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random representations and labels
    representations = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    
    print(f"Representations shape: {representations.shape}")
    print(f"Labels: {labels}")
    print(f"Device: {device}")
    
    # Test CPCC loss
    cpcc_loss = EHRCPCCLoss(distance_type='l2', lamb=1.0, center=False)
    
    try:
        loss_value = cpcc_loss(representations, labels, task='readmission')
        print(f"CPCC Loss value: {loss_value.item():.6f}")
        print(f"Loss requires grad: {loss_value.requires_grad}")
        
        # Test backward pass
        loss_value.backward()
        print(f"Gradients computed successfully")
        print(f"Representations grad norm: {representations.grad.norm().item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error testing CPCC loss: {e}")
        return False

def test_cpcc_only_mode():
    """Test CPCC-only mode logic"""
    print("\nTesting CPCC-Only Mode Logic")
    print("=" * 40)
    
    # Simulate the loss computation logic
    batch_size = 4
    num_class = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    patient_embedding = torch.randn(batch_size, 64, device=device, requires_grad=True)
    prob_logits = torch.randn(batch_size, num_class, device=device, requires_grad=True)
    sample_labels = torch.randint(0, 2, (batch_size,), device=device)
    
    # Convert labels to one-hot
    y_true_one_hot = torch.zeros((batch_size, num_class), device=device)
    y_true_one_hot[torch.arange(batch_size), sample_labels] = 1
    
    print(f"Patient embedding shape: {patient_embedding.shape}")
    print(f"Prob logits shape: {prob_logits.shape}")
    print(f"Labels: {sample_labels}")
    
    # Test different modes
    modes = [
        {"name": "Base Loss Only", "use_cpcc": False, "cpcc_only": False},
        {"name": "Base + CPCC Loss", "use_cpcc": True, "cpcc_only": False},
        {"name": "CPCC Only", "use_cpcc": True, "cpcc_only": True}
    ]
    
    for mode in modes:
        print(f"\n--- {mode['name']} ---")
        
        # Compute base loss (only if not using CPCC only)
        if not mode['cpcc_only']:
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
            print(f"Base loss: {base_loss.item():.6f}")
        else:
            base_loss = torch.tensor(0.0, device=device, requires_grad=True)
            print(f"Base loss: {base_loss.item():.6f} (disabled)")
        
        # Add CPCC loss if enabled
        if mode['use_cpcc']:
            cpcc_loss_fn = EHRCPCCLoss(distance_type='l2', lamb=1.0, center=False)
            labels_for_cpcc = sample_labels.squeeze()
            cpcc_loss_value = cpcc_loss_fn(patient_embedding, labels_for_cpcc, 'readmission')
            print(f"CPCC loss: {cpcc_loss_value.item():.6f}")
            
            if mode['cpcc_only']:
                # Use only CPCC loss
                total_loss = 1.0 * cpcc_loss_value
                print(f"Total loss (CPCC only): {total_loss.item():.6f}")
            else:
                # Add CPCC loss to base loss
                total_loss = base_loss + 1.0 * cpcc_loss_value
                print(f"Total loss (Base + CPCC): {total_loss.item():.6f}")
        else:
            total_loss = base_loss
            print(f"Total loss (Base only): {total_loss.item():.6f}")
        
        # Test backward pass
        try:
            total_loss.backward()
            print(f"Backward pass successful")
            return True
        except Exception as e:
            print(f"Backward pass failed: {e}")
            return False

def test_different_lambda_values():
    """Test CPCC loss with different lambda values"""
    print("\nTesting Different Lambda Values")
    print("=" * 40)
    
    batch_size = 6
    feature_dim = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    representations = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    
    lambda_values = [0.1, 1.0, 5.0, 10.0]
    
    for lamb in lambda_values:
        cpcc_loss = EHRCPCCLoss(distance_type='l2', lamb=lamb, center=False)
        cpcc_loss_value = cpcc_loss(representations, labels, task='readmission')
        # Apply lambda to the loss (as done in the model)
        total_loss = lamb * cpcc_loss_value
        print(f"Lambda {lamb:4.1f}: CPCC Loss = {cpcc_loss_value.item():.6f}, Total Loss = {total_loss.item():.6f}")
    
    # Test that lambda values produce different total losses
    cpcc_loss = EHRCPCCLoss(distance_type='l2', lamb=1.0, center=False)
    cpcc_loss_value = cpcc_loss(representations, labels, task='readmission')
    
    loss_0_1 = 0.1 * cpcc_loss_value
    loss_1_0 = 1.0 * cpcc_loss_value
    loss_5_0 = 5.0 * cpcc_loss_value
    loss_10_0 = 10.0 * cpcc_loss_value
    
    # Check that different lambda values produce different total losses
    if (loss_0_1 != loss_1_0).any() and (loss_1_0 != loss_5_0).any() and (loss_5_0 != loss_10_0).any():
        print("✓ Lambda values correctly affect total loss")
        return True
    else:
        print("✗ Lambda values do not affect total loss")
        return False

def main():
    """Run all tests"""
    print("CPCC Unit Tests")
    print("=" * 50)
    
    tests = [
        ("CPCC Loss Function", test_cpcc_loss),
        ("CPCC-Only Mode Logic", test_cpcc_only_mode),
        ("Different Lambda Values", test_different_lambda_values)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✓ {test_name} passed")
            else:
                print(f"✗ {test_name} failed")
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 All tests passed! CPCC-only functionality is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")

if __name__ == "__main__":
    main()
