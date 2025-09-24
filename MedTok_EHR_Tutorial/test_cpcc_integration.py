#!/usr/bin/env python3
"""
Test script to verify CPCC loss integration with EHR model.
This script tests the basic functionality without requiring full training.
"""

import torch
import torch.nn as nn
import numpy as np
from ehr_cpcc_loss import EHRCPCCLoss, EHRHierarchicalLoss

def test_cpcc_loss():
    """Test CPCC loss functionality"""
    print("Testing CPCC Loss...")
    
    # Create test data
    batch_size = 8
    feature_dim = 64
    num_classes = 2
    
    # Generate random representations and labels
    representations = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test different distance types
    distance_types = ['l2', 'l1', 'cosine', 'poincare']
    
    for distance_type in distance_types:
        print(f"\nTesting distance type: {distance_type}")
        
        # Initialize CPCC loss
        cpcc_loss = EHRCPCCLoss(
            distance_type=distance_type,
            lamb=1.0,
            center=False
        )
        
        # Test forward pass
        try:
            cpcc_loss_value = cpcc_loss(representations, labels, task='readmission')
            print(f"  CPCC Loss: {cpcc_loss_value.item():.4f}")
            
            # Test combined loss
            base_loss = torch.tensor(0.5)  # Mock base loss
            total_loss, cpcc_val = cpcc_loss.compute_combined_loss(
                base_loss, representations, labels, 'readmission'
            )
            print(f"  Combined Loss: {total_loss.item():.4f}")
            print(f"  CPCC Component: {cpcc_val.item():.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nCPCC Loss test completed!")

def test_hierarchical_loss():
    """Test hierarchical loss functionality"""
    print("\nTesting Hierarchical Loss...")
    
    # Create test data
    batch_size = 8
    feature_dim = 64
    max_codes = 10
    
    # Generate random data
    representations = torch.randn(batch_size, feature_dim)
    medical_codes = torch.randint(0, 100, (batch_size, max_codes))
    labels = torch.randint(0, 2, (batch_size,))
    
    # Initialize hierarchical loss
    hierarchical_loss = EHRHierarchicalLoss(temperature=0.1)
    
    try:
        loss_value = hierarchical_loss(representations, medical_codes, labels)
        print(f"Hierarchical Loss: {loss_value.item():.4f}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Hierarchical Loss test completed!")

def test_ehr_model_integration():
    """Test EHR model integration with CPCC loss"""
    print("\nTesting EHR Model Integration...")
    
    try:
        # Import EHR model
        from EHRModel_token import EHRModel
        
        # Create model with CPCC loss enabled
        model = EHRModel(
            model_name='Transformer',
            input_dim=64,
            num_heads=4,
            hidden_dim=128,
            output_dim=64,
            num_layers=2,
            task='readmission',
            num_class=2,
            use_cpcc=True,
            cpcc_lamb=1.0,
            cpcc_distance_type='l2',
            cpcc_center=False
        )
        
        print("EHR Model with CPCC loss created successfully!")
        print(f"Use CPCC: {model.use_cpcc}")
        print(f"CPCC Lambda: {model.cpcc_lamb}")
        print(f"CPCC Distance Type: {model.cpcc_distance_type}")
        
        # Test model forward pass (mock data)
        batch_size = 4
        mock_data = type('MockData', (), {
            'x': torch.randint(0, 1000, (batch_size, 10)),
            'visit_id': torch.randint(0, 5, (batch_size, 10, 1)),
            'timestamp_within_visits': torch.randn(batch_size, 10, 3),
            'timestamp_between_visits': torch.randn(batch_size, 10, 3),
            'gender': torch.randint(0, 5, (batch_size,)),
            'ethnicity': torch.randint(0, 100, (batch_size,)),
            'code_mask': torch.ones(batch_size, 10, 1),
            'label': torch.randint(0, 2, (batch_size,)),
            'size': lambda: batch_size
        })()
        
        # Test forward pass
        with torch.no_grad():
            patient_embedding, prob_logits, _ = model(mock_data, None)
            print(f"Patient embedding shape: {patient_embedding.shape}")
            print(f"Prob logits shape: {prob_logits.shape}")
        
        print("EHR Model integration test completed!")
        
    except Exception as e:
        print(f"Error in EHR model integration: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing CPCC Loss Integration with EHR Model")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_cpcc_loss()
    test_hierarchical_loss()
    test_ehr_model_integration()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
