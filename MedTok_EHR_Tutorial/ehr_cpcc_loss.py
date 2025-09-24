import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *
import math

class EHRCPCCLoss(nn.Module):
    """
    CPCC (Correlation Preserving Contrastive Coding) Loss adapted for EHR data.
    This loss encourages the learned representations to preserve hierarchical relationships
    in medical codes and patient outcomes.
    """
    
    def __init__(self, args=None, distance_type='l2', lamb=1.0, center=False):
        super(EHRCPCCLoss, self).__init__()
        self.distance_type = distance_type
        self.lamb = lamb
        self.center = center
        
        # For EHR data, we can create hierarchical relationships based on:
        # 1. Medical code hierarchies (ICD codes have hierarchical structure)
        # 2. Patient outcome hierarchies (mortality > readmission > length of stay)
        # 3. Visit-level hierarchies (recent visits vs historical visits)
        
        # Initialize medical code hierarchy (simplified version)
        self.medical_code_hierarchy = self._create_medical_code_hierarchy()
        
    def _create_medical_code_hierarchy(self):
        """
        Create a simplified medical code hierarchy for demonstration.
        In practice, this should be loaded from actual medical coding systems.
        """
        hierarchy = {
            # ICD-10 hierarchy example
            'A00': ['A00-A09'],  # Infectious diseases
            'A01': ['A00-A09'],
            'B00': ['B00-B99'],  # Other infectious diseases
            'C00': ['C00-C97'],  # Neoplasms
            'D00': ['D00-D89'],  # Blood disorders
            'E00': ['E00-E89'],  # Endocrine disorders
            'F00': ['F00-F99'],  # Mental disorders
            'G00': ['G00-G99'],  # Nervous system
            'H00': ['H00-H59'],  # Eye disorders
            'I00': ['I00-I99'],  # Circulatory system
            'J00': ['J00-J99'],  # Respiratory system
            'K00': ['K00-K95'],  # Digestive system
            'L00': ['L00-L99'],  # Skin disorders
            'M00': ['M00-M99'],  # Musculoskeletal
            'N00': ['N00-N99'],  # Genitourinary
            'O00': ['O00-O9A'],  # Pregnancy
            'P00': ['P00-P96'],  # Perinatal
            'Q00': ['Q00-Q99'],  # Congenital
            'R00': ['R00-R94'],  # Symptoms
            'S00': ['S00-T88'],  # Injury/Poisoning
            'U00': ['U00-U85'],  # Special purposes
            'V00': ['V00-Y99'],  # External causes
            'Z00': ['Z00-Z99'],  # Health status
        }
        return hierarchy
    
    def _compute_hierarchical_distance(self, labels, device):
        """
        Compute hierarchical distances between medical outcomes.
        For EHR data, we can define hierarchies based on:
        1. Severity levels (mortality > readmission > length of stay)
        2. Medical code categories
        3. Visit patterns
        """
        batch_size = labels.shape[0]
        distances = torch.zeros((batch_size, batch_size), device=device)
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # Define hierarchical distance based on task type
                if hasattr(self, 'task'):
                    if self.task == 'mortality':
                        # Mortality is the highest level
                        if labels[i] == labels[j]:
                            distances[i, j] = 0.0  # Same outcome
                        else:
                            distances[i, j] = 1.0  # Different mortality outcomes
                    elif self.task == 'readmission':
                        # Readmission is second level
                        if labels[i] == labels[j]:
                            distances[i, j] = 0.0
                        else:
                            distances[i, j] = 0.5
                    elif self.task == 'lenofstay':
                        # Length of stay is third level
                        if labels[i] == labels[j]:
                            distances[i, j] = 0.0
                        else:
                            distances[i, j] = 0.3
                    else:
                        # Default: binary classification distance
                        distances[i, j] = 1.0 if labels[i] != labels[j] else 0.0
                else:
                    # Default binary distance
                    distances[i, j] = 1.0 if labels[i] != labels[j] else 0.0
                
                distances[j, i] = distances[i, j]  # Symmetric
        
        return distances
    
    def _compute_representation_distance(self, representations):
        """
        Compute pairwise distances between representations based on distance_type.
        """
        if self.distance_type == 'l2':
            # Euclidean distance
            pairwise_dist = F.pdist(representations, p=2.0)
        elif self.distance_type == 'l1':
            # Manhattan distance
            pairwise_dist = F.pdist(representations, p=1.0)
        elif self.distance_type == 'cosine':
            # Cosine distance
            normalized_repr = F.normalize(representations, dim=1)
            pairwise_dist = F.pdist(normalized_repr, p=2.0)
        elif self.distance_type == 'poincare':
            # Simplified Poincare distance (without full hyperbolic geometry)
            epsilon = 1e-5
            all_norms = torch.norm(representations, dim=1, p=2).unsqueeze(-1)
            normalized_repr = representations * (1 - epsilon) / all_norms
            
            condensed_idx = torch.triu_indices(len(representations), len(representations), offset=1, device=representations.device)
            numerator_square = torch.sum((normalized_repr[None, :] - normalized_repr[:, None])**2, -1)
            numerator = numerator_square[condensed_idx[0], condensed_idx[1]]
            
            all_normalized_norms = torch.norm(normalized_repr, dim=1, p=2)
            denominator_square = ((1 - all_normalized_norms**2).reshape(-1,1)) @ (1 - all_normalized_norms**2).reshape(1,-1)
            denominator = denominator_square[condensed_idx[0], condensed_idx[1]]
            
            pairwise_dist = torch.acosh(1 + 2 * (numerator/denominator))
        else:
            # Default to L2 distance
            pairwise_dist = F.pdist(representations, p=2.0)
        
        return pairwise_dist
    
    def forward(self, representations, labels, task=None):
        """
        Compute CPCC loss for EHR data.
        
        Args:
            representations: Patient representations [batch_size, feature_dim]
            labels: Patient labels [batch_size]
            task: Task type ('mortality', 'readmission', 'lenofstay', etc.)
        """
        self.task = task
        device = representations.device
        batch_size = representations.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute hierarchical distances between labels
        hierarchical_distances = self._compute_hierarchical_distance(labels, device)
        
        # Extract upper triangular part for pairwise distances
        condensed_idx = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
        tree_pairwise_dist = hierarchical_distances[condensed_idx[0], condensed_idx[1]]
        
        # Compute representation distances
        repr_pairwise_dist = self._compute_representation_distance(representations)
        
        # Compute correlation between hierarchical and representation distances
        if len(tree_pairwise_dist) > 1 and len(repr_pairwise_dist) > 1:
            # Stack distances for correlation computation
            stacked_distances = torch.stack([repr_pairwise_dist, tree_pairwise_dist], dim=0)
            
            # Compute correlation coefficient
            correlation_matrix = torch.corrcoef(stacked_distances)
            correlation = correlation_matrix[0, 1]
            
            # CPCC loss: maximize correlation (minimize 1 - correlation)
            cpcc_loss = 1 - correlation
            
            # Handle NaN values
            if torch.isnan(cpcc_loss):
                cpcc_loss = torch.tensor(1.0, device=device)
        else:
            cpcc_loss = torch.tensor(1.0, device=device)
        
        return cpcc_loss
    
    def compute_combined_loss(self, base_loss, representations, labels, task=None):
        """
        Compute combined loss: base_loss + lambda * cpcc_loss
        
        Args:
            base_loss: Original classification loss
            representations: Patient representations
            labels: Patient labels
            task: Task type
        """
        cpcc_loss = self.forward(representations, labels, task)
        
        if self.center:
            # Add centering regularization
            center_reg = 0.01 * torch.norm(torch.mean(representations, dim=0))
            total_loss = base_loss + self.lamb * cpcc_loss + center_reg
        else:
            total_loss = base_loss + self.lamb * cpcc_loss
        
        return total_loss, cpcc_loss


class EHRHierarchicalLoss(nn.Module):
    """
    Alternative hierarchical loss for EHR data that considers medical code hierarchies.
    """
    
    def __init__(self, medical_code_hierarchy=None, temperature=0.1):
        super(EHRHierarchicalLoss, self).__init__()
        self.temperature = temperature
        self.medical_code_hierarchy = medical_code_hierarchy or {}
        
    def forward(self, representations, medical_codes, labels):
        """
        Compute hierarchical loss based on medical code relationships.
        
        Args:
            representations: Patient representations [batch_size, feature_dim]
            medical_codes: Medical codes for each patient [batch_size, max_codes]
            labels: Patient labels [batch_size]
        """
        device = representations.device
        batch_size = representations.shape[0]
        
        # Normalize representations
        normalized_repr = F.normalize(representations, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(normalized_repr, normalized_repr.t())
        
        # Create hierarchical mask based on medical code similarities
        hierarchical_mask = self._create_hierarchical_mask(medical_codes, device)
        
        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature
        
        # Compute hierarchical contrastive loss
        # Positive pairs: patients with similar medical code hierarchies
        # Negative pairs: patients with different medical code hierarchies
        
        # Extract positive and negative similarities
        positive_similarities = similarity_matrix[hierarchical_mask == 1]
        negative_similarities = similarity_matrix[hierarchical_mask == 0]
        
        if len(positive_similarities) > 0 and len(negative_similarities) > 0:
            # Compute InfoNCE-style loss
            pos_exp = torch.exp(positive_similarities)
            neg_exp = torch.exp(negative_similarities)
            
            loss = -torch.log(pos_exp / (pos_exp + neg_exp.sum()))
            loss = loss.mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def _create_hierarchical_mask(self, medical_codes, device):
        """
        Create mask indicating hierarchical relationships between patients.
        """
        batch_size = medical_codes.shape[0]
        mask = torch.zeros((batch_size, batch_size), device=device)
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # Compute medical code similarity
                codes_i = medical_codes[i]
                codes_j = medical_codes[j]
                
                # Simple overlap-based similarity
                overlap = len(set(codes_i.tolist()) & set(codes_j.tolist()))
                total_codes = len(set(codes_i.tolist()) | set(codes_j.tolist()))
                
                if total_codes > 0:
                    similarity = overlap / total_codes
                    if similarity > 0.3:  # Threshold for hierarchical relationship
                        mask[i, j] = 1
                        mask[j, i] = 1
        
        return mask
