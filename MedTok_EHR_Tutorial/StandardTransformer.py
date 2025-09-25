'''
Standard Transformer Model for EHR Data
A simplified Transformer model that maintains compatibility with the existing EHRModel interface.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule as LM
import math
from torch import Tensor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
from embedding_utils import load_embeddings_from_json
from ehr_cpcc_loss import EHRCPCCLoss, EHRHierarchicalLoss


class StandardPositionalEncoding(nn.Module):
    """Standard positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StandardTransformer(pl.LightningModule):
    """Standard Transformer model for EHR data processing."""
    
    def __init__(self, model_name, input_dim, num_feat=128, num_heads=4,
                 hidden_dim=256, output_dim=128, num_layers=3, max_visit_num=50,
                 dropout_prob=0.5, pred_threshold=0.5, max_ehr_length=3000, code_size=600000,
                 lr=0.0001, wd=0.0, lr_factor=0.01, num_class=2, task='readmission', lr_patience=100, lr_threshold=1e-4,
                 lr_threshold_mode='rel', lr_cooldown=0, min_lr=1e-8, eps=1e-8, lr_total_iters=10, memory_bank_size=512, 
                 pre_trained_embedding='../pre_trained_model_/{pretrained_model_name}/embeddings_all.npy',
                 # CPCC Loss parameters
                 use_cpcc=False, cpcc_lamb=1.0, cpcc_distance_type='l2', cpcc_center=False, cpcc_only=False,
                 hparams=None):
        
        super().__init__()
        
        self.model_name = model_name
        self.num_feat = num_feat
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.pred_threshold = pred_threshold
        self.max_visit_num = max_visit_num
        self.code_size = code_size
        self.input_dim = input_dim
        
        # Learning rate parameters
        self.lr = lr
        self.wd = wd
        self.lr_factor = lr_factor
        self.lr_total_iters = lr_total_iters
        self.mask_prob = 0.2
        
        # Define embeddings
        if pre_trained_embedding.endswith('.json'):
            # Load from JSON format
            self.emb = torch.from_numpy(load_embeddings_from_json(pre_trained_embedding)).cuda()
        else:
            # Load from numpy format
            self.emb = torch.from_numpy(np.load(pre_trained_embedding)).cuda()
        
        print("self.emb")
        print(self.emb.shape)
        print(self.emb.device)
        
        # CLS token embedding
        self.cls_emb = torch.nn.Parameter(torch.randn(1, output_dim)).cuda()
        print(self.cls_emb.device)
        
        # Missing token embedding
        self.miss_emb = torch.nn.Parameter(torch.randn(1, 256)).cuda()
        
        # Demographics embeddings
        self.gender_emb = nn.Embedding(5, input_dim)
        self.ethnicity_emb = nn.Embedding(100, input_dim)
        
        # Medical code embeddings
        self.med_code_emb = torch.concat([self.emb, self.miss_emb], dim=0)
        
        # Projection layer
        self.projector = torch.nn.Linear(256, input_dim)
        
        # Memory bank
        self.memory_bank = torch.randn((memory_bank_size, output_dim)).to(self.device)
        self.memory_bank_size = memory_bank_size
        
        # Standard Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_prob,
                batch_first=False  # Standard Transformer expects (seq_len, batch, feature)
            ),
            num_layers=num_layers
        )
        
        # Positional encoding
        self.position_encoder = StandardPositionalEncoding(
            d_model=input_dim,
            dropout=0.2,
            max_len=max_ehr_length + 1
        )
        
        # Output layers
        self.fc = nn.Linear(input_dim, output_dim)
        self.classify = nn.Linear(output_dim, num_class)
        self.task = task
        self.num_class = num_class
        
        # CPCC Loss initialization
        self.use_cpcc = use_cpcc
        self.cpcc_lamb = cpcc_lamb
        self.cpcc_distance_type = cpcc_distance_type
        self.cpcc_center = cpcc_center
        self.cpcc_only = cpcc_only
        
        if self.use_cpcc:
            self.cpcc_loss = EHRCPCCLoss(
                distance_type=cpcc_distance_type,
                lamb=cpcc_lamb,
                center=cpcc_center
            )
            print(f"CPCC Loss initialized with lambda={cpcc_lamb}, distance_type={cpcc_distance_type}, center={cpcc_center}")
    
    def forward(self, inputs, pos=None):
        """Forward pass of the model."""
        patient_embedding = self.patientEncoder(inputs)
        
        if pos is not None:
            pos_embedding = self.patientEncoder(pos)
        else:
            pos_embedding = None
        
        prob_logits = self.classify(patient_embedding)
        
        return patient_embedding, prob_logits, pos_embedding
    
    def patientEncoder(self, data):
        """Encode patient data using standard Transformer."""
        # Get medical code embeddings
        src_emb = data.x  # [bz, max_medical_code, 1]
        src_emb = self.med_code_emb[src_emb].squeeze()  # [bz, max_medical_code, 256]
        src_emb = self.projector(src_emb)  # [bz, max_medical_code, input_dim]
        
        # Add CLS token
        cls_emb = self.cls_emb.repeat(src_emb.size(0), 1).unsqueeze(1).to(src_emb.device)
        
        # Add demographics
        gender_emb = self.gender_emb(data.gender).unsqueeze(1)
        ethnicity_emb = self.ethnicity_emb(data.ethnicity).unsqueeze(1)
        
        # Concatenate all embeddings
        src_emb = torch.concat([cls_emb, gender_emb, ethnicity_emb, src_emb], dim=1)
        
        # Create mask for padding tokens
        src_mask = torch.concat([
            torch.zeros(src_emb.size(0), 3).bool().to(src_emb.device),  # CLS, gender, ethnicity
            data.code_mask.squeeze().bool()  # Medical codes
        ], dim=-1)
        
        # Transpose for Transformer: (seq_len, batch, feature)
        src_emb = src_emb.transpose(0, 1)  # [seq_len, batch, input_dim]
        
        # Add positional encoding
        src_emb = self.position_encoder(src_emb)
        
        # Pass through Transformer encoder
        encoded_output = self.transformer_encoder(src_emb, src_key_padding_mask=src_mask)
        
        # Take CLS token output (first token)
        encoded_output = encoded_output[0, :, :]  # [batch, input_dim]
        
        # Project to output dimension
        output = self.fc(encoded_output)
        
        return output
    
    def compute_metrics(self, label, logits):
        """Compute evaluation metrics."""
        if self.task == 'lenofstay' or self.task == 'phenotype' or self.task == 'drugrec':
            label_numpy = label.detach().cpu().numpy()
            print("label_numpy", label_numpy.size)
            if label_numpy.shape[-1] == 1 or len(label_numpy.shape) == 1:
                y_true_one_hot = np.zeros((label_numpy.size, self.num_class))
                y_true_one_hot[np.arange(label_numpy.size), label_numpy.flatten().astype(int)] = 1
            else:
                y_true_one_hot = label_numpy

            logits = logits.detach().cpu().numpy()
           
            auroc = roc_auc_score(y_true_one_hot, logits, average='micro')
            aupr = average_precision_score(y_true_one_hot, logits, average='micro')
            f1 = f1_score(y_true_one_hot, (logits >= 0.2).astype(int), average='weighted')
            return auroc, aupr, f1
        else:
            logits = logits[:, 1].detach().cpu().numpy()
            auc = roc_auc_score(label.cpu().numpy(), logits)
            aupr = average_precision_score(label.cpu().numpy(), logits)
            logits_i_int = [1 if j > 0.5 else 0 for j in logits]
            f1 = f1_score(label.cpu().numpy(), logits_i_int)
            return auc, aupr, f1
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        sample = batch
        patient_embedding, prob_logits, _ = self(sample, None)
        
        # Prepare labels
        if sample.label.shape[-1] == 1 or len(sample.label.shape) == 1:
            y_true_one_hot = torch.zeros((sample.label.size(0), self.num_class)).to(self.device)
            y_true_one_hot[torch.arange(sample.label.size(0)).long(), sample.label.long()] = 1
        else:
            y_true_one_hot = sample.label
        
        # Compute base loss (only if not using CPCC only)
        if not self.cpcc_only:
            if self.task == 'lenofstay':
                loss = F.cross_entropy(prob_logits, y_true_one_hot)
            else:
                loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Add CPCC loss if enabled
        if self.use_cpcc:
            if len(sample.label.shape) > 1 and sample.label.shape[-1] > 1:
                labels_for_cpcc = sample.label.argmax(dim=-1)
            else:
                labels_for_cpcc = sample.label.squeeze()
            
            cpcc_loss_value = self.cpcc_loss(patient_embedding, labels_for_cpcc, self.task)
            
            if self.cpcc_only:
                # Use only CPCC loss
                loss = self.cpcc_lamb * cpcc_loss_value
            else:
                # Add CPCC loss to base loss
                loss = loss + self.cpcc_lamb * cpcc_loss_value
            
            if self.cpcc_center:
                center_reg = 0.01 * torch.norm(torch.mean(patient_embedding, dim=0))
                loss = loss + center_reg
            
            self.log("train/cpcc_loss", cpcc_loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=sample.size(0))
            self.log("train/base_loss", loss - self.cpcc_lamb * cpcc_loss_value if not self.cpcc_only else 0.0, on_step=True, on_epoch=True, prog_bar=True, batch_size=sample.size(0))
        
        # Convert logits to probabilities
        if self.task == 'lenofstay' or self.task == 'readmission' or self.task == 'mortality':
            prob_logits = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits = torch.sigmoid(prob_logits)
        
        auc, aupr, f1 = self.compute_metrics(sample.label, prob_logits)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=sample.size(0))
        self.log("train/auc", auc, on_step=True, on_epoch=True, prog_bar=True, batch_size=sample.size(0))
        self.log("train/aupr", aupr, on_step=True, on_epoch=True, prog_bar=True, batch_size=sample.size(0))
        self.log("train/f1", f1, on_step=True, on_epoch=True, prog_bar=True, batch_size=sample.size(0))
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        sample = batch
        batch_size = sample.size(0)
        
        patient_embedding, prob_logits, _ = self(sample, None)
        
        # Prepare labels
        if sample.label.shape[-1] == 1 or len(sample.label.shape) == 1:
            y_true_one_hot = torch.zeros((sample.label.size(0), self.num_class)).to(self.device)
            y_true_one_hot[torch.arange(sample.label.size(0)).long(), sample.label.long()] = 1
        else:
            y_true_one_hot = sample.label
        
        # Compute base loss (only if not using CPCC only)
        if not self.cpcc_only:
            if self.task == 'lenofstay':
                loss = F.cross_entropy(prob_logits, y_true_one_hot)
            else:
                loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Add CPCC loss if enabled
        if self.use_cpcc:
            if len(sample.label.shape) > 1 and sample.label.shape[-1] > 1:
                labels_for_cpcc = sample.label.argmax(dim=-1)
            else:
                labels_for_cpcc = sample.label.squeeze()
            
            cpcc_loss_value = self.cpcc_loss(patient_embedding, labels_for_cpcc, self.task)
            
            if self.cpcc_only:
                # Use only CPCC loss
                loss = self.cpcc_lamb * cpcc_loss_value
            else:
                # Add CPCC loss to base loss
                loss = loss + self.cpcc_lamb * cpcc_loss_value
            
            if self.cpcc_center:
                center_reg = 0.01 * torch.norm(torch.mean(patient_embedding, dim=0))
                loss = loss + center_reg
            
            self.log("val/cpcc_loss", cpcc_loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("val/base_loss", loss - self.cpcc_lamb * cpcc_loss_value if not self.cpcc_only else 0.0, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Compute metrics
        if self.task == 'lenofstay' or self.task == 'readmission' or self.task == 'mortality':
            prob_logits = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits = torch.sigmoid(prob_logits)
        
        auc, aupr, f1 = self.compute_metrics(sample.label, F.softmax(prob_logits, dim=-1))
        
        values = {
            "val/loss": loss.detach(),
            "val/auc": auc,
            "val/aupr": aupr,
            "val/f1": f1
        }
        self.log_dict(values, batch_size=batch_size)
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        sample = batch
        batch_size = sample.size(0)
        
        patient_embedding, prob_logits, _ = self(sample, None)
        
        # Prepare labels
        if sample.label.shape[-1] == 1 or len(sample.label.shape) == 1:
            y_true_one_hot = torch.zeros((sample.label.size(0), self.num_class)).to(self.device)
            y_true_one_hot[torch.arange(sample.label.size(0)).long(), sample.label.long()] = 1
        else:
            y_true_one_hot = sample.label
        
        # Compute base loss (only if not using CPCC only)
        if not self.cpcc_only:
            if self.task == 'lenofstay':
                loss = F.cross_entropy(prob_logits, y_true_one_hot)
            else:
                loss = F.binary_cross_entropy_with_logits(prob_logits, y_true_one_hot)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Add CPCC loss if enabled
        if self.use_cpcc:
            if len(sample.label.shape) > 1 and sample.label.shape[-1] > 1:
                labels_for_cpcc = sample.label.argmax(dim=-1)
            else:
                labels_for_cpcc = sample.label.squeeze()
            
            cpcc_loss_value = self.cpcc_loss(patient_embedding, labels_for_cpcc, self.task)
            
            if self.cpcc_only:
                # Use only CPCC loss
                loss = self.cpcc_lamb * cpcc_loss_value
            else:
                # Add CPCC loss to base loss
                loss = loss + self.cpcc_lamb * cpcc_loss_value
            
            if self.cpcc_center:
                center_reg = 0.01 * torch.norm(torch.mean(patient_embedding, dim=0))
                loss = loss + center_reg
            
            self.log("test/cpcc_loss", cpcc_loss_value, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("test/base_loss", loss - self.cpcc_lamb * cpcc_loss_value if not self.cpcc_only else 0.0, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Compute metrics
        if self.task == 'lenofstay' or self.task == 'readmission' or self.task == 'mortality':
            prob_logits = F.softmax(prob_logits, dim=-1)
        else:
            prob_logits = torch.sigmoid(prob_logits)
        
        auc, aupr, f1 = self.compute_metrics(sample.label, prob_logits)
        
        values = {
            "test/loss": loss.detach(),
            "test/auc": auc,
            "test/aupr": aupr,
            "test/f1": f1
        }
        self.log_dict(values, batch_size=batch_size)
    
    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        patient_subgraph, _ = batch
        patient_embeddings, prob_logits, _ = self(patient_subgraph)
        prob_true = prob_logits[:, 1]
        return patient_embeddings, prob_true
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                'name': 'curr_lr',
            },
        }
    
    @torch.no_grad()
    def get_embeddings(self, dataloader, device):
        """Get embeddings from the model."""
        self.to(device)
        self.eval()
        embeddings = []
        for idx, batch in tqdm(dataloader, desc='Getting embeddings'):
            print(batch)
            x = batch
            x = x.to(device)
            embedding, _ = self(x)
            print("get_embeddings", embedding)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)
