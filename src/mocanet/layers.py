"""Core layers for MOCA-Net architecture."""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenRouter(nn.Module):
    """Token router that selects experts and memory slots under compute budget."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_experts: int,
        num_memory_slots: int,
        top_k_experts: int,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.num_memory_slots = num_memory_slots
        self.top_k_experts = top_k_experts
        self.temperature = temperature
        
        # Expert routing network
        self.expert_router = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_experts),
        )
        
        # Memory routing network
        self.memory_router = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_memory_slots),
        )
        
        # Budget predictor
        self.budget_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.ReLU(),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        token_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of token router.
        
        Args:
            token_embeddings: [batch_size, seq_len, embedding_dim]
            
        Returns:
            expert_weights: [batch_size, seq_len, num_experts]
            memory_weights: [batch_size, seq_len, num_memory_slots]
            expert_mask: [batch_size, seq_len, num_experts]
            memory_mask: [batch_size, seq_len, num_memory_slots]
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Route to experts
        expert_logits = self.expert_router(token_embeddings) / self.temperature
        expert_probs = F.softmax(expert_logits, dim=-1)
        
        # Top-k expert selection
        top_k_expert_probs, top_k_indices = torch.topk(
            expert_probs, k=self.top_k_experts, dim=-1
        )
        
        # Renormalize top-k probabilities to sum to 1
        top_k_expert_probs = top_k_expert_probs / top_k_expert_probs.sum(dim=-1, keepdim=True)
        
        # Create sparse expert weights
        expert_weights = torch.zeros_like(expert_probs)
        expert_weights.scatter_(-1, top_k_indices, top_k_expert_probs)
        
        # Create expert mask
        expert_mask = (expert_weights > 0).float()
        
        # Route to memory slots
        memory_logits = self.memory_router(token_embeddings) / self.temperature
        memory_probs = F.softmax(memory_logits, dim=-1)
        
        # Top-k memory selection (use fewer memory slots for efficiency)
        top_k_memory = min(self.top_k_experts, self.num_memory_slots // 4)
        top_k_memory_probs, top_k_memory_indices = torch.topk(
            memory_probs, k=top_k_memory, dim=-1
        )
        
        # Renormalize top-k memory probabilities to sum to 1
        top_k_memory_probs = top_k_memory_probs / top_k_memory_probs.sum(dim=-1, keepdim=True)
        
        # Create sparse memory weights
        memory_weights = torch.zeros_like(memory_probs)
        memory_weights.scatter_(-1, top_k_memory_indices, top_k_memory_probs)
        
        # Create memory mask
        memory_mask = (memory_weights > 0).float()
        
        # Predict compute budget usage
        budget_usage = self.budget_predictor(token_embeddings).squeeze(-1)
        
        return expert_weights, memory_weights, expert_mask, memory_mask, budget_usage


class MemoryBank(nn.Module):
    """External memory bank with learnable keys and values."""
    
    def __init__(
        self,
        num_slots: int,
        embedding_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Learnable memory slots
        self.memory_keys = nn.Parameter(torch.randn(num_slots, embedding_dim))
        self.memory_values = nn.Parameter(torch.randn(num_slots, embedding_dim))
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )
        
        # Memory decay factor
        self.memory_decay = nn.Parameter(torch.tensor(0.95))
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        memory_weights: torch.Tensor,
        update_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of memory bank.
        
        Args:
            query: [batch_size, seq_len, embedding_dim]
            memory_weights: [batch_size, seq_len, num_slots]
            update_input: Optional input for memory updates
            
        Returns:
            memory_output: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # Multi-head attention over memory
        query_reshaped = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys_reshaped = self.memory_keys.view(self.num_slots, self.num_heads, self.head_dim)
        values_reshaped = self.memory_values.view(self.num_slots, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.einsum('bshd,nhd->bsn', query_reshaped, keys_reshaped)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply memory routing weights
        attention_scores = attention_scores * memory_weights
        
        # Softmax over memory slots
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Read from memory
        memory_read = torch.einsum('bsn,nhd->bshd', attention_probs, values_reshaped)
        memory_read = memory_read.view(batch_size, seq_len, self.embedding_dim)
        
        # Apply memory decay to the read operation (this makes the parameter differentiable)
        memory_read = memory_read * self.memory_decay
        
        # Optional memory update
        if update_input is not None:
            # Compute update gate
            gate_input = torch.cat([memory_read, update_input], dim=-1)
            update_gate = self.update_gate(gate_input)
            
            # Update memory values (with decay)
            memory_update = update_gate * update_input + (1 - update_gate) * memory_read
            
            # Apply update to memory bank (detached to avoid gradient issues)
            with torch.no_grad():
                # Simple update: average over batch and sequence
                memory_update_avg = memory_update.mean(dim=(0, 1))
                self.memory_values.data = (
                    self.memory_values.data * 0.9 + memory_update_avg * 0.1
                )
            
            # Make the update gate contribute to the output in a differentiable way
            # by using it to modulate the memory read
            memory_read = memory_read * (1 + update_gate.mean(dim=-1, keepdim=True))
        
        # Project and normalize output
        output = self.output_proj(memory_read)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output


class ExpertLayer(nn.Module):
    """Mixture of experts layer with shared input/output projections."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_experts: int,
        expert_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        
        # Shared input/output projections
        self.input_proj = nn.Linear(embedding_dim, expert_hidden_dim)
        self.output_proj = nn.Linear(expert_hidden_dim, embedding_dim)
        
        # Expert networks (lightweight MLPs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of expert layer.
        
        Args:
            input_tensor: [batch_size, seq_len, embedding_dim]
            expert_weights: [batch_size, seq_len, num_experts]
            
        Returns:
            expert_output: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = input_tensor.shape
        
        # Project input
        projected_input = self.input_proj(input_tensor)
        
        # Apply each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(projected_input)
            expert_outputs.append(expert_out)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [B, S, E, H]
        
        # Weight expert outputs
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # [B, S, E, 1]
        weighted_experts = expert_outputs * expert_weights_expanded
        
        # Sum over experts
        combined_experts = weighted_experts.sum(dim=2)  # [B, S, H]
        
        # Project output
        output = self.output_proj(combined_experts)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        return output


class BudgetLoss(nn.Module):
    """Differentiable budget loss to encourage efficient resource usage."""
    
    def __init__(self, target_budget: float = 0.3, weight: float = 0.05):
        super().__init__()
        self.target_budget = target_budget
        self.weight = weight
    
    def forward(
        self,
        expert_usage: torch.Tensor,
        memory_usage: torch.Tensor,
        predicted_budget: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute budget loss.
        
        Args:
            expert_usage: [batch_size, seq_len] - fraction of experts used
            memory_usage: [batch_size, seq_len] - fraction of memory slots used
            predicted_budget: [batch_size, seq_len] - predicted budget usage
            
        Returns:
            budget_loss: scalar tensor
        """
        # Actual resource usage
        total_usage = (expert_usage + memory_usage) / 2.0
        
        # Budget violation penalty
        budget_violation = F.relu(total_usage - self.target_budget)
        budget_violation_loss = budget_violation.mean()
        
        # Prediction accuracy penalty
        prediction_loss = F.mse_loss(predicted_budget, total_usage)
        
        # Combined loss
        total_loss = budget_violation_loss + prediction_loss
        
        return self.weight * total_loss
