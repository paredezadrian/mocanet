"""Main MOCA-Net model implementation."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TokenRouter, MemoryBank, ExpertLayer, BudgetLoss
from .config import ModelConfig


class MOCANet(nn.Module):
    """Memory-Orchestrated Context Allocation Network."""
    
    def __init__(self, config: ModelConfig, vocab_size: int, num_classes: int = 1):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1000, config.embedding_dim)  # Max sequence length
        )
        
        # Core MOCA-Net components
        self.token_router = TokenRouter(
            embedding_dim=config.embedding_dim,
            num_experts=config.num_experts,
            num_memory_slots=config.num_memory_slots,
            top_k_experts=config.top_k_experts,
            temperature=config.router_temperature,
            dropout=config.dropout,
        )
        
        self.memory_bank = MemoryBank(
            num_slots=config.num_memory_slots,
            embedding_dim=config.embedding_dim,
            num_heads=config.num_memory_heads,
            dropout=config.dropout,
        )
        
        self.expert_layer = ExpertLayer(
            embedding_dim=config.embedding_dim,
            num_experts=config.num_experts,
            expert_hidden_dim=config.embedding_dim // 2,
            dropout=config.dropout,
        )
        
        # Budget loss
        self.budget_loss = BudgetLoss(
            target_budget=0.3,
            weight=config.budget_loss_weight,
        )
        
        # Output layers
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        
        # Task-specific output heads
        if num_classes > 1:
            # Classification head
            self.classifier = nn.Linear(config.embedding_dim, num_classes)
        else:
            # Language modeling head
            self.lm_head = nn.Linear(config.embedding_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Position embedding
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        
        # Output projections
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        
        # Task-specific heads
        if hasattr(self, 'classifier'):
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.classifier.bias)
        elif hasattr(self, 'lm_head'):
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.lm_head.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_router_info: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MOCA-Net.
        
        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            attention_mask: [batch_size, seq_len] - Optional attention mask
            return_router_info: Whether to return routing information
            
        Returns:
            Dictionary containing outputs and optional routing info
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Position embeddings
        if seq_len <= self.position_embedding.shape[0]:
            pos_embeddings = self.position_embedding[:seq_len]
        else:
            # Extend position embeddings if needed
            # Use a simpler approach that maintains gradients
            pos_embeddings = F.interpolate(
                self.position_embedding.unsqueeze(0),  # [1, L, D]
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).squeeze(0)  # [L, D]
        
        # Combine token and position embeddings
        embeddings = token_embeddings + pos_embeddings
        
        # Apply attention mask if provided
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
        
        # Token routing
        (
            expert_weights,
            memory_weights,
            expert_mask,
            memory_mask,
            predicted_budget,
        ) = self.token_router(embeddings)
        
        # Memory read
        memory_output = self.memory_bank(
            query=embeddings,
            memory_weights=memory_weights,
            update_input=embeddings,  # Update memory with current tokens
        )
        
        # Expert processing
        expert_output = self.expert_layer(
            input_tensor=embeddings,
            expert_weights=expert_weights,
        )
        
        # Combine memory and expert outputs
        combined_output = memory_output + expert_output
        
        # Final processing
        output = self.output_proj(combined_output)
        output = self.dropout(output)
        output = self.layer_norm(output)
        
        # Task-specific output
        if hasattr(self, 'classifier'):
            # Classification
            # Use mean pooling over sequence
            if attention_mask is not None:
                pooled_output = (output * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled_output = output.mean(dim=1)
            logits = self.classifier(pooled_output)
        else:
            # Language modeling
            logits = self.lm_head(output)
        
        # Compute budget loss
        expert_usage = expert_mask.mean(dim=-1)  # [B, S]
        memory_usage = memory_mask.mean(dim=-1)  # [B, S]
        budget_loss = self.budget_loss(expert_usage, memory_usage, predicted_budget)
        
        # Prepare output
        outputs = {
            'logits': logits,
            'budget_loss': budget_loss,
            'expert_usage': expert_usage.mean(),
            'memory_usage': memory_usage.mean(),
        }
        
        if return_router_info:
            outputs.update({
                'expert_weights': expert_weights,
                'memory_weights': memory_weights,
                'expert_mask': expert_mask,
                'memory_mask': memory_mask,
                'predicted_budget': predicted_budget,
            })
        
        return outputs
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters by component."""
        counts = {}
        
        # Core components
        counts['token_embedding'] = sum(p.numel() for p in self.token_embedding.parameters())
        counts['position_embedding'] = self.position_embedding.numel()
        counts['token_router'] = sum(p.numel() for p in self.token_router.parameters())
        counts['memory_bank'] = sum(p.numel() for p in self.memory_bank.parameters())
        counts['expert_layer'] = sum(p.numel() for p in self.expert_layer.parameters())
        counts['output_proj'] = sum(p.numel() for p in self.output_proj.parameters())
        
        # Task-specific heads
        if hasattr(self, 'classifier'):
            counts['classifier'] = sum(p.numel() for p in self.classifier.parameters())
        elif hasattr(self, 'lm_head'):
            counts['lm_head'] = sum(p.numel() for p in self.lm_head.parameters())
        
        counts['total'] = sum(counts.values())
        return counts
    
    def get_memory_state(self) -> torch.Tensor:
        """Get current memory bank state."""
        return self.memory_bank.memory_values.detach().clone()
    
    def set_memory_state(self, memory_state: torch.Tensor):
        """Set memory bank state."""
        with torch.no_grad():
            self.memory_bank.memory_values.data = memory_state.clone()
