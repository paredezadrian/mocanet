"""Unit tests for MOCA-Net layers."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.mocanet.layers import TokenRouter, MemoryBank, ExpertLayer, BudgetLoss


class TestTokenRouter:
    """Test TokenRouter layer."""
    
    @pytest.fixture
    def router(self):
        return TokenRouter(
            embedding_dim=128,
            num_experts=4,
            num_memory_slots=32,
            top_k_experts=2,
            temperature=1.0,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 10, 128)  # [batch_size, seq_len, embedding_dim]
    
    def test_router_initialization(self, router):
        """Test router initialization."""
        assert router.embedding_dim == 128
        assert router.num_experts == 4
        assert router.num_memory_slots == 32
        assert router.top_k_experts == 2
        assert router.temperature == 1.0
    
    def test_router_forward(self, router, sample_input):
        """Test router forward pass."""
        outputs = router(sample_input)
        
        # Check output types and shapes
        expert_weights, memory_weights, expert_mask, memory_mask, budget_usage = outputs
        
        assert expert_weights.shape == (2, 10, 4)
        assert memory_weights.shape == (2, 10, 32)
        assert expert_mask.shape == (2, 10, 4)
        assert memory_mask.shape == (2, 10, 32)
        assert budget_usage.shape == (2, 10)
        
        # Check that weights sum to reasonable values
        assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-6)
        assert torch.allclose(memory_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-6)
        
        # Check that masks are binary
        assert torch.all((expert_mask == 0) | (expert_mask == 1))
        assert torch.all((memory_mask == 0) | (memory_mask == 1))
        
        # Check budget usage range
        assert torch.all((budget_usage >= 0) & (budget_usage <= 1))
    
    def test_router_sparsity(self, router, sample_input):
        """Test that router produces sparse outputs."""
        outputs = router(sample_input)
        expert_weights, memory_weights, expert_mask, memory_mask, _ = outputs
        
        # Check that only top-k experts are active
        active_experts = (expert_mask > 0).sum(dim=-1)
        assert torch.all(active_experts <= router.top_k_experts)
        
        # Check that memory usage is sparse
        active_memory = (memory_mask > 0).sum(dim=-1)
        max_memory = min(router.top_k_experts, router.num_memory_slots // 4)
        assert torch.all(active_memory <= max_memory)


class TestMemoryBank:
    """Test MemoryBank layer."""
    
    @pytest.fixture
    def memory_bank(self):
        return MemoryBank(
            num_slots=16,
            embedding_dim=128,
            num_heads=2,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_query(self):
        return torch.randn(2, 8, 128)  # [batch_size, seq_len, embedding_dim]
    
    @pytest.fixture
    def sample_weights(self):
        return torch.randn(2, 8, 16)  # [batch_size, seq_len, num_slots]
    
    def test_memory_bank_initialization(self, memory_bank):
        """Test memory bank initialization."""
        assert memory_bank.num_slots == 16
        assert memory_bank.embedding_dim == 128
        assert memory_bank.num_heads == 2
        assert memory_bank.head_dim == 64
    
    def test_memory_bank_forward(self, memory_bank, sample_query, sample_weights):
        """Test memory bank forward pass."""
        # Normalize weights to be valid probabilities
        sample_weights = torch.softmax(sample_weights, dim=-1)
        
        output = memory_bank(sample_query, sample_weights)
        
        assert output.shape == (2, 8, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_memory_bank_with_update(self, memory_bank, sample_query, sample_weights):
        """Test memory bank with memory updates."""
        # Normalize weights
        sample_weights = torch.softmax(sample_weights, dim=-1)
        
        # Test with update input
        update_input = torch.randn(2, 8, 128)
        output = memory_bank(sample_query, sample_weights, update_input)
        
        assert output.shape == (2, 8, 128)
        assert not torch.isnan(output).any()
    
    def test_memory_bank_parameters(self, memory_bank):
        """Test that memory bank has learnable parameters."""
        # Check that memory keys and values are learnable
        assert memory_bank.memory_keys.requires_grad
        assert memory_bank.memory_values.requires_grad
        
        # Check that update gate is learnable
        for param in memory_bank.update_gate.parameters():
            assert param.requires_grad


class TestExpertLayer:
    """Test ExpertLayer."""
    
    @pytest.fixture
    def expert_layer(self):
        return ExpertLayer(
            embedding_dim=128,
            num_experts=4,
            expert_hidden_dim=64,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 8, 128)  # [batch_size, seq_len, embedding_dim]
    
    @pytest.fixture
    def sample_weights(self):
        # Create sparse expert weights
        weights = torch.zeros(2, 8, 4)
        weights[:, :, :2] = 0.5  # Use first 2 experts
        return weights
    
    def test_expert_layer_initialization(self, expert_layer):
        """Test expert layer initialization."""
        assert expert_layer.embedding_dim == 128
        assert expert_layer.num_experts == 4
        assert expert_layer.expert_hidden_dim == 64
        assert len(expert_layer.experts) == 4
    
    def test_expert_layer_forward(self, expert_layer, sample_input, sample_weights):
        """Test expert layer forward pass."""
        output = expert_layer(sample_input, sample_weights)
        
        assert output.shape == (2, 8, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_expert_layer_sparsity(self, expert_layer, sample_input):
        """Test that expert layer respects sparsity."""
        # Create weights that only use one expert
        sparse_weights = torch.zeros(2, 8, 4)
        sparse_weights[:, :, 0] = 1.0  # Only use first expert
        
        output = expert_layer(sample_input, sparse_weights)
        
        assert output.shape == (2, 8, 128)
        # Output should be different from zero
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_expert_layer_parameters(self, expert_layer):
        """Test that expert layer has learnable parameters."""
        # Check that input/output projections are learnable
        assert expert_layer.input_proj.weight.requires_grad
        assert expert_layer.output_proj.weight.requires_grad
        
        # Check that experts are learnable
        for expert in expert_layer.experts:
            for param in expert.parameters():
                assert param.requires_grad


class TestBudgetLoss:
    """Test BudgetLoss module."""
    
    @pytest.fixture
    def budget_loss(self):
        return BudgetLoss(target_budget=0.3, weight=0.05)
    
    @pytest.fixture
    def sample_inputs(self):
        batch_size, seq_len = 4, 8
        expert_usage = torch.rand(batch_size, seq_len) * 0.5  # Random usage 0-0.5
        memory_usage = torch.rand(batch_size, seq_len) * 0.3  # Random usage 0-0.3
        predicted_budget = torch.rand(batch_size, seq_len) * 0.4  # Random prediction 0-0.4
        
        return expert_usage, memory_usage, predicted_budget
    
    def test_budget_loss_initialization(self, budget_loss):
        """Test budget loss initialization."""
        assert budget_loss.target_budget == 0.3
        assert budget_loss.weight == 0.05
    
    def test_budget_loss_forward(self, budget_loss, sample_inputs):
        """Test budget loss forward pass."""
        expert_usage, memory_usage, predicted_budget = sample_inputs
        
        loss = budget_loss(expert_usage, memory_usage, predicted_budget)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert loss >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_budget_loss_components(self, budget_loss, sample_inputs):
        """Test that budget loss has both components."""
        expert_usage, memory_usage, predicted_budget = sample_inputs
        
        # Test with high usage (should trigger budget violation penalty)
        high_usage = torch.ones_like(expert_usage) * 0.8
        high_loss = budget_loss(high_usage, memory_usage, predicted_budget)
        
        # Test with low usage (should have minimal penalty)
        low_usage = torch.ones_like(expert_usage) * 0.1
        low_loss = budget_loss(low_usage, memory_usage, predicted_budget)
        
        # High usage should result in higher loss
        assert high_loss > low_loss


class TestLayerIntegration:
    """Test integration between layers."""
    
    @pytest.fixture
    def sample_batch(self):
        batch_size, seq_len, embedding_dim = 2, 6, 64
        return torch.randn(batch_size, seq_len, embedding_dim)
    
    def test_router_memory_integration(self, sample_batch):
        """Test integration between router and memory bank."""
        # Create layers
        router = TokenRouter(
            embedding_dim=64,
            num_experts=3,
            num_memory_slots=8,
            top_k_experts=2,
        )
        
        memory_bank = MemoryBank(
            num_slots=8,
            embedding_dim=64,
            num_heads=1,
        )
        
        # Forward pass through router
        expert_weights, memory_weights, _, _, _ = router(sample_batch)
        
        # Forward pass through memory bank
        memory_output = memory_bank(sample_batch, memory_weights)
        
        assert memory_output.shape == sample_batch.shape
        assert not torch.isnan(memory_output).any()
    
    def test_router_expert_integration(self, sample_batch):
        """Test integration between router and expert layer."""
        # Create layers
        router = TokenRouter(
            embedding_dim=64,
            num_experts=3,
            num_memory_slots=8,
            top_k_experts=2,
        )
        
        expert_layer = ExpertLayer(
            embedding_dim=64,
            num_experts=3,
            expert_hidden_dim=32,
        )
        
        # Forward pass through router
        expert_weights, _, _, _, _ = router(sample_batch)
        
        # Forward pass through expert layer
        expert_output = expert_layer(sample_batch, expert_weights)
        
        assert expert_output.shape == sample_batch.shape
        assert not torch.isnan(expert_output).any()


if __name__ == "__main__":
    pytest.main([__file__])
