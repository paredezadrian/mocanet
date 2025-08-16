"""Smoke tests for MOCA-Net to verify basic functionality."""

import pytest
import torch
import torch.nn as nn

from src.mocanet.config import Config, ModelConfig
from src.mocanet.layers import TokenRouter, MemoryBank, ExpertLayer
from src.mocanet.model import MOCANet


def test_config_creation():
    """Test that config objects can be created."""
    config = Config()
    assert isinstance(config.model, ModelConfig)
    assert config.model.embedding_dim == 128
    assert config.model.num_experts == 4


def test_layers_creation():
    """Test that layer objects can be created."""
    # Token Router
    router = TokenRouter(
        embedding_dim=64,
        num_experts=3,
        num_memory_slots=8,
        top_k_experts=2,
    )
    assert router is not None
    
    # Memory Bank
    memory = MemoryBank(
        num_slots=8,
        embedding_dim=64,
        num_heads=1,
    )
    assert memory is not None
    
    # Expert Layer
    experts = ExpertLayer(
        embedding_dim=64,
        num_experts=3,
        expert_hidden_dim=32,
    )
    assert experts is not None


def test_model_creation():
    """Test that MOCA-Net model can be created."""
    config = Config()
    config.model.embedding_dim = 64
    config.model.num_experts = 2
    config.model.num_memory_slots = 8
    
    model = MOCANet(
        config=config.model,
        vocab_size=128,
        num_classes=1,
    )
    
    assert model is not None
    assert isinstance(model, MOCANet)


def test_model_forward_pass():
    """Test that model can perform a forward pass."""
    config = Config()
    config.model.embedding_dim = 64
    config.model.num_experts = 2
    config.model.num_memory_slots = 8
    
    model = MOCANet(
        config=config.model,
        vocab_size=128,
        num_classes=1,
    )
    
    # Create sample input
    batch_size, seq_len = 2, 6
    input_ids = torch.randint(1, 128, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Check outputs
    assert 'logits' in outputs
    assert 'budget_loss' in outputs
    assert 'expert_usage' in outputs
    assert 'memory_usage' in outputs
    
    # Check shapes
    assert outputs['logits'].shape == (batch_size, seq_len, 128)
    assert outputs['budget_loss'].shape == ()
    assert outputs['expert_usage'].shape == ()
    assert outputs['memory_usage'].shape == ()


def test_model_parameter_count():
    """Test that model has reasonable number of parameters."""
    config = Config()
    config.model.embedding_dim = 64
    config.model.num_experts = 2
    config.model.num_memory_slots = 8
    
    model = MOCANet(
        config=config.model,
        vocab_size=128,
        num_classes=1,
    )
    
    param_counts = model.count_parameters()
    
    # Check that we have parameters
    assert param_counts['total'] > 0
    
    # Check that total is reasonable (should be < 1M for this config)
    assert param_counts['total'] < 1_000_000
    
    # Check individual components
    assert param_counts['token_embedding'] > 0
    assert param_counts['token_router'] > 0
    assert param_counts['memory_bank'] > 0
    assert param_counts['expert_layer'] > 0


def test_model_memory_operations():
    """Test memory bank operations."""
    config = Config()
    config.model.embedding_dim = 64
    config.model.num_experts = 2
    config.model.num_memory_slots = 8
    
    model = MOCANet(
        config=config.model,
        vocab_size=128,
        num_classes=1,
    )
    
    # Get initial memory state
    initial_memory = model.get_memory_state()
    assert initial_memory.shape == (8, 64)
    
    # Set new memory state
    new_memory = torch.randn(8, 64)
    model.set_memory_state(new_memory)
    
    # Verify memory was updated
    updated_memory = model.get_memory_state()
    assert torch.allclose(updated_memory, new_memory)


def test_training_step_simulation():
    """Test that we can simulate a training step."""
    config = Config()
    config.model.embedding_dim = 64
    config.model.num_experts = 2
    config.model.num_memory_slots = 8
    
    model = MOCANet(
        config=config.model,
        vocab_size=128,
        num_classes=1,
    )
    
    # Create sample batch
    batch_size, seq_len = 2, 6
    input_ids = torch.randint(1, 128, (batch_size, seq_len))
    targets = torch.randint(1, 128, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    
    # Compute loss
    logits = outputs['logits']
    targets_flat = targets.view(-1)
    logits_flat = logits.view(-1, logits.size(-1))
    
    task_loss = nn.CrossEntropyLoss(ignore_index=0)(logits_flat, targets_flat)
    budget_loss = outputs['budget_loss']
    
    total_loss = task_loss + budget_loss
    
    # Check that loss is finite
    assert torch.isfinite(total_loss)
    assert total_loss > 0
    
    # Simulate backward pass
    total_loss.backward()
    
    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"Parameter {name} has no gradients")
                print(f"Shape: {param.shape}")
                print(f"Value range: {param.min().item():.4f} to {param.max().item():.4f}")
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
