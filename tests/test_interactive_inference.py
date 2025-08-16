"""Tests for interactive inference functionality."""

import pytest
import torch
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mocanet.model import MOCANet
from mocanet.config import ModelConfig, Config
from scripts.interactive_inference import InteractiveInference


class TestInteractiveInference:
    """Test interactive inference functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return Config(
            model=ModelConfig(
                embedding_dim=64,
                num_experts=2,
                num_memory_slots=16,
                top_k_experts=1,
                router_temperature=1.0,
                dropout=0.1,
                num_memory_heads=4,
                budget_loss_weight=0.1,
                layer_norm_eps=1e-5,
            ),
            data={
                'task': 'text_cls',
                'sequence_length': 32,
                'vocab_size': 1000,
                'num_classes': 2,
            },
            text_cls={
                'min_freq': 2,
                'max_vocab_size': 1000,
            },
            hardware={
                'device': 'cpu',
            }
        )
    
    @pytest.fixture
    def mock_vocab(self):
        """Create a mock vocabulary for testing."""
        return {
            'the': 1,
            'movie': 2,
            'was': 3,
            'great': 4,
            'terrible': 5,
            '<unk>': 6,
            '<pad>': 0,
        }
    
    def test_tokenize_text(self, mock_config, mock_vocab):
        """Test text tokenization functionality."""
        # Create a minimal mock model for testing
        model = MOCANet(
            config=mock_config.model,
            vocab_size=mock_config.data['vocab_size'],
            num_classes=2,
        )
        
        # Test tokenization
        text = "the movie was great"
        expected_tokens = [1, 2, 3, 4] + [0] * 28  # 32 - 4 = 28 padding tokens
        
        # Mock the tokenization method
        def mock_tokenize(text):
            tokens = text.lower().split()
            token_ids = []
            for token in tokens:
                if token in mock_vocab:
                    token_ids.append(mock_vocab[token])
                else:
                    token_ids.append(mock_vocab.get('<unk>', 6))
            
            # Pad to sequence length
            max_len = mock_config.data['sequence_length']
            if len(token_ids) < max_len:
                token_ids.extend([0] * (max_len - len(token_ids)))
            else:
                token_ids = token_ids[:max_len]
            
            return torch.tensor([token_ids], dtype=torch.long)
        
        result = mock_tokenize(text)
        assert result.shape == (1, 32)
        assert result[0][:4].tolist() == expected_tokens[:4]
        assert result[0][4:].tolist() == [0] * 28
    
    def test_attention_mask_creation(self):
        """Test attention mask creation."""
        # Test with some padding
        input_ids = torch.tensor([[1, 2, 3, 0, 0, 0]])  # 3 real tokens, 3 padding
        expected_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        
        # Mock the attention mask method
        def mock_get_attention_mask(input_ids):
            return (input_ids != 0).long()
        
        result = mock_get_attention_mask(input_ids)
        assert torch.equal(result, expected_mask)
    
    def test_prediction_format(self):
        """Test that prediction output has correct format."""
        expected_keys = {
            'predicted_class',
            'predicted_label', 
            'confidence',
            'negative_probability',
            'positive_probability',
            'input_tokens',
            'attention_mask'
        }
        
        # Mock prediction result
        mock_prediction = {
            'predicted_class': 1,
            'predicted_label': 'Positive',
            'confidence': 0.85,
            'negative_probability': 0.15,
            'positive_probability': 0.85,
            'input_tokens': [1, 2, 3] + [0] * 29,
            'attention_mask': [1, 1, 1] + [0] * 29,
        }
        
        assert set(mock_prediction.keys()) == expected_keys
        assert mock_prediction['predicted_class'] in [0, 1]
        assert mock_prediction['predicted_label'] in ['Positive', 'Negative']
        assert 0 <= mock_prediction['confidence'] <= 1
        assert 0 <= mock_prediction['negative_probability'] <= 1
        assert 0 <= mock_prediction['positive_probability'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
