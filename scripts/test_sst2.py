#!/usr/bin/env python3
"""Test script for SST-2 dataset loading."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.mocanet.sst2_dataset import SST2Dataset, SST2DataManager
from src.mocanet.config import Config
import hydra
from omegaconf import DictConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="text_cls")
def test_sst2_loading(cfg: DictConfig):
    """Test SST-2 dataset loading."""
    logger.info("Testing SST-2 dataset loading...")
    
    try:
        # Test individual dataset loading
        logger.info("Testing individual dataset loading...")
        
        # Build vocabulary from training data
        train_dataset = SST2Dataset(
            split="train",
            max_length=cfg.data.sequence_length,
            build_vocab=True,
            min_freq=cfg.text_cls.min_freq,
            max_vocab_size=cfg.data.vocab_size,
        )
        
        logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
        logger.info(f"Vocabulary size: {train_dataset.vocab_size}")
        
        # Test validation dataset
        val_dataset = SST2Dataset(
            split="validation",
            max_length=cfg.data.sequence_length,
            vocab=train_dataset.vocab,
            build_vocab=False,
        )
        
        logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")
        
        # Test test dataset
        test_dataset = SST2Dataset(
            split="test",
            max_length=cfg.data.sequence_length,
            vocab=train_dataset.vocab,
            build_vocab=False,
        )
        
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        
        # Test data manager
        logger.info("Testing data manager...")
        sst2_manager = SST2DataManager(cfg)
        train_loader, val_loader, test_loader = sst2_manager.create_data_loaders()
        
        # Test a few batches
        logger.info("Testing data loading...")
        for i, (batch_texts, batch_labels) in enumerate(train_loader):
            if i >= 2:  # Just test first 2 batches
                break
            logger.info(f"Batch {i}: texts shape: {batch_texts.shape}, labels shape: {batch_labels.shape}")
            logger.info(f"Sample label: {batch_labels[0].item()}")
        
        # Test vocabulary info
        vocab_info = sst2_manager.get_vocab_info()
        logger.info(f"Vocabulary info: {vocab_info['vocab_size']} tokens")
        
        logger.info("✅ SST-2 dataset loading test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ SST-2 dataset loading test failed: {e}")
        raise


if __name__ == "__main__":
    test_sst2_loading()
