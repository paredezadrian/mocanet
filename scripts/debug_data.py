#!/usr/bin/env python3
"""Debug script for SST-2 data and model outputs."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.mocanet.sst2_dataset import SST2DataManager
from src.mocanet.config import Config
from src.mocanet.model import MOCANet
import hydra
from omegaconf import DictConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="text_cls")
def debug_data_and_model(cfg: DictConfig):
    """Debug SST-2 data loading and model outputs."""
    logger.info("Debugging SST-2 data and model...")
    
    try:
        # Load data
        sst2_manager = SST2DataManager(cfg)
        train_loader, val_loader, test_loader = sst2_manager.create_data_loaders()
        
        # Get a batch
        batch = next(iter(train_loader))
        inputs, targets = batch
        
        logger.info(f"Input shape: {inputs.shape}")
        logger.info(f"Target shape: {targets.shape}")
        logger.info(f"Input dtype: {inputs.dtype}")
        logger.info(f"Target dtype: {targets.dtype}")
        logger.info(f"Input range: [{inputs.min()}, {inputs.max()}]")
        logger.info(f"Target range: [{targets.min()}, {targets.max()}]")
        logger.info(f"Sample targets: {targets[:5]}")
        
        # Check vocabulary
        vocab_info = sst2_manager.get_vocab_info()
        logger.info(f"Vocabulary size: {vocab_info['vocab_size']}")
        logger.info(f"PAD token ID: {vocab_info['pad_token_id']}")
        
        # Create model
        model = MOCANet(
            config=cfg.model,
            vocab_size=cfg.data.vocab_size,
            num_classes=cfg.data.num_classes,
        )
        
        logger.info(f"Model created with {model.count_parameters()['total']:,} parameters")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            
        logger.info("Model outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if value.numel() > 0:
                    logger.info(f"    range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    logger.info(f"    mean: {value.mean().item():.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Test loss computation
        from src.mocanet.train import Trainer
        trainer = Trainer(cfg)
        
        # Compute loss manually
        total_loss, task_loss, budget_loss = trainer._compute_loss(outputs, targets)
        logger.info(f"Loss computation:")
        logger.info(f"  Total loss: {total_loss.item():.6f}")
        logger.info(f"  Task loss: {task_loss.item():.6f}")
        logger.info(f"  Budget loss: {budget_loss.item():.6f}")
        
        # Test metrics
        metrics = trainer._compute_metrics(outputs, targets)
        logger.info(f"Metrics: {metrics}")
        
        logger.info("✅ Debug completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    debug_data_and_model()
