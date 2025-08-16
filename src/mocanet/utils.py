"""Utility functions for MOCA-Net."""

import os
import random
import pickle
from typing import Dict, Any, Optional
import torch
import numpy as np


def set_seeds(seeds_config) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seeds_config.random)
    np.random.seed(seeds_config.numpy)
    torch.manual_seed(seeds_config.torch)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seeds_config.torch)
        torch.cuda.manual_seed_all(seeds_config.torch)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def save_checkpoint(checkpoint: Dict[str, Any], filename: str) -> None:
    """Save checkpoint to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)


def load_checkpoint(filename: str) -> Dict[str, Any]:
    """Load checkpoint from file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
    
    # First try with weights_only=True and safe globals (PyTorch 2.6+ compatible)
    try:
        from torch.serialization import add_safe_globals
        # Try to import and add safe globals for all our custom config classes
        try:
            from mocanet.config import (
                Config, ModelConfig, TrainingConfig, DataConfig, 
                LoggingConfig, HardwareConfig, SeedsConfig, 
                CopyTaskConfig, TextClassificationConfig
            )
            add_safe_globals([
                Config, ModelConfig, TrainingConfig, DataConfig,
                LoggingConfig, HardwareConfig, SeedsConfig,
                CopyTaskConfig, TextClassificationConfig
            ])
            checkpoint = torch.load(filename, map_location='cpu', weights_only=True)
        except ImportError:
            # If import fails, just try weights_only=True without safe globals
            checkpoint = torch.load(filename, map_location='cpu', weights_only=True)
    except Exception as safe_load_error:
        # Fall back to weights_only=False for backward compatibility
        checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp."""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def save_config(config: Any, filename: str) -> None:
    """Save configuration to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if hasattr(config, 'model_dump'):
        # Pydantic v2 config
        config_dict = config.model_dump()
    elif hasattr(config, 'dict'):
        # Pydantic v1 config (fallback)
        config_dict = config.dict()
    else:
        config_dict = config
    
    with open(filename, 'w') as f:
        import yaml
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config(filename: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(filename, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    return config


def set_deterministic_mode() -> None:
    """Set deterministic mode for PyTorch."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    if torch.cuda.is_available():
        return {
            'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'gpu_max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    else:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'cpu_rss': memory_info.rss / 1024**3,  # GB
            'cpu_vms': memory_info.vms / 1024**3,  # GB
        }


def log_memory_usage(logger=None) -> None:
    """Log current memory usage."""
    memory_info = get_memory_usage()
    
    if logger:
        for key, value in memory_info.items():
            logger.info(f"{key}: {value:.2f} GB")
    else:
        for key, value in memory_info.items():
            print(f"{key}: {value:.2f} GB")
