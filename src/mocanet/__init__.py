"""MOCA-Net: Memory-Orchestrated Context Allocation Network."""

__version__ = "0.1.0"

from .config import Config, ModelConfig, TrainingConfig, DataConfig
from .model import MOCANet
from .layers import TokenRouter, MemoryBank, ExpertLayer
from .data import CopyTaskDataset, TextClassificationDataset
from .train import Trainer
from .eval import Evaluator

__all__ = [
    "Config",
    "ModelConfig", 
    "TrainingConfig",
    "DataConfig",
    "MOCANet",
    "TokenRouter",
    "MemoryBank",
    "ExpertLayer",
    "CopyTaskDataset",
    "TextClassificationDataset",
    "Trainer",
    "Evaluator",
]
