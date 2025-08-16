"""Configuration classes for MOCA-Net."""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    embedding_dim: int = Field(128, description="Token embedding dimension")
    num_experts: int = Field(4, description="Number of expert networks")
    num_memory_slots: int = Field(32, description="Number of memory slots")
    top_k_experts: int = Field(2, description="Top-k experts to route to")
    num_memory_heads: int = Field(1, description="Number of memory read heads")
    router_temperature: float = Field(1.0, description="Router temperature for sparsity")
    budget_loss_weight: float = Field(0.05, description="Weight for budget loss term")
    dropout: float = Field(0.1, description="Dropout rate")
    layer_norm_eps: float = Field(1e-5, description="LayerNorm epsilon")


class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = Field(32, description="Training batch size")
    max_steps: int = Field(5000, description="Maximum training steps")
    learning_rate: float = Field(1e-3, description="Learning rate")
    weight_decay: float = Field(1e-4, description="Weight decay")
    warmup_steps: int = Field(200, description="Warmup steps")
    gradient_clip_norm: float = Field(1.0, description="Gradient clipping norm")
    save_every: int = Field(1000, description="Save checkpoint every N steps")
    eval_every: int = Field(500, description="Evaluate every N steps")


class DataConfig(BaseModel):
    """Data configuration."""
    task: Literal["copy", "text_classification"] = Field("copy", description="Task type")
    sequence_length: int = Field(60, description="Sequence length")
    vocab_size: int = Field(128, description="Vocabulary size")
    num_classes: int = Field(2, description="Number of classes for classification")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_every: int = Field(100, description="Log every N steps")
    tensorboard: bool = Field(True, description="Enable tensorboard logging")
    rich_progress: bool = Field(True, description="Use rich progress bars")


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    device: Literal["auto", "cpu", "cuda"] = Field("auto", description="Device to use")
    num_workers: int = Field(0, description="Number of data loader workers")
    pin_memory: bool = Field(False, description="Pin memory for faster transfer")


class SeedsConfig(BaseModel):
    """Random seeds configuration."""
    random: int = Field(42, description="Python random seed")
    torch: int = Field(42, description="PyTorch seed")
    numpy: int = Field(42, description="NumPy seed")


class CopyTaskConfig(BaseModel):
    """Copy task specific configuration."""
    min_length: int = Field(10, description="Minimum sequence length")
    max_length: int = Field(60, description="Maximum sequence length")
    delay_range: List[int] = Field([3, 8], description="Range for random delay tokens")


class TextClassificationConfig(BaseModel):
    """Text classification specific configuration."""
    dataset: str = Field("sst2_tiny", description="Dataset name")
    max_samples: int = Field(10000, description="Maximum number of samples")
    min_freq: int = Field(2, description="Minimum token frequency")
    pretrained_embeddings: bool = Field(False, description="Use pretrained embeddings")


class Config(BaseModel):
    """Main configuration class."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    copy_task: Optional[CopyTaskConfig] = Field(default_factory=CopyTaskConfig)
    text_cls: Optional[TextClassificationConfig] = Field(default_factory=TextClassificationConfig)
