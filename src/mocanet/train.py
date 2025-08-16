"""Training module for MOCA-Net."""

import os
import time
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)

from .config import Config
from .model import MOCANet
from .data import DataManager
from .utils import set_seeds, get_device, save_checkpoint, load_checkpoint


class Trainer:
    """MOCA-Net trainer."""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.device = get_device(config.hardware.device)
        
        # Set seeds
        set_seeds(config.seeds)
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize data
        self.data_manager = DataManager(config)
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Training state
        self.global_step = 0
        self.best_val_metric = 0.0
        self.training_history = []
        
        # Initialize training iterator
        self.train_iter = iter(self.train_loader)
        
        # Create output directory
        os.makedirs("runs", exist_ok=True)
    
    def _create_model(self) -> MOCANet:
        """Create MOCA-Net model."""
        if self.config.data.task == "copy":
            num_classes = 1  # Language modeling
        else:
            num_classes = self.config.data.num_classes
        
        model = MOCANet(
            config=self.config.model,
            vocab_size=self.config.data.vocab_size,
            num_classes=num_classes,
        )
        
        # Print model info
        param_counts = model.count_parameters()
        self.console.print(f"[green]Model created with {param_counts['total']:,} parameters[/green]")
        
        for component, count in param_counts.items():
            if component != 'total':
                self.console.print(f"  {component}: {count:,}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.max_steps,
            eta_min=self.config.training.learning_rate * 0.1,
        )
    
    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create data loaders based on task."""
        if self.config.data.task == "copy":
            return self.data_manager.create_copy_task_data()
        else:
            return self.data_manager.create_text_classification_data()
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss for the current batch."""
        if self.config.data.task == "copy":
            # Language modeling loss
            logits = outputs['logits']  # [B, S, V]
            targets = targets.view(-1)  # [B*S]
            logits = logits.view(-1, logits.size(-1))  # [B*S, V]
            
            # Cross-entropy loss
            loss = nn.CrossEntropyLoss(ignore_index=0)(logits, targets)
        else:
            # Classification loss
            logits = outputs['logits']  # [B, C]
            loss = nn.CrossEntropyLoss()(logits, targets)
        
        # Add budget loss
        budget_loss = outputs['budget_loss']
        total_loss = loss + budget_loss
        
        return total_loss, loss, budget_loss
    
    def _compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        if self.config.data.task == "copy":
            # Copy task accuracy
            logits = outputs['logits']  # [B, S, V]
            predictions = logits.argmax(dim=-1)  # [B, S]
            
            # Compute accuracy (ignore padding tokens)
            mask = (targets != 0).float()
            correct = (predictions == targets).float() * mask
            accuracy = correct.sum() / mask.sum()
            
            return {"accuracy": accuracy.item()}
        else:
            # Classification accuracy
            logits = outputs['logits']  # [B, C]
            predictions = logits.argmax(dim=-1)  # [B]
            accuracy = (predictions == targets).float().mean()
            
            return {"accuracy": accuracy.item()}
    
    def _train_step(self, batch: tuple) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        if self.config.data.task == "copy":
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        else:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        total_loss, task_loss, budget_loss = self._compute_loss(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Debug gradients
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.gradient_clip_norm,
        )
        
        # Update parameters
        self.optimizer.step()
        self.scheduler.step()
        
        # Debug: check if loss becomes NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"Loss became NaN/Inf at step {self.global_step}")
            return {
                "total_loss": 0.0,
                "task_loss": 0.0,
                "budget_loss": 0.0,
                "expert_usage": outputs["expert_usage"].item(),
                "memory_usage": outputs["memory_usage"].item(),
                "gradient_norm": total_norm,
            }
        
        # Debug: log actual loss values
        if self.global_step % 10 == 0:  # Log every 10 steps
            logger.info(f"Step {self.global_step}: loss={total_loss.item():.6f}, task_loss={task_loss.item():.6f}, budget_loss={budget_loss.item():.6f}")
        
        # Return metrics
        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "budget_loss": budget_loss.item(),
            "expert_usage": outputs["expert_usage"].item(),
            "memory_usage": outputs["memory_usage"].item(),
            "gradient_norm": total_norm,
        }
    
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_metrics = {"accuracy": 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.config.data.task == "copy":
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute metrics
                batch_metrics = self._compute_metrics(outputs, targets)
                
                # Accumulate
                for key, value in batch_metrics.items():
                    total_metrics[key] += value
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return total_metrics
    
    def train(self):
        """Main training loop."""
        self.console.print(Panel.fit(
            f"[bold blue]Training MOCA-Net on {self.config.data.task}[/bold blue]\n"
            f"Device: {self.device}\n"
            f"Max steps: {self.config.training.max_steps}\n"
            f"Batch size: {self.config.training.batch_size}",
            title="Training Configuration"
        ))
        
        # Training progress
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            
            training_task = progress.add_task(
                "Training", total=self.config.training.max_steps
            )
            
            for step in range(self.config.training.max_steps):
                # Get batch
                try:
                    batch = next(self.train_iter)
                except (StopIteration, AttributeError):
                    self.train_iter = iter(self.train_loader)
                    batch = next(self.train_iter)
                
                # Training step
                step_metrics = self._train_step(batch)
                
                # Update progress
                progress.update(training_task, advance=1)
                
                # Log metrics
                if step % self.config.logging.log_every == 0:
                    self._log_metrics(step, step_metrics)
                
                # Validation
                if step % self.config.training.eval_every == 0:
                    val_metrics = self._validate()
                    self._log_validation(step, val_metrics)
                    
                    # Save best model
                    if val_metrics["accuracy"] > self.best_val_metric:
                        self.best_val_metric = val_metrics["accuracy"]
                        self._save_best_model()
                
                # Save checkpoint
                if step % self.config.training.save_every == 0:
                    self._save_checkpoint(step)
                
                # Increment global step
                self.global_step += 1
        
        # Final validation
        final_val_metrics = self._validate()
        self.console.print(f"\n[green]Training completed![/green]")
        self.console.print(f"Final validation accuracy: {final_val_metrics['accuracy']:.4f}")
        
        # Save final model
        self._save_checkpoint(self.config.training.max_steps, is_final=True)
    
    def _log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training metrics."""
        # Store in history
        self.training_history.append({
            "step": step,
            **metrics,
        })
        
        # Print to console
        metrics_str = " | ".join([
            f"{k}: {v:.4f}" for k, v in metrics.items()
        ])
        self.console.print(f"Step {step:4d}: {metrics_str}")
    
    def _log_validation(self, step: int, metrics: Dict[str, float]):
        """Log validation metrics."""
        self.console.print(f"\n[blue]Validation at step {step}:[/blue]")
        for key, value in metrics.items():
            self.console.print(f"  {key}: {value:.4f}")
    
    def _save_checkpoint(self, step: int, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_metric": self.best_val_metric,
            "training_history": self.training_history,
        }
        
        if is_final:
            filename = f"runs/mocanet_final.pt"
        else:
            filename = f"runs/mocanet_step_{step}.pt"
        
        save_checkpoint(checkpoint, filename)
        self.console.print(f"Checkpoint saved: {filename}")
    
    def _save_best_model(self):
        """Save best model based on validation metric."""
        checkpoint = {
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_val_metric": self.best_val_metric,
        }
        
        filename = "runs/mocanet_best.pt"
        save_checkpoint(checkpoint, filename)
        self.console.print(f"Best model saved: {filename}")


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig):
    """Main training function."""
    # Convert to Pydantic config
    config = Config(**cfg)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
