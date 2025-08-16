"""Ablation studies for MOCA-Net."""

import os
import time
from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import numpy as np
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from .config import Config
from .model import MOCANet
from .data import DataManager
from .utils import set_seeds, get_device, save_checkpoint


class AblationStudy:
    """Run ablation studies on MOCA-Net components."""
    
    def __init__(self, config: Config):
        self.config = config
        self.console = Console()
        self.device = get_device(config.hardware.device)
        
        # Set seeds
        set_seeds(config.seeds)
        
        # Create output directory
        os.makedirs("runs/ablation", exist_ok=True)
        
        # Results storage
        self.ablation_results = {}
    
    def run_ablation_studies(self):
        """Run all ablation studies."""
        self.console.print(Panel.fit(
            "[bold blue]Running MOCA-Net Ablation Studies[/bold blue]\n"
            "Comparing different model configurations and component ablations",
            title="Ablation Configuration"
        ))
        
        # Run different ablations
        ablations = [
            ("baseline", "Baseline MOCA-Net"),
            ("no_memory", "No External Memory"),
            ("no_experts", "No Mixture of Experts"),
            ("dense_routing", "Dense Routing (No Sparsity)"),
            ("smaller_model", "Smaller Model (Half Size)"),
        ]
        
        for ablation_name, description in ablations:
            self.console.print(f"\n[blue]Running: {description}[/blue]")
            self._run_single_ablation(ablation_name, description)
        
        # Save results
        self._save_ablation_results()
        
        # Print summary
        self._print_ablation_summary()
    
    def _run_single_ablation(self, ablation_name: str, description: str):
        """Run a single ablation study."""
        # Create modified config
        modified_config = self._create_ablation_config(ablation_name)
        
        # Create model
        model = self._create_ablation_model(ablation_name, modified_config)
        model.to(self.device)
        
        # Create data
        data_manager = DataManager(modified_config)
        train_loader, val_loader = data_manager.create_copy_task_data()
        
        # Quick training run
        start_time = time.time()
        metrics = self._quick_train_and_eval(
            model, train_loader, val_loader, modified_config
        )
        training_time = time.time() - start_time
        
        # Store results
        self.ablation_results[ablation_name] = {
            "description": description,
            "config": modified_config.model_dump() if hasattr(modified_config, 'model_dump') else str(modified_config),
            "metrics": metrics,
            "training_time": training_time,
            "model_size": self._count_model_parameters(model),
        }
        
        self.console.print(f"  ✓ Completed in {training_time:.1f}s")
        self.console.print(f"  ✓ Final accuracy: {metrics['final_accuracy']:.4f}")
    
    def _create_ablation_config(self, ablation_name: str) -> Config:
        """Create modified config for ablation study."""
        # Start with base config
        config_dict = self.config.model_dump() if hasattr(self.config, 'model_dump') else self.config
        
        if ablation_name == "no_memory":
            # Disable memory bank
            config_dict['model']['num_memory_slots'] = 0
            config_dict['model']['num_memory_heads'] = 0
            
        elif ablation_name == "no_experts":
            # Disable mixture of experts
            config_dict['model']['num_experts'] = 1
            config_dict['model']['top_k_experts'] = 1
            
        elif ablation_name == "dense_routing":
            # Use dense routing instead of sparse
            config_dict['model']['top_k_experts'] = config_dict['model']['num_experts']
            
        elif ablation_name == "smaller_model":
            # Reduce model size
            config_dict['model']['embedding_dim'] = config_dict['model']['embedding_dim'] // 2
            config_dict['model']['num_experts'] = max(2, config_dict['model']['num_experts'] // 2)
            config_dict['model']['num_memory_slots'] = max(16, config_dict['model']['num_memory_slots'] // 2)
        
        # Convert back to Config object
        return Config(**config_dict)
    
    def _create_ablation_model(self, ablation_name: str, config: Config) -> MOCANet:
        """Create model for ablation study."""
        if ablation_name == "no_memory":
            # Create a modified model without memory bank
            return self._create_model_without_memory(config)
        else:
            # Use standard MOCA-Net
            return MOCANet(
                config=config.model,
                vocab_size=config.data.vocab_size,
                num_classes=1,  # Copy task
            )
    
    def _create_model_without_memory(self, config: Config) -> MOCANet:
        """Create MOCA-Net without external memory."""
        # This is a simplified version - in practice you'd modify the model class
        # For now, we'll use a very small memory bank
        config.model.num_memory_slots = 1
        config.model.num_memory_heads = 1
        
        return MOCANet(
            config=config.model,
            vocab_size=config.data.vocab_size,
            num_classes=1,
        )
    
    def _quick_train_and_eval(
        self,
        model: MOCANet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ) -> Dict[str, float]:
        """Quick training and evaluation for ablation study."""
        # Reduced training for quick comparison
        max_steps = min(500, config.training.max_steps)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Training loop
        model.train()
        step_metrics = []
        
        train_iter = iter(train_loader)
        
        for step in range(max_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            if config.data.task == "copy":
                logits = outputs['logits']
                targets_flat = targets.view(-1)
                logits_flat = logits.view(-1, logits.size(-1))
                loss = nn.CrossEntropyLoss(ignore_index=0)(logits_flat, targets_flat)
            else:
                logits = outputs['logits']
                loss = nn.CrossEntropyLoss()(logits, targets)
            
            # Add budget loss
            total_loss = loss + outputs['budget_loss']
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Store metrics every 100 steps
            if step % 100 == 0:
                step_metrics.append({
                    "step": step,
                    "loss": total_loss.item(),
                    "accuracy": self._compute_batch_accuracy(outputs, targets),
                })
        
        # Final evaluation
        model.eval()
        final_accuracy = self._evaluate_model(model, val_loader)
        
        return {
            "final_accuracy": final_accuracy,
            "final_loss": step_metrics[-1]["loss"] if step_metrics else 0.0,
            "step_metrics": step_metrics,
        }
    
    def _compute_batch_accuracy(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> float:
        """Compute accuracy for a single batch."""
        logits = outputs['logits']
        predictions = logits.argmax(dim=-1)
        
        # For copy task, ignore padding tokens
        mask = (targets != 0).float()
        correct = (predictions == targets).float() * mask
        accuracy = correct.sum() / mask.sum()
        
        return accuracy.item()
    
    def _evaluate_model(self, model: MOCANet, val_loader: DataLoader) -> float:
        """Evaluate model on validation set."""
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                accuracy = self._compute_batch_accuracy(outputs, targets)
                
                total_accuracy += accuracy
                num_batches += 1
        
        return total_accuracy / num_batches if num_batches > 0 else 0.0
    
    def _count_model_parameters(self, model: MOCANet) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _save_ablation_results(self):
        """Save ablation study results."""
        filename = "runs/ablation/ablation_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        results_copy = {}
        for key, value in self.ablation_results.items():
            if isinstance(value, dict):
                results_copy[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.integer):
                        results_copy[key][k] = int(v)
                    elif isinstance(v, np.floating):
                        results_copy[key][k] = float(v)
                    else:
                        results_copy[key][k] = v
            else:
                results_copy[key] = value
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        self.console.print(f"\n[green]Ablation results saved to {filename}[/green]")
    
    def _print_ablation_summary(self):
        """Print summary of all ablation studies."""
        table = Table(title="Ablation Study Results")
        table.add_column("Ablation", style="cyan")
        table.add_column("Description", style="blue")
        table.add_column("Accuracy", style="magenta")
        table.add_column("Training Time", style="green")
        table.add_column("Model Size", style="yellow")
        
        for ablation_name, results in self.ablation_results.items():
            table.add_row(
                ablation_name.replace('_', ' ').title(),
                results['description'],
                f"{results['metrics']['final_accuracy']:.4f}",
                f"{results['training_time']:.1f}s",
                f"{results['model_size']:,}",
            )
        
        self.console.print(table)
        
        # Find best performing ablation
        best_ablation = max(
            self.ablation_results.items(),
            key=lambda x: x[1]['metrics']['final_accuracy']
        )
        
        self.console.print(f"\n[green]Best performing ablation:[/green] {best_ablation[0]}")
        self.console.print(f"Accuracy: {best_ablation[1]['metrics']['final_accuracy']:.4f}")


@hydra.main(version_base=None, config_path="../../configs", config_name="copy_task")
def main(cfg: DictConfig):
    """Main ablation function."""
    # Convert to Pydantic config
    config = Config(**cfg)
    
    # Run ablation studies
    ablation_study = AblationStudy(config)
    ablation_study.run_ablation_studies()


if __name__ == "__main__":
    main()
