"""Evaluation module for MOCA-Net."""

import os
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import Config
from .model import MOCANet
from .data import DataManager
from .utils import get_device, load_checkpoint


class Evaluator:
    """MOCA-Net evaluator."""
    
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.console = Console()
        self.device = get_device(config.hardware.device)
        
        # Load model from checkpoint
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize data
        self.data_manager = DataManager(config)
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Create output directory
        os.makedirs("runs/eval", exist_ok=True)
    
    def _load_model(self, checkpoint_path: str) -> MOCANet:
        """Load model from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Use config from checkpoint to ensure model architecture matches
        checkpoint_config = checkpoint['config']
        
        # Create model using checkpoint config
        if self.config.data.task == "copy":
            num_classes = 1
        else:
            num_classes = self.config.data.num_classes
        
        model = MOCANet(
            config=checkpoint_config.model,
            vocab_size=checkpoint_config.data.vocab_size,
            num_classes=num_classes,
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.console.print(f"[green]Model loaded from {checkpoint_path}[/green]")
        self.console.print(f"Checkpoint step: {checkpoint['step']}")
        if 'best_val_metric' in checkpoint:
            self.console.print(f"Best validation metric: {checkpoint['best_val_metric']:.4f}")
        
        # Update our config to match the checkpoint config
        self.config = checkpoint_config
        
        return model
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders based on task."""
        if self.config.data.task == "copy":
            return self.data_manager.create_copy_task_data()
        else:
            return self.data_manager.create_text_classification_data()
    
    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation."""
        self.console.print(Panel.fit(
            f"[bold blue]Evaluating MOCA-Net on {self.config.data.task}[/bold blue]\n"
            f"Device: {self.device}",
            title="Evaluation Configuration"
        ))
        
        # Evaluate on validation set
        val_metrics = self._evaluate_validation()
        
        # Task-specific evaluation
        if self.config.data.task == "copy":
            copy_metrics = self._evaluate_copy_task()
            val_metrics.update(copy_metrics)
        else:
            cls_metrics = self._evaluate_classification()
            val_metrics.update(cls_metrics)
        
        # Print results
        self._print_results(val_metrics)
        
        # Save results
        self._save_results(val_metrics)
        
        return val_metrics
    
    def _evaluate_validation(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.console.print("\n[blue]Running validation evaluation...[/blue]")
        
        total_metrics = {"accuracy": 0.0}
        all_predictions = []
        all_targets = []
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
                batch_metrics = self._compute_batch_metrics(outputs, targets)
                
                # Accumulate
                for key, value in batch_metrics.items():
                    if key in total_metrics:
                        total_metrics[key] += value
                
                # Store predictions and targets for detailed analysis
                if self.config.data.task == "copy":
                    predictions = outputs['logits'].argmax(dim=-1)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                else:
                    predictions = outputs['logits'].argmax(dim=-1)
                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                
                num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        # Store for detailed analysis
        self.all_predictions = torch.cat(all_predictions, dim=0)
        self.all_targets = torch.cat(all_targets, dim=0)
        
        return total_metrics
    
    def _compute_batch_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute metrics for a single batch."""
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
    
    def _evaluate_copy_task(self) -> Dict[str, float]:
        """Evaluate copy task specific metrics."""
        self.console.print("\n[blue]Computing copy task metrics...[/blue]")
        
        predictions = self.all_predictions
        targets = self.all_targets
        
        # Sequence-level accuracy
        seq_correct = 0
        total_seqs = 0
        
        for pred_seq, target_seq in zip(predictions, targets):
            # Find non-padding tokens
            non_pad_mask = target_seq != 0
            if non_pad_mask.sum() > 0:
                # Check if the sequence is correctly copied
                pred_non_pad = pred_seq[non_pad_mask]
                target_non_pad = target_seq[non_pad_mask]
                
                if torch.equal(pred_non_pad, target_non_pad):
                    seq_correct += 1
                total_seqs += 1
        
        seq_accuracy = seq_correct / total_seqs if total_seqs > 0 else 0.0
        
        # Token-level accuracy (excluding padding)
        non_pad_mask = targets != 0
        token_correct = (predictions == targets).float() * non_pad_mask.float()
        token_accuracy = token_correct.sum() / non_pad_mask.sum()
        
        return {
            "sequence_accuracy": seq_accuracy,
            "token_accuracy": token_accuracy.item(),
        }
    
    def _evaluate_classification(self) -> Dict[str, float]:
        """Evaluate classification specific metrics."""
        self.console.print("\n[blue]Computing classification metrics...[/blue]")
        
        predictions = self.all_predictions
        targets = self.all_targets
        
        # Convert to numpy for sklearn
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        # Confusion matrix
        cm = confusion_matrix(target_np, pred_np)
        
        # Classification report
        report = classification_report(target_np, pred_np, output_dict=True)
        
        # Extract metrics
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(cm, "runs/eval/confusion_matrix.png")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.console.print(f"[green]Confusion matrix saved to {save_path}[/green]")
    
    def _print_results(self, metrics: Dict[str, float]):
        """Print evaluation results."""
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
            else:
                table.add_row(metric.replace('_', ' ').title(), str(value))
        
        self.console.print(table)
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save evaluation results to file."""
        import json
        
        results = {
            "task": self.config.data.task,
            "metrics": metrics,
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else str(self.config),
        }
        
        filename = f"runs/eval/results_{self.config.data.task}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.console.print(f"[green]Results saved to {filename}[/green]")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MOCA-Net model")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file path")
    parser.add_argument("--task", choices=["copy", "text_cls"], help="Task to evaluate")
    
    args = parser.parse_args()
    
    # Load config
    from .utils import load_config
    config_dict = load_config(args.config)
    
    # Override task if specified
    if args.task:
        config_dict['data']['task'] = args.task
    
    # Convert to Pydantic config
    from .config import Config
    config = Config(**config_dict)
    
    # Run evaluation
    evaluator = Evaluator(config, args.checkpoint)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
