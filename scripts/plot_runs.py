#!/usr/bin/env python3
"""Plot training results and ablation studies for MOCA-Net."""

import os
import json
import argparse
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


def plot_training_curves(runs_dir: str, output_dir: str):
    """Plot training curves from training runs."""
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find training history files
    training_files = []
    for file_path in runs_path.rglob("*.pt"):
        if "final" in file_path.name or "best" in file_path.name:
            training_files.append(file_path)
    
    if not training_files:
        print("No training checkpoint files found.")
        return
    
    # Load training histories
    training_histories = []
    for file_path in training_files:
        try:
            import torch
            from torch.serialization import add_safe_globals
            
            # First try with weights_only=True and safe globals (PyTorch 2.6+ compatible)
            try:
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
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                except ImportError:
                    # If import fails, just try weights_only=True without safe globals
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            except Exception as safe_load_error:
                print(f"Safe loading failed for {file_path}, trying with weights_only=False: {safe_load_error}")
                # Fall back to weights_only=False for backward compatibility
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if 'training_history' in checkpoint:
                history = checkpoint['training_history']
                if history:
                    # Convert to DataFrame
                    df = pd.DataFrame(history)
                    df['run_name'] = file_path.stem
                    training_histories.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not training_histories:
        print("No training history data found.")
        return
    
    # Combine all histories
    combined_df = pd.concat(training_histories, ignore_index=True)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MOCA-Net Training Curves', fontsize=16)
    
    # Loss curves
    for run_name in combined_df['run_name'].unique():
        run_data = combined_df[combined_df['run_name'] == run_name]
        axes[0, 0].plot(run_data['step'], run_data['total_loss'], label=run_name, alpha=0.8)
        axes[0, 1].plot(run_data['step'], run_data['task_loss'], label=run_name, alpha=0.8)
    
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Task Loss')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Budget usage
    for run_name in combined_df['run_name'].unique():
        run_data = combined_df[combined_df['run_name'] == run_name]
        if 'expert_usage' in run_data.columns:
            axes[1, 0].plot(run_data['step'], run_data['expert_usage'], label=run_name, alpha=0.8)
        if 'memory_usage' in run_data.columns:
            axes[1, 1].plot(run_data['step'], run_data['memory_usage'], label=run_name, alpha=0.8)
    
    axes[1, 0].set_title('Expert Usage')
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Usage Fraction')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Memory Usage')
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Usage Fraction')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path / 'training_curves.png'}")


def plot_ablation_results(runs_dir: str, output_dir: str):
    """Plot ablation study results."""
    ablation_file = Path(runs_dir) / "ablation" / "ablation_results.json"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not ablation_file.exists():
        print("No ablation results found.")
        return
    
    # Load ablation results
    with open(ablation_file, 'r') as f:
        ablation_results = json.load(f)
    
    # Extract data for plotting
    ablation_names = []
    accuracies = []
    training_times = []
    model_sizes = []
    
    for ablation_name, results in ablation_results.items():
        ablation_names.append(ablation_name.replace('_', ' ').title())
        accuracies.append(results['metrics']['final_accuracy'])
        training_times.append(results['training_time'])
        model_sizes.append(results['model_size'])
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MOCA-Net Ablation Study Results', fontsize=16)
    
    # Accuracy comparison
    bars1 = axes[0].bar(ablation_names, accuracies, color='skyblue', alpha=0.8)
    axes[0].set_title('Final Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    bars2 = axes[1].bar(ablation_names, training_times, color='lightcoral', alpha=0.8)
    axes[1].set_title('Training Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Model size comparison
    bars3 = axes[2].bar(ablation_names, model_sizes, color='lightgreen', alpha=0.8)
    axes[2].set_title('Model Size')
    axes[2].set_ylabel('Parameters')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    for bar, size in zip(bars3, model_sizes):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + size*0.01,
                     f'{size:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ablation results saved to {output_path / 'ablation_results.png'}")
    
    # Create summary table
    summary_data = []
    for i, ablation_name in enumerate(ablation_names):
        summary_data.append({
            'Ablation': ablation_name,
            'Accuracy': f"{accuracies[i]:.4f}",
            'Training Time': f"{training_times[i]:.1f}s",
            'Model Size': f"{model_sizes[i]:,}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / 'ablation_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Ablation summary saved to {summary_file}")


def plot_evaluation_results(runs_dir: str, output_dir: str):
    """Plot evaluation results."""
    eval_dir = Path(runs_dir) / "eval"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not eval_dir.exists():
        print("No evaluation results found.")
        return
    
    # Find evaluation result files
    eval_files = list(eval_dir.glob("results_*.json"))
    
    if not eval_files:
        print("No evaluation result files found.")
        return
    
    # Load and plot evaluation results
    eval_results = {}
    for eval_file in eval_files:
        task_name = eval_file.stem.replace('results_', '')
        with open(eval_file, 'r') as f:
            eval_results[task_name] = json.load(f)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('MOCA-Net Evaluation Results', fontsize=16)
    
    # Accuracy comparison across tasks
    tasks = list(eval_results.keys())
    accuracies = [eval_results[task]['metrics']['accuracy'] for task in tasks]
    
    bars = axes[0].bar(tasks, accuracies, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0].set_title('Task Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')
    
    # Detailed metrics for classification task
    if 'text_classification' in eval_results:
        cls_metrics = eval_results['text_classification']['metrics']
        metric_names = ['precision', 'recall', 'f1_score']
        metric_values = [cls_metrics.get(metric, 0) for metric in metric_names]
        
        bars2 = axes[1].bar(metric_names, metric_values, color='lightgreen', alpha=0.8)
        axes[1].set_title('Text Classification Metrics')
        axes[1].set_ylabel('Score')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, metric_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation results saved to {output_path / 'evaluation_results.png'}")


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="Plot MOCA-Net training and evaluation results")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing training runs")
    parser.add_argument("--output-dir", default="runs/plots", help="Output directory for plots")
    parser.add_argument("--plot-type", choices=["all", "training", "ablation", "evaluation"], 
                       default="all", help="Type of plots to generate")
    
    args = parser.parse_args()
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    print(f"Generating plots from {args.runs_dir} to {args.output_dir}")
    
    if args.plot_type in ["all", "training"]:
        plot_training_curves(args.runs_dir, args.output_dir)
    
    if args.plot_type in ["all", "ablation"]:
        plot_ablation_results(args.runs_dir, args.output_dir)
    
    if args.plot_type in ["all", "evaluation"]:
        plot_evaluation_results(args.runs_dir, args.output_dir)
    
    print("Plotting completed!")


if __name__ == "__main__":
    main()
