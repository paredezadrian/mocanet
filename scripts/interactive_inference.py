#!/usr/bin/env python3
"""Interactive inference script for MOCA-Net SST-2 sentiment analysis."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
import argparse

from src.mocanet.model import MOCANet
from src.mocanet.sst2_dataset import SST2Dataset
from src.mocanet.utils import load_checkpoint, get_device
from src.mocanet.config import Config


class InteractiveInference:
    """Interactive inference interface for MOCA-Net SST-2 model."""
    
    def __init__(self, checkpoint_path: str, config_path: str = "configs/text_cls.yaml"):
        self.console = Console()
        self.checkpoint_path = checkpoint_path
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.device = get_device(self.config.hardware.device)
        
        # Load model and vocabulary
        self.model, self.vocab, self.idx_to_label = self._load_model_and_vocab()
        
        self.console.print(Panel.fit(
            "[bold blue]MOCA-Net Interactive Sentiment Analysis[/bold blue]\n"
            f"Model: {os.path.basename(checkpoint_path)}\n"
            f"Device: {self.device}\n"
            f"Vocabulary Size: {len(self.vocab)}",
            title="Model Loaded Successfully"
        ))
    
    def _load_config(self, config_path: str) -> Config:
        """Load configuration from file."""
        from src.mocanet.utils import load_config
        
        config_dict = load_config(config_path)
        return Config(**config_dict)
    
    def _load_model_and_vocab(self):
        """Load trained model and vocabulary."""
        # Load checkpoint
        checkpoint = load_checkpoint(self.checkpoint_path)
        checkpoint_config = checkpoint['config']
        
        # Create model
        model = MOCANet(
            config=checkpoint_config.model,
            vocab_size=checkpoint_config.data.vocab_size,
            num_classes=2,  # SST-2 is binary classification
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load vocabulary from training dataset
        train_dataset = SST2Dataset(
            split="train",
            max_length=self.config.data.sequence_length,
            build_vocab=True,
            min_freq=self.config.text_cls.min_freq,
            max_vocab_size=self.config.data.vocab_size,
        )
        
        vocab = train_dataset.vocab
        idx_to_label = {0: "Negative", 1: "Positive"}
        
        self.console.print(f"[green]✓ Model loaded from checkpoint (step {checkpoint['step']})[/green]")
        self.console.print(f"[green]✓ Vocabulary loaded ({len(vocab)} tokens)[/green]")
        
        return model, vocab, idx_to_label
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize input text and convert to tensor."""
        # Simple tokenization (split on whitespace and convert to lowercase)
        tokens = text.lower().split()
        
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Use unknown token ID if token not in vocabulary
                token_ids.append(self.vocab.get('<unk>', 1))
        
        # Pad or truncate to sequence length
        max_len = self.config.data.sequence_length
        if len(token_ids) < max_len:
            token_ids.extend([0] * (max_len - len(token_ids)))  # Pad with 0
        else:
            token_ids = token_ids[:max_len]  # Truncate
        
        # Convert to tensor and add batch dimension
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)
    
    def _get_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask for non-padded tokens."""
        return (input_ids != 0).long()
    
    def predict_sentiment(self, text: str) -> dict:
        """Predict sentiment for input text."""
        # Tokenize input
        input_ids = self._tokenize_text(text)
        attention_mask = self._get_attention_mask(input_ids)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = F.softmax(logits, dim=-1)
            
            # Get prediction and confidence
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get probabilities for both classes
            neg_prob = probabilities[0][0].item()
            pos_prob = probabilities[0][1].item()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.idx_to_label[predicted_class],
            'confidence': confidence,
            'negative_probability': neg_prob,
            'positive_probability': pos_prob,
            'input_tokens': input_ids[0].tolist(),
            'attention_mask': attention_mask[0].tolist()
        }
    
    def display_prediction(self, text: str, prediction: dict):
        """Display prediction results in a formatted way."""
        # Create result table
        table = Table(title="Sentiment Analysis Result")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Input Text", f'"{text}"')
        table.add_row("Predicted Sentiment", f"[bold]{prediction['predicted_label']}[/bold]")
        table.add_row("Confidence", f"{prediction['confidence']:.3f}")
        table.add_row("Negative Probability", f"{prediction['negative_probability']:.3f}")
        table.add_row("Positive Probability", f"{prediction['positive_probability']:.3f}")
        
        self.console.print(table)
        
        # Add confidence indicator
        confidence_level = "High" if prediction['confidence'] > 0.8 else "Medium" if prediction['confidence'] > 0.6 else "Low"
        confidence_color = "green" if confidence_level == "High" else "yellow" if confidence_level == "Medium" else "red"
        
        self.console.print(f"\nConfidence Level: [{confidence_color}]{confidence_level}[/{confidence_color}]")
    
    def run_interactive(self):
        """Run the interactive inference loop."""
        self.console.print("\n[bold blue]Interactive Sentiment Analysis Started[/bold blue]")
        self.console.print("Type 'quit' to exit, 'help' for commands, or enter any sentence to analyze.\n")
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[cyan]Enter text[/cyan]")
                
                if user_input.lower() == 'quit':
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_model_stats()
                    continue
                
                elif not user_input.strip():
                    self.console.print("[yellow]Please enter some text to analyze.[/yellow]")
                    continue
                
                # Make prediction
                self.console.print("\n[blue]Analyzing sentiment...[/blue]")
                prediction = self.predict_sentiment(user_input)
                
                # Display results
                self.display_prediction(user_input, prediction)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted by user. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                self.console.print(f"\n[red]Error during prediction: {e}[/red]")
    
    def _show_help(self):
        """Show available commands."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan")
        help_table.add_column("Description", style="white")
        
        help_table.add_row("quit", "Exit the interactive session")
        help_table.add_row("help", "Show this help message")
        help_table.add_row("stats", "Show model statistics and configuration")
        help_table.add_row("<text>", "Analyze sentiment of the given text")
        
        self.console.print(help_table)
    
    def _show_model_stats(self):
        """Show model statistics and configuration."""
        stats_table = Table(title="Model Statistics")
        stats_table.add_column("Property", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Model Architecture", "MOCA-Net")
        stats_table.add_row("Task", "Binary Sentiment Classification")
        stats_table.add_row("Dataset", "Stanford SST-2")
        stats_table.add_row("Vocabulary Size", str(len(self.vocab)))
        stats_table.add_row("Sequence Length", str(self.config.data.sequence_length))
        stats_table.add_row("Embedding Dimension", str(self.config.model.embedding_dim))
        stats_table.add_row("Number of Experts", str(self.config.model.num_experts))
        stats_table.add_row("Memory Slots", str(self.config.model.num_memory_slots))
        
        self.console.print(stats_table)


def main():
    """Main function for interactive inference."""
    parser = argparse.ArgumentParser(description="Interactive inference with MOCA-Net SST-2 model")
    parser.add_argument("checkpoint", help="Path to model checkpoint file")
    parser.add_argument("--config", default="configs/text_cls.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)
    
    try:
        # Initialize interactive inference
        inference = InteractiveInference(args.checkpoint, args.config)
        
        # Run interactive loop
        inference.run_interactive()
        
    except Exception as e:
        print(f"Error initializing inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
