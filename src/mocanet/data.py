"""Data loading and preprocessing for MOCA-Net tasks."""

import os
import random
import string
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from .sst2_dataset import SST2DataManager

logger = logging.getLogger(__name__)


class CopyTaskDataset(Dataset):
    """Dataset for the copy/recall task."""
    
    def __init__(
        self,
        num_samples: int,
        min_length: int,
        max_length: int,
        vocab_size: int,
        delay_range: Tuple[int, int],
        sequence_length: int,
    ):
        self.num_samples = num_samples
        self.min_length = min_length
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.delay_range = delay_range
        self.sequence_length = sequence_length
        
        # Generate samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate copy task samples."""
        samples = []
        
        for _ in range(self.num_samples):
            # Random sequence length
            seq_len = random.randint(self.min_length, self.max_length)
            
            # Random delay
            delay = random.randint(*self.delay_range)
            
            # Generate random input sequence
            input_seq = torch.randint(1, self.vocab_size - 1, (seq_len,))
            
            # Create target sequence with delay
            target_seq = torch.zeros(self.sequence_length, dtype=torch.long)
            
            # Place input sequence at the beginning
            target_seq[:seq_len] = input_seq
            
            # Place delayed copy after delay tokens
            start_idx = seq_len + delay
            if start_idx + seq_len <= self.sequence_length:
                target_seq[start_idx:start_idx + seq_len] = input_seq
            
            # Pad input sequence
            input_padded = torch.zeros(self.sequence_length, dtype=torch.long)
            input_padded[:seq_len] = input_seq
            
            samples.append((input_padded, target_seq))
        
        return samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class TextClassificationDataset(Dataset):
    """Dataset for text classification (SST-2 subset)."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        sequence_length: int,
        max_samples: Optional[int] = None,
    ):
        self.texts = texts[:max_samples] if max_samples else texts
        self.labels = labels[:max_samples] if max_samples else labels
        self.vocab = vocab
        self.sequence_length = sequence_length
        
        # Tokenize texts
        self.tokenized = self._tokenize_texts()
    
    def _tokenize_texts(self) -> List[torch.Tensor]:
        """Tokenize all texts."""
        tokenized = []
        
        for text in self.texts:
            # Simple word-based tokenization
            words = text.lower().split()
            tokens = []
            
            for word in words:
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                else:
                    tokens.append(self.vocab['<UNK>'])
            
            # Truncate or pad
            if len(tokens) > self.sequence_length:
                tokens = tokens[:self.sequence_length]
            else:
                tokens.extend([self.vocab['<PAD>']] * (self.sequence_length - len(tokens)))
            
            tokenized.append(torch.tensor(tokens, dtype=torch.long))
        
        return tokenized
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.tokenized[idx], self.labels[idx]


class DataManager:
    """Manages data loading and preprocessing."""
    
    def __init__(self, config):
        self.config = config
        self.vocab = None
        self.vocab_size = None
    
    def create_copy_task_data(self) -> Tuple[DataLoader, DataLoader]:
        """Create copy task data loaders."""
        # Training data
        train_dataset = CopyTaskDataset(
            num_samples=10000,
            min_length=self.config.copy_task.min_length,
            max_length=self.config.copy_task.max_length,
            vocab_size=self.config.data.vocab_size,
            delay_range=self.config.copy_task.delay_range,
            sequence_length=self.config.data.sequence_length,
        )
        
        # Validation data
        val_dataset = CopyTaskDataset(
            num_samples=2000,
            min_length=self.config.copy_task.min_length,
            max_length=self.config.copy_task.max_length,
            vocab_size=self.config.data.vocab_size,
            delay_range=self.config.copy_task.delay_range,
            sequence_length=self.config.data.sequence_length,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory,
        )
        
        return train_loader, val_loader
    
    def create_text_classification_data(self) -> Tuple[DataLoader, DataLoader]:
        """Create text classification data loaders."""
        
        # Check if we should use real SST-2 dataset
        if hasattr(self.config.text_cls, 'use_real_sst2') and self.config.text_cls.use_real_sst2:
            try:
                logger.info("Using real Stanford SST-2 dataset from Hugging Face")
                sst2_manager = SST2DataManager(self.config)
                train_loader, val_loader, test_loader = sst2_manager.create_data_loaders()
                
                # Update vocab info
                vocab_info = sst2_manager.get_vocab_info()
                self.vocab = vocab_info['vocab']
                self.vocab_size = vocab_info['vocab_size']
                
                return train_loader, val_loader
            except Exception as e:
                logger.warning(f"Failed to load real SST-2 dataset: {e}")
                logger.warning("Falling back to synthetic data")
                return self._create_synthetic_text_classification_data()
        else:
            return self._create_synthetic_text_classification_data()
    
    def _create_synthetic_text_classification_data(self) -> Tuple[DataLoader, DataLoader]:
        """Create text classification data loaders with synthetic data."""
        # Create a simple vocabulary
        self.vocab = self._create_vocab()
        self.vocab_size = len(self.vocab)
        
        # Generate synthetic text data
        texts, labels = self._generate_synthetic_text_data()
        
        # Split into train/val
        split_idx = int(0.8 * len(texts))
        train_texts, train_labels = texts[:split_idx], labels[:split_idx]
        val_texts, val_labels = texts[split_idx:], labels[split_idx:]
        
        # Create datasets
        train_dataset = TextClassificationDataset(
            texts=train_texts,
            labels=train_labels,
            vocab=self.vocab,
            sequence_length=self.config.data.sequence_length,
            max_samples=self.config.text_cls.max_samples // 2,
        )
        
        val_dataset = TextClassificationDataset(
            texts=val_texts,
            labels=val_labels,
            vocab=self.vocab,
            sequence_length=self.config.data.sequence_length,
            max_samples=self.config.text_cls.max_samples // 2,
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory,
        )
        
        return train_loader, val_loader
    
    def _create_vocab(self) -> Dict[str, int]:
        """Create a simple vocabulary for text classification."""
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
        }
        
        # Add common words
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'good', 'bad', 'great', 'terrible',
            'amazing', 'awful', 'excellent', 'horrible', 'wonderful', 'fantastic',
            'movie', 'film', 'book', 'story', 'plot', 'character', 'acting',
            'director', 'writer', 'performance', 'entertaining', 'boring',
            'interesting', 'exciting', 'funny', 'sad', 'happy', 'angry', 'love',
            'hate', 'like', 'dislike', 'enjoy', 'recommend', 'avoid', 'watch',
            'read', 'see', 'hear', 'feel', 'think', 'believe', 'know', 'understand'
        ]
        
        for i, word in enumerate(common_words):
            vocab[word] = i + 4
        
        return vocab
    
    def _generate_synthetic_text_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic text data for training."""
        texts = []
        labels = []
        
        # Positive sentiment templates
        positive_templates = [
            "This {noun} is {positive_adj} and {positive_adj}.",
            "I {positive_verb} this {noun} because it's {positive_adj}.",
            "The {noun} was {positive_adj} and {positive_adj}.",
            "What a {positive_adj} {noun}! I {positive_verb} it.",
            "This {noun} is {positive_adj} and {positive_adj}.",
        ]
        
        # Negative sentiment templates
        negative_templates = [
            "This {noun} is {negative_adj} and {negative_adj}.",
            "I {negative_verb} this {noun} because it's {negative_adj}.",
            "The {noun} was {negative_adj} and {negative_adj}.",
            "What a {negative_adj} {noun}! I {negative_verb} it.",
            "This {noun} is {negative_adj} and {negative_adj}.",
        ]
        
        # Vocabulary for templates
        nouns = ['movie', 'book', 'story', 'film', 'show', 'play', 'novel', 'article']
        positive_adj = ['amazing', 'wonderful', 'fantastic', 'excellent', 'great', 'good', 'brilliant']
        negative_adj = ['terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing', 'boring']
        positive_verb = ['love', 'enjoy', 'adore', 'like', 'recommend', 'appreciate']
        negative_verb = ['hate', 'dislike', 'loathe', 'avoid', 'regret', 'disappoint']
        
        # Generate positive samples
        for _ in range(self.config.text_cls.max_samples // 2):
            template = random.choice(positive_templates)
            text = template.format(
                noun=random.choice(nouns),
                positive_adj=random.choice(positive_adj),
                positive_verb=random.choice(positive_verb),
            )
            texts.append(text)
            labels.append(1)  # Positive
        
        # Generate negative samples
        for _ in range(self.config.text_cls.max_samples // 2):
            template = random.choice(negative_templates)
            text = template.format(
                noun=random.choice(nouns),
                negative_adj=random.choice(negative_adj),
                negative_verb=random.choice(negative_verb),
            )
            texts.append(text)
            labels.append(0)  # Negative
        
        return texts, labels
