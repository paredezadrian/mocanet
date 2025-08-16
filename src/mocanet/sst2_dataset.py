"""Stanford SST-2 dataset loader for MOCA-Net text classification."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class SST2Dataset(Dataset):
    """Stanford Sentiment Treebank v2 (SST-2) dataset."""
    
    def __init__(
        self,
        split: str = "train",
        max_length: int = 128,
        vocab: Optional[Dict[str, int]] = None,
        build_vocab: bool = True,
        min_freq: int = 2,
        max_vocab_size: int = 10000,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize SST-2 dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_length: Maximum sequence length
            vocab: Pre-built vocabulary (if None, will build from data)
            build_vocab: Whether to build vocabulary from data
            min_freq: Minimum token frequency for vocabulary
            max_vocab_size: Maximum vocabulary size
            cache_dir: Directory to cache downloaded dataset
        """
        self.split = split
        self.max_length = max_length
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.cache_dir = cache_dir
        
        # Load dataset from Hugging Face
        self.dataset = self._load_dataset()
        
        # Build or use vocabulary
        if vocab is not None:
            self.vocab = vocab
        elif build_vocab:
            self.vocab = self._build_vocab()
        else:
            raise ValueError("Either vocab or build_vocab=True must be provided")
        
        self.vocab_size = len(self.vocab)
        
        # Tokenize all texts
        self.tokenized_texts, self.labels = self._tokenize_dataset()
        
        logger.info(f"SST-2 {split} dataset loaded: {len(self.tokenized_texts)} samples")
        logger.info(f"Vocabulary size: {self.vocab_size}")
    
    def _load_dataset(self):
        """Load SST-2 dataset from Hugging Face."""
        try:
            logger.info(f"Loading SST-2 dataset (split: {self.split})...")
            dataset = load_dataset("stanfordnlp/sst2", split=self.split, cache_dir=self.cache_dir)
            logger.info(f"Successfully loaded {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load SST-2 dataset: {e}")
            raise
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from training data."""
        logger.info("Building vocabulary from SST-2 training data...")
        
        # Load training data for vocabulary building
        train_dataset = load_dataset("stanfordnlp/sst2", split="train", cache_dir=self.cache_dir)
        
        # Count token frequencies
        token_freq = {}
        for example in train_dataset:
            text = example['sentence'].lower()
            tokens = text.split()
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Filter by minimum frequency and sort by frequency
        filtered_tokens = [(token, freq) for token, freq in token_freq.items() 
                          if freq >= self.min_freq]
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Build vocabulary
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
        }
        
        # Add most frequent tokens
        for i, (token, _) in enumerate(filtered_tokens[:self.max_vocab_size - 4]):
            vocab[token] = i + 4
        
        logger.info(f"Built vocabulary with {len(vocab)} tokens")
        return vocab
    
    def _tokenize_dataset(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Tokenize all texts in the dataset."""
        tokenized_texts = []
        labels = []
        
        for example in self.dataset:
            text = example['sentence'].lower()
            label = example['label']
            
            # Tokenize text
            tokens = text.split()
            token_ids = []
            
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.vocab['<UNK>'])
            
            # Truncate or pad to max_length
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            else:
                token_ids.extend([self.vocab['<PAD>']] * (self.max_length - len(token_ids)))
            
            tokenized_texts.append(torch.tensor(token_ids, dtype=torch.long))
            labels.append(label)
        
        return tokenized_texts, labels
    
    def __len__(self) -> int:
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.tokenized_texts[idx], self.labels[idx]


class SST2DataManager:
    """Manages SST-2 dataset loading and preprocessing."""
    
    def __init__(self, config, cache_dir: Optional[str] = None):
        self.config = config
        self.cache_dir = cache_dir
        self.vocab = None
        self.vocab_size = None
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test data loaders."""
        
        # Build vocabulary from training data
        logger.info("Building vocabulary from training data...")
        train_for_vocab = SST2Dataset(
            split="train",
            max_length=self.config.data.sequence_length,
            build_vocab=True,
            min_freq=self.config.text_cls.min_freq,
            max_vocab_size=self.config.data.vocab_size,
            cache_dir=self.cache_dir
        )
        self.vocab = train_for_vocab.vocab
        self.vocab_size = train_for_vocab.vocab_size
        
        # Create datasets with shared vocabulary
        train_dataset = SST2Dataset(
            split="train",
            max_length=self.config.data.sequence_length,
            vocab=self.vocab,
            build_vocab=False,
            cache_dir=self.cache_dir
        )
        
        val_dataset = SST2Dataset(
            split="validation",
            max_length=self.config.data.sequence_length,
            vocab=self.vocab,
            build_vocab=False,
            cache_dir=self.cache_dir
        )
        
        test_dataset = SST2Dataset(
            split="test",
            max_length=self.config.data.sequence_length,
            vocab=self.vocab,
            build_vocab=False,
            cache_dir=self.cache_dir
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
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.num_workers,
            pin_memory=self.config.hardware.pin_memory,
        )
        
        logger.info(f"Created data loaders:")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information."""
        return {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'pad_token_id': self.vocab['<PAD>'],
            'unk_token_id': self.vocab['<UNK>'],
            'start_token_id': self.vocab['<START>'],
            'end_token_id': self.vocab['<END>'],
        }
