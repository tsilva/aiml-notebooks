#!/usr/bin/env python
"""
Character-Level Name Generation with RNN - W&B Sweep Version

This script trains a character-level RNN to generate names and is designed
to work with Weights & Biases hyperparameter sweeps.

Usage:
    # Single run with defaults
    uv run python name_generation_rnn_sweep.py

    # With W&B sweep (after creating sweep)
    wandb agent <sweep_id>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
import numpy as np
import urllib.request
import wandb
from pytorch_lightning.loggers import WandbLogger


class CharacterTokenizer:
    """Tokenizer for character-level text encoding/decoding."""
    def __init__(self, texts, special_token='.'):
        """Build vocabulary from a list of texts."""
        chars = sorted(list(set(''.join(texts))))
        self.chars = [special_token] + chars

        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        self.special_token = special_token
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text to list of indices."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Convert list of indices to text."""
        return ''.join([self.idx_to_char[i] for i in indices])

    def __repr__(self):
        return f"CharacterTokenizer(vocab_size={self.vocab_size}, chars={''.join(self.chars)})"


class NamesDataset(Dataset):
    """Dataset that converts names to character sequences with start/end tokens."""
    def __init__(self, names, tokenizer, max_length=None):
        self.names = names
        self.tokenizer = tokenizer
        self.max_length = max_length or max(len(n) for n in names) + 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        name_with_tokens = self.tokenizer.special_token + name + self.tokenizer.special_token
        indices = self.tokenizer.encode(name_with_tokens)

        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        return x, y


class NameGeneratorRNN(L.LightningModule):
    """Character-level RNN for name generation."""
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.2, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded, hidden)
        rnn_out = self.dropout(rnn_out)
        logits = self.fc(rnn_out)
        return logits, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.criterion(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self(x)
        loss = self.criterion(logits.view(-1, self.hparams.vocab_size), y.view(-1))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @torch.no_grad()
    def generate(self, tokenizer, max_length=20, temperature=1.0, num_samples=1):
        """Generate names using the tokenizer."""
        self.eval()
        generated_names = []

        for _ in range(num_samples):
            current_idx = tokenizer.char_to_idx[tokenizer.special_token]
            name_chars = []
            hidden = None

            for _ in range(max_length):
                x = torch.tensor([[current_idx]], dtype=torch.long, device=self.device)
                logits, hidden = self(x, hidden)
                probs = F.softmax(logits[0, -1] / temperature, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                next_char = tokenizer.idx_to_char[next_idx]

                if next_char == tokenizer.special_token:
                    break

                name_chars.append(next_char)
                current_idx = next_idx

            generated_names.append(''.join(name_chars))

        return generated_names


def collate_fn(batch):
    """Pad variable-length sequences."""
    xs, ys = zip(*batch)
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    return xs_padded, ys_padded


def train():
    """Main training function for W&B sweep."""

    # Initialize W&B run (sweep agent will configure this)
    run = wandb.init(project='name-generation-rnn')

    # Get configuration from sweep (or use defaults)
    config = wandb.config

    # Default configuration
    CONFIG = {
        'seed': 42,
        'train_split': 0.9,
        'batch_size': getattr(config, 'batch_size', 128),
        'embedding_dim': getattr(config, 'embedding_dim', 64),
        'hidden_size': getattr(config, 'hidden_size', 256),
        'num_layers': getattr(config, 'num_layers', 2),
        'dropout': getattr(config, 'dropout', 0.2),
        'learning_rate': getattr(config, 'learning_rate', 1e-3),
        'max_epochs': getattr(config, 'max_epochs', 10),
        'log_every_n_steps': 20,
        'max_length': 15,
        'temperature': getattr(config, 'temperature', 0.8),
        'sample_size': 20,
        'novelty_sample_size': 100,
    }

    print(f"\nRunning with config:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Set random seeds
    L.seed_everything(CONFIG['seed'])

    # Download and load names dataset
    print("Loading dataset...")
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, 'names.txt')

    with open('names.txt', 'r') as f:
        names = f.read().splitlines()

    names = [name.strip().lower() for name in names if name.strip()]
    print(f"Loaded {len(names)} names")

    # Build tokenizer
    tokenizer = CharacterTokenizer(names, special_token='.')
    print(tokenizer)

    # Create dataset
    dataset = NamesDataset(names, tokenizer)
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, collate_fn=collate_fn)

    # Initialize model
    print("\nInitializing model...")
    model = NameGeneratorRNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        learning_rate=CONFIG['learning_rate'],
    )

    # Create W&B logger (will use existing run from wandb.init())
    wandb_logger = WandbLogger()

    # Train model
    print("\nTraining...")
    trainer = L.Trainer(
        max_epochs=CONFIG['max_epochs'],
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=CONFIG['log_every_n_steps'],
        logger=wandb_logger,
    )
    trainer.fit(model, train_loader, val_loader)

    # Generate sample names
    print("\nGenerating sample names...")
    generated = model.generate(
        tokenizer,
        max_length=CONFIG['max_length'],
        temperature=CONFIG['temperature'],
        num_samples=CONFIG['sample_size']
    )
    generated = [name.capitalize() for name in generated]
    print(", ".join(generated))

    # Analyze novelty
    print("\n" + "="*50)
    print("NOVELTY ANALYSIS")
    print("="*50)

    sample = model.generate(
        tokenizer,
        max_length=CONFIG['max_length'],
        temperature=CONFIG['temperature'],
        num_samples=CONFIG['novelty_sample_size']
    )
    sample = [name for name in sample if name]

    original_names_set = set(names)
    new_names = [name for name in sample if name not in original_names_set]
    existing_names = [name for name in sample if name in original_names_set]

    # Calculate metrics
    novelty_pct = len(new_names) / len(sample) * 100
    uniqueness_pct = len(set(sample)) / len(sample) * 100

    print(f"Total: {len(sample)}, Unique: {len(set(sample))}")
    print(f"✨ NEW: {len(new_names)} ({novelty_pct:.1f}%)")
    print(f"♻️  EXISTING: {len(existing_names)} ({len(existing_names)/len(sample)*100:.1f}%)")

    print(f"\nNEW: {', '.join([n.capitalize() for n in new_names[:10]])}")
    print(f"EXISTING: {', '.join([n.capitalize() for n in existing_names[:10]])}")

    # Log novelty metrics to W&B
    wandb.log({
        'novelty_percentage': novelty_pct,
        'uniqueness_percentage': uniqueness_pct,
        'total_generated': len(sample),
        'new_names_count': len(new_names),
        'memorized_names_count': len(existing_names),
    })

    # Log sample names as a table
    wandb.log({
        'sample_new_names': wandb.Table(
            columns=['name'],
            data=[[n.capitalize()] for n in new_names[:20]]
        )
    })

    print("\n✅ Training complete!")

    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    train()
