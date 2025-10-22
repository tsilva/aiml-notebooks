"""
W&B Sweep Runner for Name Generation RNN using Papermill

This script runs hyperparameter sweeps on the name-generation-rnn notebook
using Weights & Biases and Papermill.

Usage:
    # Initialize and run sweep
    uv run python run_sweep.py

    # Or run as an agent (if sweep already exists)
    wandb agent <sweep_id>
"""

import wandb
import papermill as pm
from pathlib import Path
import os

# Ensure output directory exists
output_dir = Path("notebooks/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the sweep configuration
sweep_config = {
    'name': 'name-generation-rnn-sweep',
    'method': 'bayes',  # Options: 'grid', 'random', 'bayes'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Model architecture
        'embedding_dim': {
            'values': [32, 64, 128]
        },
        'hidden_size': {
            'values': [128, 256, 512]
        },
        'num_layers': {
            'values': [1, 2, 3]
        },

        # Regularization
        'dropout': {
            'min': 0.0,
            'max': 0.5,
            'distribution': 'uniform'
        },

        # Optimization
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01,
            'distribution': 'log_uniform_values'
        },
        'batch_size': {
            'values': [64, 128, 256]
        },

        # Training
        'max_epochs': {
            'value': 30  # Fixed value
        },

        # Generation
        'temperature': {
            'min': 0.5,
            'max': 1.5,
            'distribution': 'uniform'
        },
    }
}


def train():
    """Execute a single training run with parameters from W&B sweep."""
    # Initialize W&B run (sweep agent will configure this)
    run = wandb.init()

    # Get sweep parameters
    config = wandb.config

    # Create parameters dict for papermill
    # These will override the CONFIG dictionary in the notebook
    parameters = {
        # Hyperparameters from sweep
        'embedding_dim': config.embedding_dim,
        'hidden_size': config.hidden_size,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'max_epochs': config.max_epochs,
        'temperature': config.temperature,

        # Enable W&B and set run name
        'use_wandb': True,
        'wandb_project': 'name-generation-rnn',
        'wandb_run_name': run.name,
    }

    # Define paths
    input_notebook = 'notebooks/name-generation-rnn.ipynb'
    output_notebook = f'notebooks/output/sweep-run-{run.id}.ipynb'

    print(f"\n{'='*60}")
    print(f"Running sweep trial: {run.name}")
    print(f"Parameters: {parameters}")
    print(f"Output notebook: {output_notebook}")
    print(f"{'='*60}\n")

    try:
        # Execute notebook with papermill
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            parameters=parameters,
            kernel_name='python3',
            progress_bar=True,
        )
        print(f"\nSuccessfully completed run: {run.name}")

    except Exception as e:
        print(f"\nError during notebook execution: {e}")
        # Log the error to W&B
        wandb.log({'error': str(e)})
        raise

    finally:
        # Finish the W&B run
        wandb.finish()


def main():
    """Initialize sweep and run agents."""
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project='name-generation-rnn'
    )

    print(f"\nSweep initialized: {sweep_id}")
    print(f"View at: https://wandb.ai/{os.environ.get('WANDB_ENTITY', 'your-username')}/name-generation-rnn/sweeps/{sweep_id}")
    print(f"\nTo run additional agents in parallel, execute:")
    print(f"  wandb agent {sweep_id}")
    print()

    # Run the agent (can specify count to limit number of runs)
    # Remove or adjust count parameter to run more/fewer trials
    wandb.agent(
        sweep_id,
        function=train,
        count=5  # Run 5 trials (remove this for unlimited)
    )


if __name__ == '__main__':
    main()
