"""
W&B Sweep Runner for Name Generation RNN

This script initializes and runs W&B sweeps for hyperparameter optimization.

Usage:
    # Run a quick 3-trial test sweep
    uv run python run_wandb_sweep.py

    # Or manually start a sweep and run agents
    uv run python run_wandb_sweep.py --init-only
    wandb agent <sweep_id>
"""

import wandb
import subprocess
import argparse

# Minimal test sweep - 3 learning rates using grid search
TEST_SWEEP_CONFIG = {
    'name': 'name-rnn-quick-test',
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'program': 'name_generation_rnn_sweep.py',  # Script to run
    'parameters': {
        'learning_rate': {'values': [0.001, 0.003, 0.01]},
        'embedding_dim': {'value': 64},
        'hidden_size': {'value': 128},
        'num_layers': {'value': 2},
        'dropout': {'value': 0.2},
        'batch_size': {'value': 128},
        'max_epochs': {'value': 5},  # Only 5 epochs for quick testing
        'temperature': {'value': 0.8},
    }
}

# Full hyperparameter sweep - Bayesian optimization
FULL_SWEEP_CONFIG = {
    'name': 'name-rnn-bayesian-sweep',
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'program': 'name_generation_rnn_sweep.py',  # Script to run
    'parameters': {
        'embedding_dim': {'values': [32, 64, 128]},
        'hidden_size': {'values': [128, 256, 512]},
        'num_layers': {'values': [1, 2, 3]},
        'dropout': {'min': 0.0, 'max': 0.5, 'distribution': 'uniform'},
        'learning_rate': {'min': 0.0001, 'max': 0.01, 'distribution': 'log_uniform_values'},
        'batch_size': {'values': [64, 128, 256]},
        'max_epochs': {'value': 20},
        'temperature': {'min': 0.5, 'max': 1.5, 'distribution': 'uniform'},
    }
}


def run_sweep(sweep_config, count=None, project='name-generation-rnn'):
    """Initialize and run a W&B sweep.

    Args:
        sweep_config: Sweep configuration dictionary
        count: Number of runs to execute (None = run until interrupted)
        project: W&B project name
    """
    print("\n" + "="*60)
    print(f"INITIALIZING W&B SWEEP: {sweep_config['name']}")
    print("="*60)
    print(f"Method: {sweep_config['method']}")
    print(f"Metric: {sweep_config['metric']['name']} ({sweep_config['metric']['goal']})")
    print(f"Parameters: {len(sweep_config['parameters'])} hyperparameters")
    print("="*60 + "\n")

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=project)

    print(f"✨ Sweep ID: {sweep_id}")
    print(f"📊 View at: https://wandb.ai//name-generation-rnn/sweeps/{sweep_id.split('/')[-1]}")
    print()

    if count:
        print(f"Starting {count} sweep trials...")
    else:
        print("Starting sweep (Ctrl+C to stop)...")

    print("\nTo run additional agents in parallel, execute in another terminal:")
    print(f"  wandb agent {sweep_id}")
    print()

    # Run sweep agent
    # Instead of calling train() directly, we'll use subprocess to run the script
    # This ensures each run is isolated
    cmd = ['wandb', 'agent', sweep_id]
    if count:
        cmd.extend(['--count', str(count)])

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n\nSweep failed with error: {e}")
        return

    print("\n" + "="*60)
    print("✅ SWEEP COMPLETE!")
    print("="*60)
    print(f"View results: https://wandb.ai//name-generation-rnn/sweeps/{sweep_id.split('/')[-1]}")
    print("="*60 + "\n")


def init_sweep_only(sweep_config, project='name-generation-rnn'):
    """Initialize a sweep and print the command to run agents."""
    sweep_id = wandb.sweep(sweep_config, project=project)

    print(f"\n✨ Sweep initialized: {sweep_id}")
    print(f"📊 View at: https://wandb.ai//name-generation-rnn/sweeps/{sweep_id.split('/')[-1]}")
    print(f"\nTo run sweep agents:")
    print(f"  uv run wandb agent {sweep_id}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Run W&B hyperparameter sweep')
    parser.add_argument('--test', action='store_true',
                       help='Run quick 3-trial test sweep (default)')
    parser.add_argument('--full', action='store_true',
                       help='Run full Bayesian optimization sweep')
    parser.add_argument('--count', type=int, default=None,
                       help='Number of trials to run (default: all for grid, unlimited for bayes)')
    parser.add_argument('--init-only', action='store_true',
                       help='Only initialize sweep, don\'t run agent')

    args = parser.parse_args()

    # Choose sweep config
    if args.full:
        sweep_config = FULL_SWEEP_CONFIG
        default_count = 10 if args.count is None else args.count
    else:
        sweep_config = TEST_SWEEP_CONFIG
        default_count = 3 if args.count is None else args.count

    # Initialize or run sweep
    if args.init_only:
        init_sweep_only(sweep_config)
    else:
        run_sweep(sweep_config, count=default_count)


if __name__ == '__main__':
    main()
