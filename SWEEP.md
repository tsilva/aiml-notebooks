# Running W&B Sweeps with Papermill

This document explains how to run hyperparameter sweeps on Jupyter notebooks using Weights & Biases (W&B) and Papermill.

## Prerequisites

1. **W&B Account**: Sign up at https://wandb.ai
2. **W&B Login**: Run `wandb login` and enter your API key
3. **Dependencies**: Already installed via `uv sync`

## Quick Start

Run a hyperparameter sweep on the name-generation-rnn notebook:

```bash
uv run python run_sweep.py
```

This will:
1. Initialize a W&B sweep with the configuration defined in `run_sweep.py`
2. Run 5 training trials with different hyperparameter combinations
3. Save output notebooks to `notebooks/output/`
4. Track all metrics in your W&B dashboard

## Sweep Configuration

The sweep is configured in `run_sweep.py` with these hyperparameters:

| Parameter | Type | Range/Values |
|-----------|------|--------------|
| `embedding_dim` | Categorical | [32, 64, 128] |
| `hidden_size` | Categorical | [128, 256, 512] |
| `num_layers` | Categorical | [1, 2, 3] |
| `dropout` | Continuous | 0.0 - 0.5 |
| `learning_rate` | Log-uniform | 0.0001 - 0.01 |
| `batch_size` | Categorical | [64, 128, 256] |
| `temperature` | Continuous | 0.5 - 1.5 |

The sweep uses **Bayesian optimization** to intelligently search the hyperparameter space.

## Advanced Usage

### Customize Sweep Configuration

Edit `run_sweep.py` to modify:

1. **Search method**: Change `method` to `'grid'`, `'random'`, or `'bayes'`
2. **Metric to optimize**: Change `metric.name` (e.g., `'novelty_percentage'`)
3. **Hyperparameter ranges**: Add/remove parameters in `sweep_config`
4. **Number of trials**: Change `count=5` in `wandb.agent()`

### Run Multiple Agents in Parallel

After initializing a sweep, run additional agents in parallel to speed up the search:

```bash
# Terminal 1
uv run python run_sweep.py

# Terminal 2 (copy the sweep_id from Terminal 1)
wandb agent <sweep_id>

# Terminal 3
wandb agent <sweep_id>
```

### View Results

1. **W&B Dashboard**: Visit the URL printed when the sweep starts
2. **Output Notebooks**: Check `notebooks/output/sweep-run-*.ipynb`
3. **Best Configuration**: View in W&B sweep overview

### Resume Existing Sweep

If you have a sweep ID from a previous run:

```python
# In run_sweep.py, replace main() with:
def main():
    sweep_id = '<your-sweep-id>'  # e.g., 'username/project/abc123'
    wandb.agent(sweep_id, function=train, count=10)
```

Or run directly:

```bash
wandb agent <sweep_id>
```

## How It Works

1. **Papermill Integration**: The notebook has a tagged "parameters" cell that Papermill injects values into
2. **W&B Tracking**: Each trial logs metrics (loss, novelty, etc.) to W&B
3. **Bayesian Optimization**: W&B suggests new hyperparameter combinations based on previous results
4. **Output Storage**: Each trial saves an executed notebook with results

## Troubleshooting

### "No module named wandb"
```bash
uv sync
```

### "Not logged in to W&B"
```bash
wandb login
```

### Sweep fails with notebook error
- Check the output notebook in `notebooks/output/` for error details
- Ensure the base notebook runs successfully without sweep

### Want to test without W&B
Edit the notebook and set:
```python
use_wandb = False
```

## Sweep Strategies

### Grid Search
- **When**: Small hyperparameter space, want to try all combinations
- **Config**: `method: 'grid'`

### Random Search
- **When**: Large space, want diverse sampling
- **Config**: `method: 'random'`

### Bayesian Optimization (Recommended)
- **When**: Expensive trials, want intelligent search
- **Config**: `method: 'bayes'`

## Best Practices

1. **Start Small**: Run 5-10 trials first to validate setup
2. **Monitor First Trial**: Watch the first run to catch errors early
3. **Use Sensible Ranges**: Don't make ranges too wide (e.g., LR: 1e-5 to 1.0)
4. **Check Outputs**: Periodically review generated notebooks for issues
5. **Clean Up**: Delete old output notebooks to save space

## Example Output

```
Sweep initialized: username/name-generation-rnn/abc123xyz
View at: https://wandb.ai/username/name-generation-rnn/sweeps/abc123xyz

Running sweep trial: autumn-sweep-1
Parameters: {'embedding_dim': 64, 'hidden_size': 256, ...}
Output notebook: notebooks/output/sweep-run-abc123.ipynb

Successfully completed run: autumn-sweep-1
```

## References

- [W&B Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Papermill Documentation](https://papermill.readthedocs.io/)
- [PyTorch Lightning + W&B](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html)
