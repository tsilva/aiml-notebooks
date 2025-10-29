#!/usr/bin/env python
"""
Get Top Runs from W&B Sweep

This script fetches the top N runs from a Weights & Biases sweep URL.
It returns the run IDs and configurations of the best performing runs
based on the sweep's optimization metric (default: top 3).

Usage:
    # Basic usage (shows top 3 runs by default, swept parameters only)
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id

    # Show top 5 runs
    uv run python get_best_sweep_run.py entity/project/sweep_id --top-n 5

    # Show only the best run
    uv run python get_best_sweep_run.py entity/project/sweep_id -n 1

    # Show all config parameters (swept + fixed)
    uv run python get_best_sweep_run.py entity/project/sweep_id --show-all-config

    # JSON output (array of top runs with swept_params and fixed_params)
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id --json

    # Verbose mode (see all runs and their metrics)
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id --verbose
"""

import argparse
import json
import re
import sys
from typing import Tuple, Dict, Any, Optional

import wandb


def parse_sweep_url(url_or_path: str) -> Tuple[str, str, str]:
    """Parse a W&B sweep URL or path to extract entity, project, and sweep_id.

    Args:
        url_or_path: Either a full URL or a path in format entity/project/sweep_id

    Returns:
        Tuple of (entity, project, sweep_id)

    Examples:
        >>> parse_sweep_url("https://wandb.ai/entity/project/sweeps/sweep_id")
        ('entity', 'project', 'sweep_id')

        >>> parse_sweep_url("entity/project/sweep_id")
        ('entity', 'project', 'sweep_id')
    """
    # Try to match full URL format
    url_pattern = r'https?://wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?]+)'
    match = re.match(url_pattern, url_or_path)

    if match:
        entity, project, sweep_id = match.groups()
        return entity, project, sweep_id

    # Try to match path format (entity/project/sweep_id)
    path_pattern = r'^([^/]+)/([^/]+)/([^/]+)$'
    match = re.match(path_pattern, url_or_path)

    if match:
        entity, project, sweep_id = match.groups()
        return entity, project, sweep_id

    raise ValueError(
        f"Invalid sweep URL or path format: {url_or_path}\n"
        f"Expected formats:\n"
        f"  - https://wandb.ai/ENTITY/PROJECT/sweeps/SWEEP_ID\n"
        f"  - ENTITY/PROJECT/SWEEP_ID"
    )


def get_best_runs(entity: str, project: str, sweep_id: str,
                  top_n: int = 3, verbose: bool = False, debug: bool = False,
                  show_all_config: bool = False) -> Optional[list]:
    """Fetch the top N best runs from a W&B sweep.

    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        sweep_id: Sweep ID
        top_n: Number of top runs to return (default: 3)
        verbose: If True, print detailed information about all runs
        debug: If True, print available metrics for debugging
        show_all_config: If True, show all config params (not just swept ones)

    Returns:
        List of dictionaries, each containing:
        - run_id: The run ID
        - run_name: The run name
        - run_url: URL to the run
        - metric_name: Name of the optimization metric
        - metric_value: Best metric value
        - config: Run configuration
        - swept_params: Parameters that were swept
        - fixed_params: Parameters that were fixed
        - state: Run state (finished, running, crashed, etc.)
        - rank: Rank of this run (1 = best, 2 = second best, etc.)
    """
    api = wandb.Api()

    # Fetch sweep
    try:
        sweep_path = f"{entity}/{project}/{sweep_id}"
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"Error fetching sweep: {e}", file=sys.stderr)
        return None

    # Get metric configuration
    metric_name = sweep.config.get('metric', {}).get('name')
    metric_goal = sweep.config.get('metric', {}).get('goal', 'minimize')

    if not metric_name:
        print("Warning: No metric defined in sweep config", file=sys.stderr)
        print("Available sweep config:", sweep.config, file=sys.stderr)
        return None

    # Get sweep parameters to identify which params were being swept
    sweep_parameters = sweep.config.get('parameters', {})

    if verbose:
        print(f"\nSweep: {sweep.name}")
        print(f"Metric: {metric_name} ({metric_goal})")
        print(f"Method: {sweep.config.get('method', 'unknown')}")
        print(f"\nFetching runs...")

    # Get all runs from the sweep
    runs = list(sweep.runs)

    if not runs:
        print("No runs found in sweep", file=sys.stderr)
        return None

    if verbose:
        print(f"Found {len(runs)} runs\n")

    # Filter runs that have the metric
    runs_with_metric = []
    for run in runs:
        # Convert summary to dict to safely access metrics
        summary_dict = None
        metric_value = None

        try:
            summary_dict = dict(run.summary)
            metric_value = summary_dict.get(metric_name)
        except (TypeError, AttributeError):
            # If conversion fails, try direct access
            try:
                metric_value = run.summary[metric_name]
            except (KeyError, TypeError):
                pass

        # If summary doesn't have the metric, try getting the last value from history
        if metric_value is None:
            try:
                history = run.history(keys=[metric_name], pandas=False)
                if history:
                    # Get the last logged value
                    for row in reversed(history):
                        if metric_name in row and row[metric_name] is not None:
                            metric_value = row[metric_name]
                            break
            except Exception:
                pass

        # Debug: show available metrics
        if debug:
            if summary_dict:
                print(f"\nRun {run.id} ({run.state}) summary metrics:")
                for key in sorted(summary_dict.keys()):
                    val = summary_dict[key]
                    if isinstance(val, (int, float)):
                        print(f"  {key}: {val}")
            else:
                print(f"\nRun {run.id} ({run.state}): No summary data")

            if metric_value is not None:
                print(f"  Found {metric_name} in history: {metric_value}")

        if metric_value is not None:
            runs_with_metric.append((run, metric_value))
            if verbose:
                state = run.state
                print(f"  {run.id} ({state}): {metric_name}={metric_value:.6f}")

    if not runs_with_metric:
        print(f"No runs have the metric '{metric_name}'", file=sys.stderr)
        return None

    # Sort runs based on goal (best first)
    if metric_goal == 'minimize':
        sorted_runs = sorted(runs_with_metric, key=lambda x: x[1])
    else:  # maximize
        sorted_runs = sorted(runs_with_metric, key=lambda x: x[1], reverse=True)

    # Get top N runs
    top_runs = sorted_runs[:top_n]

    # Helper function to extract config from a run
    def extract_config(run):
        """Extract and parse config from a run."""
        config = {}
        try:
            # Check if config is a JSON string
            if isinstance(run.config, str):
                config_raw = json.loads(run.config)
                # Extract the 'value' from each parameter (W&B sweep format)
                for key, val in config_raw.items():
                    if isinstance(val, dict) and 'value' in val:
                        config[key] = val['value']
                    else:
                        config[key] = val
            elif hasattr(run.config, 'keys') and callable(run.config.keys):
                for key in run.config.keys():
                    config[key] = run.config[key]
            elif hasattr(run.config, 'items') and callable(run.config.items):
                config = {k: v for k, v in run.config.items()}
            else:
                # Try direct dict conversion
                config = dict(run.config)

            # Remove internal W&B metadata (starts with _) unless in debug mode
            if not debug:
                config = {k: v for k, v in config.items() if not k.startswith('_')}

        except Exception as e:
            if verbose or debug:
                print(f"Warning: Could not extract config for {run.id}: {e}", file=sys.stderr)

        return config

    # Helper function to separate swept vs fixed parameters
    def is_swept_param(param_name: str, sweep_param_config) -> bool:
        """Check if a parameter was swept (has multiple values) or fixed (single value)."""
        if not isinstance(sweep_param_config, dict):
            return False

        # Check for various sweep configurations
        if 'values' in sweep_param_config:
            # Multiple values = swept parameter
            values = sweep_param_config['values']
            return isinstance(values, list) and len(values) > 1
        elif 'value' in sweep_param_config:
            # Single value = fixed parameter
            return False
        elif 'min' in sweep_param_config or 'max' in sweep_param_config:
            # Range-based sweep
            return True
        elif 'distribution' in sweep_param_config:
            # Distribution-based sweep
            return True

        return False

    def separate_params(config):
        """Separate config into swept and fixed parameters."""
        swept_params = {}
        fixed_params = {}

        for key, value in config.items():
            if key in sweep_parameters:
                if is_swept_param(key, sweep_parameters[key]):
                    swept_params[key] = value
                else:
                    fixed_params[key] = value
            else:
                # Parameters not in sweep config are considered fixed (from notebook defaults)
                fixed_params[key] = value

        return swept_params, fixed_params

    # Build results for top N runs
    results = []
    for rank, (run, metric_value) in enumerate(top_runs, start=1):
        config = extract_config(run)
        swept_params, fixed_params = separate_params(config)

        result = {
            'rank': rank,
            'run_id': run.id,
            'run_name': run.name,
            'run_url': run.url,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'metric_goal': metric_goal,
            'config': config,  # Full config
            'swept_params': swept_params,  # Only parameters that were varied
            'fixed_params': fixed_params,  # Parameters that were constant
            'state': run.state,
            'sweep_name': sweep.name,
            'sweep_url': sweep.url,
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Get the top N runs from a W&B sweep (default: top 3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (shows top 3 runs with swept parameters only)
  python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id

  # Show top 5 runs
  python get_best_sweep_run.py entity/project/sweep_id --top-n 5

  # Show only the best run
  python get_best_sweep_run.py entity/project/sweep_id -n 1

  # Show all config parameters (swept + fixed)
  python get_best_sweep_run.py entity/project/sweep_id --show-all-config

  # JSON output (array of runs with swept_params and fixed_params)
  python get_best_sweep_run.py entity/project/sweep_id --json

  # Verbose mode (see all runs and their metrics during sweep)
  python get_best_sweep_run.py entity/project/sweep_id --verbose
        """
    )

    parser.add_argument('sweep', type=str,
                       help='W&B sweep URL or path (entity/project/sweep_id)')
    parser.add_argument('--top-n', '-n', type=int, default=3,
                       help='Number of top runs to show (default: 3)')
    parser.add_argument('--json', action='store_true',
                       help='Output result as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information about all runs')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Show available metrics in each run (for debugging)')
    parser.add_argument('--show-all-config', '-a', action='store_true',
                       help='Show all config parameters (not just swept ones)')

    args = parser.parse_args()

    # Parse sweep URL/path
    try:
        entity, project, sweep_id = parse_sweep_url(args.sweep)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get best runs
    results = get_best_runs(entity, project, sweep_id,
                           top_n=args.top_n,
                           verbose=args.verbose,
                           debug=args.debug,
                           show_all_config=args.show_all_config)

    if not results:
        sys.exit(1)

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if args.verbose:
            print("\n" + "="*60)

        print(f"TOP {len(results)} RUN{'S' if len(results) > 1 else ''}")
        print("="*60)

        for i, result in enumerate(results):
            if i > 0:
                print()
                print("-" * 60)
                print()

            # Show rank with medal emojis for top 3
            rank_display = f"#{result['rank']}"
            if result['rank'] == 1:
                rank_display = f"🥇 #{result['rank']} (BEST)"
            elif result['rank'] == 2:
                rank_display = f"🥈 #{result['rank']}"
            elif result['rank'] == 3:
                rank_display = f"🥉 #{result['rank']}"

            print(f"Rank:        {rank_display}")
            print(f"Run ID:      {result['run_id']}")
            print(f"Run Name:    {result['run_name']}")
            print(f"State:       {result['state']}")
            print(f"Metric:      {result['metric_name']} = {result['metric_value']:.6f} ({result['metric_goal']})")
            print(f"Run URL:     {result['run_url']}")
            print()

            # Show swept parameters prominently
            if result['swept_params']:
                print("Swept Parameters:")
                for key, value in sorted(result['swept_params'].items()):
                    print(f"  {key:30s} = {value}")
                print()

            # Optionally show fixed parameters (only for first run to avoid repetition)
            if args.show_all_config and result['fixed_params'] and i == 0:
                print("Fixed Parameters (same for all runs):")
                for key, value in sorted(result['fixed_params'].items()):
                    print(f"  {key:30s} = {value}")
                print()

        print("="*60)


if __name__ == '__main__':
    main()
