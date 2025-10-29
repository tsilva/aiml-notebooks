#!/usr/bin/env python
"""
Get Best Run from W&B Sweep

This script fetches the best run from a Weights & Biases sweep URL.
It returns the run ID and configuration of the best performing run
based on the sweep's optimization metric.

Usage:
    # From full sweep URL
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id

    # From sweep path (entity/project/sweep_id)
    uv run python get_best_sweep_run.py entity/project/sweep_id

    # With JSON output (for scripting)
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id --json

    # Verbose mode (see all runs and their metrics)
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id --verbose

    # Debug mode (show available metrics in each run)
    uv run python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id --debug
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


def get_best_run(entity: str, project: str, sweep_id: str,
                 verbose: bool = False, debug: bool = False) -> Optional[Dict[str, Any]]:
    """Fetch the best run from a W&B sweep.

    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        sweep_id: Sweep ID
        verbose: If True, print detailed information about all runs
        debug: If True, print available metrics for debugging

    Returns:
        Dictionary containing:
        - run_id: The run ID
        - run_name: The run name
        - run_url: URL to the run
        - metric_name: Name of the optimization metric
        - metric_value: Best metric value
        - config: Run configuration
        - state: Run state (finished, running, crashed, etc.)
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

    # Find best run based on goal
    if metric_goal == 'minimize':
        best_run, best_value = min(runs_with_metric, key=lambda x: x[1])
    else:  # maximize
        best_run, best_value = max(runs_with_metric, key=lambda x: x[1])

    # Extract config (convert to regular dict for JSON serialization)
    config = {}
    try:
        # Check if config is a JSON string
        if isinstance(best_run.config, str):
            config_raw = json.loads(best_run.config)
            # Extract the 'value' from each parameter (W&B sweep format)
            for key, val in config_raw.items():
                if isinstance(val, dict) and 'value' in val:
                    config[key] = val['value']
                else:
                    config[key] = val
        elif hasattr(best_run.config, 'keys') and callable(best_run.config.keys):
            for key in best_run.config.keys():
                config[key] = best_run.config[key]
        elif hasattr(best_run.config, 'items') and callable(best_run.config.items):
            config = {k: v for k, v in best_run.config.items()}
        else:
            # Try direct dict conversion
            config = dict(best_run.config)

        # Remove internal W&B metadata (starts with _) unless in debug mode
        if not debug:
            config = {k: v for k, v in config.items() if not k.startswith('_')}

    except Exception as e:
        if verbose or debug:
            print(f"Warning: Could not extract config: {e}", file=sys.stderr)
            print(f"Config type: {type(best_run.config)}", file=sys.stderr)

    result = {
        'run_id': best_run.id,
        'run_name': best_run.name,
        'run_url': best_run.url,
        'metric_name': metric_name,
        'metric_value': best_value,
        'metric_goal': metric_goal,
        'config': config,
        'state': best_run.state,
        'sweep_name': sweep.name,
        'sweep_url': sweep.url,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Get the best run from a W&B sweep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From full URL
  python get_best_sweep_run.py https://wandb.ai/entity/project/sweeps/sweep_id

  # From path format
  python get_best_sweep_run.py entity/project/sweep_id

  # JSON output (for scripting)
  python get_best_sweep_run.py entity/project/sweep_id --json

  # Verbose mode (see all runs)
  python get_best_sweep_run.py entity/project/sweep_id --verbose

  # Debug mode (show available metrics in each run)
  python get_best_sweep_run.py entity/project/sweep_id --debug
        """
    )

    parser.add_argument('sweep', type=str,
                       help='W&B sweep URL or path (entity/project/sweep_id)')
    parser.add_argument('--json', action='store_true',
                       help='Output result as JSON')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information about all runs')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Show available metrics in each run (for debugging)')

    args = parser.parse_args()

    # Parse sweep URL/path
    try:
        entity, project, sweep_id = parse_sweep_url(args.sweep)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Get best run
    result = get_best_run(entity, project, sweep_id, verbose=args.verbose, debug=args.debug)

    if not result:
        sys.exit(1)

    # Output result
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if args.verbose:
            print("\n" + "="*60)
        print(f"BEST RUN")
        print("="*60)
        print(f"Run ID:      {result['run_id']}")
        print(f"Run Name:    {result['run_name']}")
        print(f"State:       {result['state']}")
        print(f"Metric:      {result['metric_name']} = {result['metric_value']:.6f} ({result['metric_goal']})")
        print(f"Run URL:     {result['run_url']}")
        print()
        print("Configuration:")
        print("-" * 60)
        for key, value in sorted(result['config'].items()):
            print(f"  {key:30s} = {value}")
        print("="*60)


if __name__ == '__main__':
    main()
