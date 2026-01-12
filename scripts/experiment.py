#!/usr/bin/env python3
"""
Experiment tracking for entity resolution benchmark.

Provides standardized experiment logging with:
- Unique run IDs (timestamp-based)
- Git commit tracking
- Consistent result format
- Central experiment registry
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

EXPERIMENTS_FILE = Path("data/outputs/experiments.jsonl")


def get_git_info() -> Dict[str, str]:
    """Get current git commit and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}


def generate_run_id(method: str) -> str:
    """Generate unique run ID: method_YYYYMMDD_HHMMSS."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{method}_{timestamp}"


class Experiment:
    """
    Track a single experiment run.

    Usage:
        exp = Experiment(method="nomenklatura", input_file="sample_1000.json")
        exp.set_params(algorithm="regression-v1", threshold=0.5)
        # ... run experiment ...
        exp.set_metrics(accuracy=0.95, f1=0.92, ...)
        exp.save()
    """

    def __init__(self, method: str, input_file: str):
        self.run_id = generate_run_id(method)
        self.timestamp = datetime.now().isoformat()
        self.method = method
        self.input_file = str(input_file)
        self.git = get_git_info()
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.notes: str = ""

    def set_params(self, **kwargs):
        """Set experiment parameters."""
        self.params.update(kwargs)
        return self

    def set_metrics(self, **kwargs):
        """Set result metrics."""
        self.metrics.update(kwargs)
        return self

    def add_note(self, note: str):
        """Add a note to the experiment."""
        self.notes = note
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "method": self.method,
            "input_file": self.input_file,
            "git": self.git,
            "params": self.params,
            "metrics": self.metrics,
            "notes": self.notes
        }

    def save(self, output_dir: str = "data/outputs") -> Path:
        """
        Save experiment results.

        Creates:
        - Individual result file: {output_dir}/{run_id}.json
        - Appends to registry: data/outputs/experiments.jsonl

        Returns path to the individual result file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.to_dict()

        # Save individual result
        result_file = output_dir / f"{self.run_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Append to registry
        EXPERIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENTS_FILE, 'a') as f:
            f.write(json.dumps(result) + "\n")

        return result_file

    def print_summary(self):
        """Print a formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.run_id}")
        print(f"{'='*60}")
        print(f"Method:     {self.method}")
        print(f"Input:      {self.input_file}")
        print(f"Git:        {self.git['branch']}@{self.git['commit']}")
        print(f"Timestamp:  {self.timestamp}")

        if self.params:
            print(f"\nParameters:")
            for k, v in self.params.items():
                print(f"  {k}: {v}")

        if self.metrics:
            print(f"\nMetrics:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                elif isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")

        if self.notes:
            print(f"\nNotes: {self.notes}")

        print(f"{'='*60}\n")


def list_experiments(limit: int = 10) -> list:
    """Load recent experiments from registry."""
    if not EXPERIMENTS_FILE.exists():
        return []

    experiments = []
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))

    return experiments[-limit:]


def compare_experiments(run_ids: Optional[list] = None, method: Optional[str] = None):
    """
    Print comparison table of experiments.

    Args:
        run_ids: Specific run IDs to compare
        method: Filter by method name
    """
    experiments = []
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            if line.strip():
                exp = json.loads(line)
                if run_ids and exp["run_id"] not in run_ids:
                    continue
                if method and exp["method"] != method:
                    continue
                experiments.append(exp)

    if not experiments:
        print("No experiments found.")
        return

    # Print comparison table
    print(f"\n{'='*100}")
    print(f"{'Run ID':<35} {'Method':<15} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8}")
    print(f"{'='*100}")

    for exp in experiments:
        m = exp.get("metrics", {})
        print(f"{exp['run_id']:<35} {exp['method']:<15} "
              f"{m.get('f1', 0):>8.4f} {m.get('accuracy', 0):>8.4f} "
              f"{m.get('precision', 0):>8.4f} {m.get('recall', 0):>8.4f}")

    print(f"{'='*100}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment tracking utilities")
    parser.add_argument("--list", "-l", action="store_true", help="List recent experiments")
    parser.add_argument("--compare", "-c", action="store_true", help="Compare experiments")
    parser.add_argument("--method", "-m", help="Filter by method")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of experiments to show")

    args = parser.parse_args()

    if args.list:
        experiments = list_experiments(args.limit)
        for exp in experiments:
            m = exp.get("metrics", {})
            print(f"{exp['run_id']}: F1={m.get('f1', 0):.4f}, Acc={m.get('accuracy', 0):.4f}")
    elif args.compare:
        compare_experiments(method=args.method)
    else:
        parser.print_help()
