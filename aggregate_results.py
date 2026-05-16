"""
Compatibility wrapper for the Snellius baseline aggregation path.

Run after all manifest-driven SLURM jobs have finished:
    python aggregate_results.py
"""

from __future__ import annotations

from argparse import Namespace

from baseline_snellius import DEFAULT_PAIR, aggregate_command


def main() -> None:
    aggregate_command(
        Namespace(
            project_dir=None,
            manifest=None,
            pair=DEFAULT_PAIR,
            allow_partial=False,
        )
    )


if __name__ == "__main__":
    main()
