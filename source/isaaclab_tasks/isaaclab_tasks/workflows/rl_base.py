from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


class RlBase(ABC):
    """Minimal abstract base for RL framework adapters.

    A concrete implementation should wrap a specific RL library (e.g., RSL-RL)
    and implement the learn() and inference() entry-points that the workflow calls.
    """

    def __init__(self, experiment_manager) -> None:
        self.exp_mgr = experiment_manager

    def add_algo_args(self, parser: argparse.ArgumentParser) -> None:
        """Add framework-specific CLI arguments (kept minimal)."""
        arg_group = parser.add_argument_group("algo", description="Algorithm arguments.")
        arg_group.add_argument("--max_iterations", type=int, default=None, help="RL iterations (training only).")

    def update_algo_cfg(self, args_cli: argparse.Namespace) -> None:
        """Allow workflow to pass top-level args into the agent config if needed.

        Default implementation is a no-op; concrete adapters may override.
        """
        return None

    @abstractmethod
    def learn(self) -> None:
        """Run training loop (or resume), respecting exp_mgr configuration."""
        ...

    @abstractmethod
    def inference(self) -> None:
        """Run evaluation/inference without performing training updates."""
        ...


