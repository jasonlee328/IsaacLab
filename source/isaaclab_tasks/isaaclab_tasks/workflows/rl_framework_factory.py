from __future__ import annotations

from .rl_frameworks.rslrl_base import RslRlBase

class RlFrameworkFactory:
    def __init__(self, name: str):
        self._name = name.lower()

    def create(self):
        if self._name in {"rslrl", "rsl-rl"}:
            return RslRlBase
        raise ValueError(f"Unknown rl framework: {self._name}")