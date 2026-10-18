"""Model registry keyed by HF `architectures[]` names. Mirrors sglang/srt/models/registry.py."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Type

import lula.models as pkg
from torch import nn


class _ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Type[nn.Module]] = {}
        self._imported = False

    def register(self, *arch_names: str):
        def deco(cls):
            for name in arch_names:
                self._models[name] = cls
            if not hasattr(cls, "ARCHITECTURES"):
                cls.ARCHITECTURES = list(arch_names)
            return cls
        return deco

    def import_models(self) -> None:
        if self._imported:
            return
        for m in pkgutil.iter_modules(pkg.__path__):
            if m.name != "registry":
                importlib.import_module(f"lula.models.{m.name}")
        self._imported = True

    def resolve(self, architectures: List[str]) -> Type[nn.Module]:
        self.import_models()
        for arch in architectures:
            if arch in self._models:
                return self._models[arch]
        raise ValueError(f"No registered model for architectures {architectures}; have {sorted(self._models)}")

    def supported(self) -> List[str]:
        self.import_models()
        return sorted(self._models)


ModelRegistry = _ModelRegistry()
