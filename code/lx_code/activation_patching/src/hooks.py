"""Hook utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List


class HookManager:
    def __init__(self) -> None:
        self._handles: List = []

    def add(self, handle) -> None:
        self._handles.append(handle)

    def remove_all(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def register_forward_hooks(layers: Iterable, hook_fn: Callable) -> HookManager:
    manager = HookManager()
    for layer in layers:
        handle = layer.register_forward_hook(hook_fn)
        manager.add(handle)
    return manager
