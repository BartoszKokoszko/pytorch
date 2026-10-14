import contextlib
import os
from collections.abc import Generator
from typing import Any

import torch
from torch._C._functorch import (
    get_single_level_autograd_function_allowed,
    set_single_level_autograd_function_allowed,
    unwrap_dead_wrappers as _unwrap_dead_wrappers_fast,
    unwrap_if_dead,
)
from torch.utils._exposed_in import exposed_in


__all__ = [
    "exposed_in",
    "argnums_t",
    "enable_single_level_autograd_function",
    "unwrap_dead_wrappers",
]


def _unwrap_dead_wrappers_slow(args: tuple[Any, ...]) -> tuple[Any, ...]:
    # Pre-#189577 (ad17677ba8d14e6e05289447682b28c215ad8038) pure-Python
    # implementation. NB: doesn't use tree_map_only for performance reasons.
    result = tuple(
        unwrap_if_dead(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
    )
    return result


# Set PYTORCH_DISABLE_UNWRAP_DEAD_WRAPPERS_FAST=1 to make this re-export the
# pre-#189577 pure-Python implementation (matching the runtime toggle in
# torch/csrc/functorch/init.cpp and torch/_dynamo/trace_rules.py), for
# bisecting/comparing against the pre-#189577 behavior without a rebuild.
if os.environ.get("PYTORCH_DISABLE_UNWRAP_DEAD_WRAPPERS_FAST") == "1":
    unwrap_dead_wrappers = _unwrap_dead_wrappers_slow
else:
    unwrap_dead_wrappers = _unwrap_dead_wrappers_fast


@contextlib.contextmanager
def enable_single_level_autograd_function() -> Generator[None, None, None]:
    try:
        prev_state = get_single_level_autograd_function_allowed()
        set_single_level_autograd_function_allowed(True)
        yield
    finally:
        set_single_level_autograd_function_allowed(prev_state)


argnums_t = int | tuple[int, ...]
