"""Load YAML configs and seed all random number generators from them."""
import random
from pathlib import Path

import numpy as np
import yaml


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    """Read a YAML file. An `inherit:` key names a parent file (relative to this one)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    return _merge(load_config(path.parent / parent), cfg)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
