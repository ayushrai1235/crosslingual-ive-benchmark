"""
Reproducibility and Environment Capture module.
Captures system specs, library versions, hardware config, and seeds for open science.
"""

import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import numpy as np
import yaml
from src.logging_utils import logger


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def capture_environment_metadata() -> Dict[str, Any]:
    """Collects complete execution environment specifications."""
    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "packages": {}
    }

    # Record critical package versions
    packages_to_check = [
        "torch", "transformers", "accelerate", "bitsandbytes",
        "pydantic", "scipy", "statsmodels", "numpy", "pandas",
        "matplotlib", "seaborn", "yaml", "gradio"
    ]
    for pkg_name in packages_to_check:
        try:
            mod = __import__(pkg_name)
            meta["packages"][pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            meta["packages"][pkg_name] = "not_installed"

    # Hardware (CUDA / GPU)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        meta["hardware"] = {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                }
                for i in range(torch.cuda.device_count())
            ] if cuda_available else []
        }
    except Exception as e:
        meta["hardware"] = {"error": str(e)}

    return meta


def save_environment_metadata(output_path: str | Path = "results/reproducibility_metadata.json") -> None:
    """Saves environment metadata to disk."""
    import json
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = capture_environment_metadata()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Reproducibility metadata saved to {path}")
