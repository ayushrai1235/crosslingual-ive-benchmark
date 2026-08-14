"""
Model Registry module.
Loads, validates, and manages the pre-registered 9 LLM judge models.
"""

from typing import List, Dict, Optional
from pathlib import Path
from src.config import get_models_config, ModelEntry
from src.logging_utils import logger


class ModelRegistry:
    """Registry managing the 9 pre-registered LLM judges."""

    def __init__(self, config_path: str | Path = "config/models.yaml"):
        self.config_path = config_path
        self._models: Dict[str, ModelEntry] = {}
        self.load_models()

    def load_models(self) -> None:
        """Loads and registers models from YAML config."""
        models_list = get_models_config(self.config_path)
        self._models = {m.id: m for m in models_list}
        logger.info(f"Loaded {len(self._models)} models into registry: {list(self._models.keys())}")

    def get_model(self, model_id: str) -> ModelEntry:
        if model_id not in self._models:
            raise KeyError(f"Model ID '{model_id}' not found in registry. Registered: {list(self._models.keys())}")
        return self._models[model_id]

    def list_models(self, enabled_only: bool = True) -> List[ModelEntry]:
        if enabled_only:
            return [m for m in self._models.values() if m.enabled]
        return list(self._models.values())

    def get_models_by_family(self, family: str) -> List[ModelEntry]:
        return [m for m in self._models.values() if m.family.lower() == family.lower()]

    def get_models_by_category(self, category: str) -> List[ModelEntry]:
        return [m for m in self._models.values() if m.category.lower() == category.lower()]
