"""
Configuration loading and validation module.
Reads experiment, models, and languages YAML configuration files with strict typing.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from pydantic import BaseModel, Field


class StageConfig(BaseModel):
    scenario_count: int
    enabled_models: int
    description: str


class BootstrapConfig(BaseModel):
    n_resamples: int = 10000
    confidence_level: float = 0.95
    resample_unit: str = "scenario"
    method: str = "percentile"


class StatisticalTestingConfig(BaseModel):
    alpha: float = 0.05
    multiplicity_correction: str = "holm"
    primary_test: str = "paired_wilcoxon"
    secondary_model: str = "linear_mixed_effects"


class TranslationThresholds(BaseModel):
    semantic_equivalence_min: float = 0.85
    quantitative_equivalence_min: float = 1.0
    severity_equivalence_min: float = 0.80
    emotional_equivalence_min: float = 0.75
    factual_equivalence_min: float = 0.90
    cultural_neutrality_min: float = 0.80


class PathsConfig(BaseModel):
    scenarios_dir: str = "data/scenarios"
    translations_dir: str = "data/translations"
    validation_dir: str = "data/validation"
    manifest_path: str = "data/dataset_manifest.json"
    judgments_dir: str = "data/judgments"
    language_control_dir: str = "data/language_control"
    raw_results_dir: str = "results/raw"
    processed_results_dir: str = "results/processed"
    tables_dir: str = "results/tables"
    figures_dir: str = "figures"
    logs_dir: str = "logs"


class ExperimentConfig(BaseModel):
    experiment_name: str
    version: str
    primary_hypothesis: str
    null_hypothesis: str
    languages: List[str]
    conditions: List[str]
    stages: Dict[str, StageConfig]
    budget: float = 100.0
    min_allocation: float = 0.0
    max_allocation: float = 100.0
    temperature: float = 0.0
    max_new_tokens: int = 128
    seed: int = 42
    batch_size: int = 1
    bootstrap: BootstrapConfig
    statistical_testing: StatisticalTestingConfig
    translation_thresholds: TranslationThresholds
    paths: PathsConfig


class ModelEntry(BaseModel):
    id: str
    name: str
    family: str
    category: str
    hf_id: str
    revision: str = "main"
    parameters: str
    license: str
    officially_documented_languages: List[str] = Field(default_factory=list)
    experiment_languages: List[str] = Field(default_factory=lambda: ["en", "hi", "es"])
    language_support_notes: Dict[str, str] = Field(default_factory=dict)
    architecture: str = "causal_lm"  # "causal_lm" | "seq2seq"
    is_gated: bool = False
    disable_thinking: bool = False
    quantization: str = "4bit"
    compute_dtype: str = "bfloat16"
    estimated_vram_gb: float = 6.0
    chat_template: Optional[str] = "plain_prompt"
    temperature: float = 0.0
    max_new_tokens: int = 128
    enabled: bool = True

    # Helper property for backward compatibility with older tests if any
    @property
    def supported_languages(self) -> List[str]:
        return self.officially_documented_languages

    @property
    def dtype(self) -> str:
        return self.compute_dtype

    @property
    def expected_vram_gb(self) -> float:
        return self.estimated_vram_gb


class LanguageEntry(BaseModel):
    name: str
    native_name: str
    iso_639_1: str
    script: str
    direction: str = "ltr"
    family: str
    benchmark_status: str
    notes: str


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_experiment_config(config_path: str | Path = "config/experiment.yaml") -> ExperimentConfig:
    data = load_yaml(config_path)
    return ExperimentConfig(**data)


def get_models_config(config_path: str | Path = "config/models.yaml") -> List[ModelEntry]:
    data = load_yaml(config_path)
    raw_models = data.get("models", [])
    return [ModelEntry(**m) for m in raw_models]


def get_languages_config(config_path: str | Path = "config/languages.yaml") -> Dict[str, LanguageEntry]:
    data = load_yaml(config_path)
    raw_langs = data.get("languages", {})
    return {k: LanguageEntry(**v) for k, v in raw_langs.items()}
