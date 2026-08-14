"""
Dataset Freeze and Integrity Manager.
Computes and verifies SHA-256 manifests to guarantee stimulus immutability during experiments.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from src.schemas import DatasetManifest, Scenario
from src.logging_utils import logger


def compute_file_sha256(file_path: str | Path, normalize_newlines: bool = True) -> str:
    """
    Computes the SHA-256 checksum of a file.
    When normalize_newlines is True (default), normalizes CRLF (\\r\\n) to LF (\\n) to guarantee
    cross-platform cryptographic determinism across Windows, Linux (Kaggle/Colab), and macOS.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        content = f.read()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    hasher.update(content)
    return hasher.hexdigest()


def create_dataset_manifest(
    scenarios_dir: str | Path = "data/scenarios",
    translations_dir: str | Path = "data/translations",
    manifest_path: str | Path = "data/dataset_manifest.json",
    languages: List[str] | None = None,
    conditions: List[str] | None = None
) -> DatasetManifest:
    """Scans all scenario files, computes cryptographic hashes, and generates a frozen manifest."""
    scenarios_path = Path(scenarios_dir)
    translations_path = Path(translations_dir)
    manifest_file = Path(manifest_path)

    if languages is None:
        languages = ["en", "hi", "es"]
    if conditions is None:
        conditions = ["identifiable", "statistical"]

    file_hashes: Dict[str, str] = {}
    scenario_files = sorted(list(scenarios_path.glob("*.json")))

    for s_file in scenario_files:
        rel_path = s_file.as_posix()
        file_hashes[rel_path] = compute_file_sha256(s_file)

    if translations_path.exists():
        translation_files = sorted(list(translations_path.glob("*.json")))
        for t_file in translation_files:
            rel_path = t_file.as_posix()
            file_hashes[rel_path] = compute_file_sha256(t_file)

    manifest = DatasetManifest(
        manifest_version="1.0.0",
        created_at=datetime.now(timezone.utc).isoformat(),
        frozen=True,
        file_hashes=file_hashes,
        scenario_count=len(scenario_files),
        languages=languages,
        conditions=conditions
    )

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(manifest.model_dump_json(indent=2) + "\n")

    logger.info(f"Dataset manifest created with {len(file_hashes)} file hashes at {manifest_file}")
    return manifest


def load_dataset_manifest(manifest_path: str | Path = "data/dataset_manifest.json") -> DatasetManifest:
    """Loads and parses the dataset manifest."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetManifest(**data)


def verify_dataset_integrity(manifest_path: str | Path = "data/dataset_manifest.json") -> Tuple[bool, List[str]]:
    """
    Verifies that all files recorded in the manifest exist and match their SHA-256 hashes.
    Returns (is_valid, list_of_errors).
    """
    path = Path(manifest_path)
    if not path.exists():
        return False, [f"Manifest file not found at {manifest_path}"]

    manifest = load_dataset_manifest(path)
    errors: List[str] = []

    for rel_path, expected_hash in manifest.file_hashes.items():
        f_path = Path(rel_path)
        if not f_path.exists():
            errors.append(f"Missing file: {rel_path}")
            continue
        actual_hash = compute_file_sha256(f_path)
        if actual_hash != expected_hash:
            errors.append(
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )

    is_valid = len(errors) == 0
    if is_valid:
        logger.info(f"Dataset integrity verified: all {len(manifest.file_hashes)} files intact.")
    else:
        logger.error(f"Dataset integrity FAILED with {len(errors)} error(s): {errors}")

    return is_valid, errors
