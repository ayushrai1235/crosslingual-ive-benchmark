"""
CLI script: Freeze dataset and generate cryptographic SHA-256 manifest.
Usage: python experiments/freeze_dataset.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_manager import create_dataset_manifest, verify_dataset_integrity
from src.logging_utils import logger


def main():
    logger.info("Freezing experimental dataset and computing cryptographic SHA-256 manifest...")
    manifest = create_dataset_manifest(
        scenarios_dir="data/scenarios",
        translations_dir="data/translations",
        manifest_path="data/dataset_manifest.json"
    )

    is_valid, errors = verify_dataset_integrity("data/dataset_manifest.json")
    if not is_valid:
        logger.error(f"Integrity check failed: {errors}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DATASET FREEZE COMPLETE & VERIFIED")
    print("=" * 60)
    print(f"Manifest Version : {manifest.manifest_version}")
    print(f"Total Scenarios  : {manifest.scenario_count}")
    print(f"Total Files      : {len(manifest.file_hashes)}")
    print(f"Languages        : {', '.join(manifest.languages)}")
    print(f"Conditions       : {', '.join(manifest.conditions)}")
    print(f"Manifest Path    : data/dataset_manifest.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
