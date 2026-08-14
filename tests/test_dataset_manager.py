"""
Unit tests for cryptographic dataset manifest creation, verification, and tamper detection.
"""

import json
from pathlib import Path
import pytest
from src.schemas import Scenario, ScenarioCanonicalFacts
from src.dataset_manager import create_dataset_manifest, verify_dataset_integrity


@pytest.fixture
def temp_dataset(tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    translations_dir = tmp_path / "translations"
    scenarios_dir.mkdir(parents=True)
    translations_dir.mkdir(parents=True)

    # Create 2 sample scenarios
    for idx in range(1, 3):
        s = Scenario(
            scenario_id=f"IVE{idx:03d}",
            domain="medical_aid",
            canonical_facts=ScenarioCanonicalFacts(
                victim_count=50,
                intervention_cost=40.0,
                available_budget=100.0,
                expected_benefit="Prevent medical complications",
                severity=8,
                urgency=8,
                domain="medical_aid"
            ),
            identifiable={
                "en": f"David {idx} is a child who needs critical medical aid.",
                "hi": f"डेविड {idx} एक बच्चा है जिसे चिकित्सा सहायता की आवश्यकता है।",
                "es": f"David {idx} es un niño que necesita ayuda médica crítica."
            },
            statistical={
                "en": f"50 pediatric patients in region {idx} need critical medical aid.",
                "hi": f"क्षेत्र {idx} में 50 बाल चिकित्सा रोगियों को सहायता की आवश्यकता है।",
                "es": f"50 pacientes pediátricos en la región {idx} necesitan ayuda médica crítica."
            }
        )
        with open(scenarios_dir / f"{s.scenario_id}.json", "w", encoding="utf-8") as f:
            f.write(s.model_dump_json(indent=2))

    return tmp_path, scenarios_dir, translations_dir


def test_manifest_creation_and_integrity_check(temp_dataset):
    tmp_path, scenarios_dir, translations_dir = temp_dataset
    manifest_path = tmp_path / "dataset_manifest.json"

    # Create manifest
    manifest = create_dataset_manifest(
        scenarios_dir=scenarios_dir,
        translations_dir=translations_dir,
        manifest_path=manifest_path
    )

    assert manifest.scenario_count == 2
    assert len(manifest.file_hashes) == 2

    # Verify integrity
    is_valid, errors = verify_dataset_integrity(manifest_path)
    assert is_valid is True
    assert len(errors) == 0


def test_tamper_detection(temp_dataset):
    tmp_path, scenarios_dir, translations_dir = temp_dataset
    manifest_path = tmp_path / "dataset_manifest.json"

    create_dataset_manifest(
        scenarios_dir=scenarios_dir,
        translations_dir=translations_dir,
        manifest_path=manifest_path
    )

    # Tamper with one file
    tampered_file = scenarios_dir / "IVE001.json"
    with open(tampered_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["canonical_facts"]["available_budget"] = 999.0  # Unauthorized modification
    with open(tampered_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Integrity check must catch the discrepancy
    is_valid, errors = verify_dataset_integrity(manifest_path)
    assert is_valid is False
    assert len(errors) > 0
    assert any("mismatch" in err.lower() for err in errors)
