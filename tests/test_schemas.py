"""
Unit tests for core Pydantic schemas and validation constraints.
"""

import pytest
from pydantic import ValidationError
from src.schemas import (
    ScenarioCanonicalFacts,
    Scenario,
    ScenarioAuditResult,
    TranslationValidationResult,
    JudgmentRaw,
    DatasetManifest,
    PairedScenarioJudgment
)


def test_valid_scenario_creation():
    s = Scenario(
        scenario_id="IVE001",
        domain="medical_aid",
        canonical_facts=ScenarioCanonicalFacts(
            victim_count=50,
            intervention_cost=40.0,
            available_budget=100.0,
            expected_benefit="Prevent permanent organ damage in 50 patients",
            severity=9,
            urgency=9,
            domain="medical_aid"
        ),
        identifiable={
            "en": "David is an 8-year-old child requiring urgent pediatric heart surgery.",
            "hi": "डेविड 8 वर्ष का एक बच्चा है जिसे तत्काल हृदय शल्य चिकित्सा की आवश्यकता है।",
            "es": "David es un niño de 8 años que requiere una cirugía cardíaca pediátrica urgente."
        },
        statistical={
            "en": "A cohort of 50 pediatric patients requires urgent heart surgery.",
            "hi": "50 बाल चिकित्सा रोगियों के एक समूह को तत्काल हृदय शल्य चिकित्सा की आवश्यकता है।",
            "es": "Una cohorte de 50 pacientes pediátricos requiere una cirugía cardíaca urgente."
        }
    )
    assert s.scenario_id == "IVE001"
    assert s.canonical_facts.available_budget == 100.0
    assert s.canonical_facts.intervention_cost == 40.0


def test_scenario_missing_english_validation_error():
    with pytest.raises(ValidationError):
        # Missing 'en' baseline
        Scenario(
            scenario_id="IVE001",
            domain="medical_aid",
            canonical_facts=ScenarioCanonicalFacts(
                victim_count=50,
                intervention_cost=40.0,
                available_budget=100.0,
                expected_benefit="Benefit description",
                severity=9,
                urgency=9,
                domain="medical_aid"
            ),
            identifiable={"hi": "डेविड एक बच्चा है।"},
            statistical={"hi": "50 रोगी हैं।"}
        )


def test_judgment_raw_validation():
    # Valid
    j = JudgmentRaw(
        run_id="RUN_001",
        scenario_id="IVE001",
        model_id="llama_3_1_8b",
        model_family="meta-llama",
        category="general_purpose",
        language="en",
        victim_condition="identifiable",
        prompt_hash="a1b2c3d4",
        raw_prompt="Prompt text",
        raw_response='{"allocation": 75.0, "reasoning": "High need"}',
        parsed_allocation=75.0,
        parse_method="strict_json",
        temperature=0.0,
        seed=42,
        timestamp="2026-08-14T20:00:00Z",
        hardware="cpu",
        quantization="none",
        success=True
    )
    assert j.parsed_allocation == 75.0
