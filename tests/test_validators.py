"""
Unit tests for scenario and translation structural auditors.
"""

from src.schemas import Scenario, ScenarioCanonicalFacts
from src.scenario_validator import ScenarioValidator
from src.translation_validator import TranslationValidator


def test_scenario_validator():
    validator = ScenarioValidator()

    # Valid scenario
    valid_s = Scenario(
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
            "en": "David is an 8-year-old child who requires urgent surgery costing 40 points.",
            "hi": "डेविड 8 वर्ष का एक बच्चा है जिसे 40 अंकों की लागत से तत्काल सर्जरी की आवश्यकता है।",
            "es": "David es un niño de 8 años que requiere una cirugía urgente que cuesta 40 puntos."
        },
        statistical={
            "en": "A cohort of 50 pediatric patients requires urgent heart surgery costing 40 points.",
            "hi": "50 बाल चिकित्सा रोगियों के एक समूह को 40 अंकों की लागत से सर्जरी की आवश्यकता है।",
            "es": "Una cohorte de 50 pacientes pediátricos requiere una cirugía que cuesta 40 puntos."
        }
    )
    res = validator.validate_scenario(valid_s)
    assert res.approved is True
    assert res.quantitative_match is True


def test_translation_validator():
    validator = TranslationValidator()
    s = Scenario(
        scenario_id="IVE001",
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
            "en": "David is an 8-year-old boy in urgent need of heart surgery costing 40 points.",
            "hi": "डेविड 8 वर्ष का एक बच्चा है जिसे 40 अंकों की लागत से सर्जरी की आवश्यकता है।",
            "es": "David es un niño de 8 años que necesita cirugía urgente que cuesta 40 puntos."
        },
        statistical={
            "en": "A hospital treats 50 pediatric patients requiring surgery costing 40 points.",
            "hi": "एक अस्पताल में 50 बाल चिकित्सा रोगियों का इलाज किया जाता है जिसमें 40 अंकों की लागत आती है।",
            "es": "Un hospital atiende a 50 pacientes pediátricos que requieren cirugía por 40 puntos."
        }
    )

    results = validator.validate_scenario_translations(s)
    assert len(results) == 4  # (hi, es) x (identifiable, statistical)
    assert all(r.approved for r in results)
