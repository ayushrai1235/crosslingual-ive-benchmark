"""
CLI script: Automated validation and audit of cross-lingual translations.
Usage: python experiments/validate_translations.py
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import Scenario
from src.translation_validator import TranslationValidator
from src.logging_utils import logger


def main():
    logger.info("Starting automated translation validation audit...")
    scenarios_dir = Path("data/scenarios")
    scenario_files = sorted(list(scenarios_dir.glob("*.json")))

    if not scenario_files:
        logger.error("No scenario files found to validate.")
        sys.exit(1)

    scenarios = []
    for f in scenario_files:
        with open(f, "r", encoding="utf-8") as fp:
            scenarios.append(Scenario(**json.load(fp)))

    validator = TranslationValidator(validation_dir="data/validation")
    all_passed, results = validator.validate_all(scenarios)

    print("\n" + "=" * 70)
    print(f"TRANSLATION AUDIT SUMMARY: {len(results)} Translation Units Audited")
    print("=" * 70)
    for r in results:
        status = "PASSED" if r.approved else "FLAGGED"
        print(
            f"[{status}] Scenario {r.scenario_id} ({r.target_language.upper()} - {r.condition.capitalize()}): "
            f"Semantic={r.semantic_equivalence:.2f}, Quant={r.quantitative_equivalence:.2f}, "
            f"Severity={r.severity_equivalence:.2f}, Reasons={r.reasons}"
        )
    print("=" * 70)
    print(f"Overall Status: {'ALL PASSED' if all_passed else 'SOME FLAGGED'}\n")

    if not all_passed:
        logger.warning("Some translations were flagged during automated audit.")


if __name__ == "__main__":
    main()
