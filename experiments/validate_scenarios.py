"""
CLI script: Automated validation and audit of scenario candidates.
Usage: python experiments/validate_scenarios.py
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import Scenario
from src.scenario_validator import ScenarioValidator
from src.logging_utils import logger


def main():
    logger.info("Starting automated scenario validation audit...")
    scenarios_dir = Path("data/scenarios")
    if not scenarios_dir.exists():
        logger.error(f"Directory {scenarios_dir} does not exist. Run experiments/generate_candidates.py first.")
        sys.exit(1)

    scenario_files = sorted(list(scenarios_dir.glob("*.json")))
    if not scenario_files:
        logger.error("No scenario JSON files found to validate.")
        sys.exit(1)

    scenarios = []
    for f in scenario_files:
        with open(f, "r", encoding="utf-8") as fp:
            scenarios.append(Scenario(**json.load(fp)))

    validator = ScenarioValidator(validation_dir="data/validation")
    all_passed, results = validator.validate_all(scenarios)

    print("\n" + "=" * 60)
    print(f"SCENARIO AUDIT SUMMARY: {len(results)} Scenarios Audited")
    print("=" * 60)
    for r in results:
        status = "PASSED" if r.approved else "FLAGGED"
        print(f"[{status}] Scenario {r.scenario_id}: {r.audit_notes}")
    print("=" * 60)
    print(f"Overall Status: {'ALL PASSED' if all_passed else 'SOME FLAGGED'}\n")

    if not all_passed:
        logger.warning("Some scenarios were flagged during automated audit.")


if __name__ == "__main__":
    main()
