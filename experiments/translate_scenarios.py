"""
CLI script: Multilingual translation pipeline (English -> Hindi & Spanish).
Usage: python experiments/translate_scenarios.py
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import Scenario
from src.translator import ScenarioTranslator
from src.logging_utils import logger


def main():
    logger.info("Applying gold-standard cross-lingual translations (hi, es)...")
    scenarios_dir = Path("data/scenarios")
    scenario_files = sorted(list(scenarios_dir.glob("*.json")))

    if not scenario_files:
        logger.error("No scenario files found to translate. Run experiments/generate_candidates.py first.")
        sys.exit(1)

    scenarios = []
    for f in scenario_files:
        with open(f, "r", encoding="utf-8") as fp:
            scenarios.append(Scenario(**json.load(fp)))

    translator = ScenarioTranslator(translations_dir="data/translations")
    translated_scenarios = translator.apply_canonical_translations(scenarios)

    # Save enriched scenarios back to data/scenarios/
    for s in translated_scenarios:
        f_path = scenarios_dir / f"{s.scenario_id}.json"
        with open(f_path, "w", encoding="utf-8") as fp:
            fp.write(s.model_dump_json(indent=2))

    # Save translation audit manifest to data/translations/
    translator.save_translations_manifest(translated_scenarios)
    logger.info(f"Successfully translated {len(translated_scenarios)} scenarios into Hindi and Spanish.")


if __name__ == "__main__":
    main()
