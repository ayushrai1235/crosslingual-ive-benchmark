"""
CLI script: Generate candidate experimental scenarios across 5 humanitarian domains.
Usage: python experiments/generate_candidates.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scenario_generator import ScenarioGenerator
from src.logging_utils import logger


def main():
    logger.info("Generating candidate experimental scenarios across 5 domains...")
    generator = ScenarioGenerator(output_dir="data/scenarios")
    scenarios = generator.generate_canonical_seed_scenarios()
    generator.save_scenarios(scenarios)
    logger.info(f"Successfully generated {len(scenarios)} candidate scenarios in data/scenarios/")


if __name__ == "__main__":
    main()
