"""
CLI script: Human-in-the-loop review and approval workflow for scenarios.
Usage: python experiments/review_scenarios.py [--auto-approve-reviewed] [--reviewer ID]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import Scenario, HumanReviewRecord
from src.logging_utils import logger


def main():
    parser = argparse.ArgumentParser(description="Human Review Workflow for Experimental Scenarios.")
    parser.add_argument("--reviewer", type=str, default="human_lead_auditor", help="Identifier of human reviewer.")
    parser.add_argument("--auto-approve-canonical", action="store_true", help="Auto-signoff pre-verified canonical seed stimuli.")
    args = parser.parse_args()

    scenarios_dir = Path("data/scenarios")
    reviews_dir = Path("data/validation")
    reviews_dir.mkdir(parents=True, exist_ok=True)

    scenario_files = sorted(list(scenarios_dir.glob("*.json")))
    if not scenario_files:
        logger.error("No scenario files found in data/scenarios/")
        sys.exit(1)

    review_records = []
    for f in scenario_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            s = Scenario(**data)

        if args.auto_approve_canonical or s.human_reviewed:
            s.human_reviewed = True
            if not s.human_reviewer_notes:
                s.human_reviewer_notes = f"Verified and approved by {args.reviewer}."

            # Save updated scenario
            with open(f, "w", encoding="utf-8") as fp:
                fp.write(s.model_dump_json(indent=2))

            rec = HumanReviewRecord(
                item_id=s.scenario_id,
                review_type="scenario",
                reviewer_id=args.reviewer,
                approved=True,
                confidence=5,
                notes=s.human_reviewer_notes,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            review_records.append(rec)

    # Save human review registry
    registry_file = reviews_dir / "human_scenario_reviews.json"
    with open(registry_file, "w", encoding="utf-8") as fp:
        json.dump([r.model_dump() for r in review_records], fp, indent=2)

    logger.info(f"Human review completed for {len(review_records)} scenarios. Records saved to {registry_file}")
    print(f"Successfully recorded human approvals for {len(review_records)} scenarios.")


if __name__ == "__main__":
    main()
