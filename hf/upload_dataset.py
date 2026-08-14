"""
Hugging Face Dataset packaging and upload script.
Transforms data/scenarios into Hugging Face Dataset format and publishes to the Hub.
Usage: python hf/upload_dataset.py --repo-id <hf_username/repo_name> [--private]
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_dataset_rows(scenarios_dir: str = "data/scenarios") -> List[Dict[str, Any]]:
    path = Path(scenarios_dir)
    files = sorted(list(path.glob("*.json")))
    rows = []

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            s = json.load(fp)

        # Flatten into tabular rows for multi-lingual and multi-condition evaluation
        for lang in ["en", "hi", "es"]:
            for cond in ["identifiable", "statistical"]:
                if lang == "en":
                    prompt_text = s[f"{cond}_condition"]["text"]
                else:
                    trans = s.get("translations", {}).get(lang, {})
                    prompt_text = trans.get(f"{cond}_text", "")

                rows.append({
                    "scenario_id": s["scenario_id"],
                    "domain": s.get("domain", ""),
                    "language": lang,
                    "condition": cond,
                    "prompt_text": prompt_text,
                    "total_budget": s.get("total_budget", 100.0),
                    "intervention_cost": s.get("intervention_cost", 40.0),
                    "victim_count": s.get("victim_count", 50),
                    "human_reviewed": s.get("human_reviewed", False)
                })

    return rows


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Dataset repository ID (e.g. org/crosslingual-ive).")
    parser.add_argument("--private", action="store_true", help="Set repository to private.")
    args = parser.parse_args()

    try:
        from datasets import Dataset
    except ImportError:
        print("Please install 'datasets' package: pip install datasets")
        return

    rows = load_dataset_rows()
    if not rows:
        print("No scenario records found to package.")
        return

    hf_dataset = Dataset.from_list(rows)
    print(f"Created HF Dataset with {len(hf_dataset)} rows.")
    print(hf_dataset)

    # Push to hub
    print(f"Uploading dataset to Hugging Face Hub: {args.repo_id}...")
    hf_dataset.push_to_hub(
        repo_id=args.repo_id,
        private=args.private
    )
    print("Dataset successfully uploaded!")


if __name__ == "__main__":
    main()
