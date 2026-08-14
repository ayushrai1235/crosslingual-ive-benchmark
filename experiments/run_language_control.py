"""
CLI script: Execute language comprehension control battery across English, Hindi, and Spanish.
Evaluates: basic comprehension, numerical comprehension, instruction following, factual understanding.

Outputs: results/tables/language_control.csv
Usage: python experiments/run_language_control.py [--model MODEL_ID] [--mock]
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import LanguageControlItem, LanguageControlResult
from src.model_registry import ModelRegistry
from src.model_runner import get_model_runner
from src.logging_utils import logger


def load_control_items(path: str | Path = "data/language_control/items.json") -> List[LanguageControlItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [LanguageControlItem(**item) for item in data]


def main():
    parser = argparse.ArgumentParser(description="Run language comprehension control battery.")
    parser.add_argument("--model", type=str, default=None, help="Specific model ID to evaluate (or all if omitted).")
    parser.add_argument("--mock", action="store_true", help="Use test MockRunner.")
    args = parser.parse_args()

    items = load_control_items()
    registry = ModelRegistry()
    models = [registry.get_model(args.model)] if args.model else registry.list_models(enabled_only=True)

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "language_control.csv"
    out_json = out_dir / "language_control_results.json"

    all_results: List[LanguageControlResult] = []
    summary_rows: List[Dict[str, Any]] = []

    for model_entry in models:
        logger.info(f"Running language control battery on {model_entry.name} (mock={args.mock})...")
        runner = get_model_runner(model_entry, use_mock=args.mock)
        model_results = []
        try:
            for item in items:
                prompt = (
                    f"You are taking an objective language comprehension and instruction-following test.\n\n"
                    f"QUESTION:\n{item.question}\n\n"
                    f"INSTRUCTIONS:\nAnswer directly. Return strictly a JSON: {{\"answer\": \"<your concise answer>\"}}"
                )
                raw_resp = runner.generate(prompt)
                parsed_ans = ""
                try:
                    data = json.loads(raw_resp.strip())
                    parsed_ans = str(data.get("answer", "")).strip()
                except Exception:
                    parsed_ans = raw_resp.strip()

                is_correct = (parsed_ans.lower() in [v.lower() for v in item.acceptable_variants]) or (item.expected_answer.lower() in parsed_ans.lower())

                res = LanguageControlResult(
                    model_id=model_entry.id,
                    item_id=item.item_id,
                    language=item.language,
                    category=item.category,
                    raw_response=raw_resp,
                    parsed_answer=parsed_ans,
                    is_correct=is_correct,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                model_results.append(res)
                all_results.append(res)
        finally:
            runner.unload()

        # Compute summary scores per language for this model
        for lang in ["en", "hi", "es"]:
            lang_items = [r for r in model_results if r.language == lang]
            total = len(lang_items)
            correct = sum(1 for r in lang_items if r.is_correct)
            acc = round((correct / total) * 100, 1) if total > 0 else 0.0

            # Breakdowns by category
            basic_acc = round(sum(1 for r in lang_items if r.category == "basic_comprehension" and r.is_correct) / max(1, sum(1 for r in lang_items if r.category == "basic_comprehension")) * 100, 1)
            num_acc = round(sum(1 for r in lang_items if r.category == "numerical_comprehension" and r.is_correct) / max(1, sum(1 for r in lang_items if r.category == "numerical_comprehension")) * 100, 1)
            instr_acc = round(sum(1 for r in lang_items if r.category == "instruction_following" and r.is_correct) / max(1, sum(1 for r in lang_items if r.category == "instruction_following")) * 100, 1)
            fact_acc = round(sum(1 for r in lang_items if r.category == "factual_understanding" and r.is_correct) / max(1, sum(1 for r in lang_items if r.category == "factual_understanding")) * 100, 1)

            summary_rows.append({
                "model_id": model_entry.id,
                "model_name": model_entry.name,
                "family": model_entry.family,
                "language": lang,
                "total_items": total,
                "correct_items": correct,
                "accuracy_pct": acc,
                "basic_comprehension_pct": basic_acc,
                "numerical_comprehension_pct": num_acc,
                "instruction_following_pct": instr_acc,
                "factual_understanding_pct": fact_acc,
                "evaluation_mode": "mock" if args.mock else "live_gpu"
            })

    # Write summary CSV
    fieldnames = [
        "model_id", "model_name", "family", "language", "total_items",
        "correct_items", "accuracy_pct", "basic_comprehension_pct",
        "numerical_comprehension_pct", "instruction_following_pct",
        "factual_understanding_pct", "evaluation_mode"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write detailed JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in all_results], f, indent=2, ensure_ascii=False)

    logger.info(f"Language control evaluation completed. CSV saved to {out_csv}")
    print(f"\nLanguage control test completed: {len(all_results)} evaluations recorded across {len(models)} models.")
    print(f"Results written to {out_csv}\n")


if __name__ == "__main__":
    main()
