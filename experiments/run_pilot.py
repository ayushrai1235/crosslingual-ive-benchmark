"""
Master CLI Benchmark Runner for Cross-Lingual Identifiable Victim Effect (IVE).
Executes the preregistered 9-model judge panel across English, Hindi, and Spanish.

Experimental Stages:
- Smoke Test       : 1 scenario  × 1 language  × 2 conditions × 1 model  = 2 judgments
- Software Pilot   : 3 scenarios  × 3 languages × 2 conditions × 1 model  = 18 judgments (mock)
- Scientific Pilot : 10 scenarios × 3 languages × 2 conditions × 9 models = 540 judgments
- Full Benchmark   : 20 scenarios × 3 languages × 2 conditions × 9 models = 1,080 judgments

Usage:
  python experiments/run_pilot.py --smoke-test
  python experiments/run_pilot.py --scientific-pilot
  python experiments/run_pilot.py --full-benchmark
  python experiments/run_pilot.py --model llama_3_1_8b
  python experiments/run_pilot.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import Scenario
from src.config import get_experiment_config, ModelEntry
from src.model_registry import ModelRegistry
from src.judge_runner import JudgeRunner
from src.dataset_manager import verify_dataset_integrity
from src.reproducibility import set_seed, save_environment_metadata
from src.logging_utils import logger


def load_scenarios(scenarios_dir: str | Path = "data/scenarios", count: int | None = None) -> List[Scenario]:
    path = Path(scenarios_dir)
    files = sorted(list(path.glob("*.json")))
    if count is not None:
        files = files[:count]
    scenarios = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            scenarios.append(Scenario(**json.load(fp)))
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Master LLM Judge Benchmark Runner.")
    parser.add_argument("--smoke-test", action="store_true", help="Environment validation: 1 scenario × 1 lang × 2 conds × 1 model = 2 judgments.")
    parser.add_argument("--scientific-pilot", action="store_true", help="Stage 2: Run 10 scenarios across 9 models (540 judgments).")
    parser.add_argument("--full-benchmark", action="store_true", help="Stage 3: Run full 20 scenarios across 9 models (1,080 judgments).")
    parser.add_argument("--model", type=str, default=None, help="Run specific model ID (e.g. qwen_2_5_7b).")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of model IDs to run.")
    parser.add_argument("--exclude-models", type=str, default=None, help="Comma-separated list of model IDs to skip (e.g. llama_3_1_8b when HF access is pending).")
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language codes (default: en,hi,es or en for smoke-test).")
    parser.add_argument("--dry-run", action="store_true", help="Run with isolated mock runner for testing.")
    parser.add_argument("--resume", action="store_true", help="Resume from previous checkpoints.")
    parser.add_argument("--skip-integrity-check", action="store_true", help="Skip dataset SHA-256 verification.")
    args = parser.parse_args()

    # 1. Set global deterministic seed
    set_seed(42)
    save_environment_metadata("results/reproducibility_metadata.json")

    # 2. Verify Dataset Freeze Manifest
    if not args.skip_integrity_check:
        logger.info("Verifying dataset SHA-256 integrity before execution...")
        is_valid, errors = verify_dataset_integrity("data/dataset_manifest.json")
        if not is_valid:
            logger.error(f"Dataset integrity verification failed: {errors}. Aborting run.")
            sys.exit(1)

    exp_config = get_experiment_config()
    registry = ModelRegistry()

    # Model resolution helper
    all_enabled_models = registry.list_models(enabled_only=True)
    excluded_ids = set([m.strip() for m in args.exclude_models.split(",")]) if args.exclude_models else set()

    if args.model:
        selected_models = [registry.get_model(args.model)]
    elif args.models:
        target_ids = [m.strip() for m in args.models.split(",")]
        selected_models = [registry.get_model(mid) for mid in target_ids]
    else:
        selected_models = [m for m in all_enabled_models if m.id not in excluded_ids]

    if excluded_ids:
        logger.info(f"Explicitly excluded models: {list(excluded_ids)} (e.g. pending access). No substitutions applied.")

    # Determine Scenarios, Models, and Languages
    if args.smoke_test:
        scenarios = load_scenarios(count=1)
        # For smoke test: default to first selected model (e.g., qwen_2_5_7b or specified model)
        models = selected_models[:1]
        languages = ["en"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Environment Smoke Test (1 scenario × {len(languages)} lang × 2 conds × {len(models)} model = {len(scenarios) * len(languages) * 2 * len(models)} judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments/smoke_test"
    elif args.scientific_pilot:
        scenarios = load_scenarios(count=10)
        models = selected_models
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Stage 2: Scientific Pilot (10 scenarios × {len(languages)} langs × 2 conds × {len(models)} models = {len(scenarios) * len(languages) * 2 * len(models)} judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"
    elif args.full_benchmark:
        scenarios = load_scenarios(count=20)
        models = selected_models
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Stage 3: Full Benchmark (20 scenarios × {len(languages)} langs × 2 conds × {len(models)} models = {len(scenarios) * len(languages) * 2 * len(models)} judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"
    elif args.model or args.models:
        scenarios = load_scenarios()
        models = selected_models
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Selected Model Run: {[m.id for m in models]} ({len(scenarios)} scenarios)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"
    else:
        logger.info("No stage flag specified. Defaulting to --scientific-pilot.")
        scenarios = load_scenarios(count=10)
        models = selected_models
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Stage 2: Scientific Pilot (10 scenarios × {len(languages)} langs × 2 conds × {len(models)} models = {len(scenarios) * len(languages) * 2 * len(models)} judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"

    logger.info(f"Initiating {stage_name}: {len(scenarios)} scenarios, {len(models)} models, langs={languages}.")

    judge_runner = JudgeRunner(
        prompt_template_path="prompts/judge.txt",
        output_dir=output_dir
    )

    total_judgments = []
    model_manifests: List[Dict[str, Any]] = []
    cumulative_stats: Dict[str, int] = {
        "requested_judgments": 0,
        "inference_attempts": 0,
        "successful_inferences": 0,
        "valid_parsed_judgments": 0,
        "inference_failures": 0,
        "parse_failures": 0,
        "skipped_completed": 0,
    }

    # Sequential execution: one model at a time with clean unloading and isolated error handling
    for i, model_entry in enumerate(models, 1):
        logger.info(f"[{i}/{len(models)}] Running Judge: {model_entry.name} ({model_entry.family}, arch={model_entry.architecture})")
        try:
            judgments, stats = judge_runner.run_model_evaluation(
                model_entry=model_entry,
                scenarios=scenarios,
                languages=languages,
                use_mock=args.dry_run,
                resume=args.resume
            )
            total_judgments.extend(judgments)
            model_manifests.append(stats)
            for k in cumulative_stats.keys():
                if k in stats and isinstance(stats[k], int):
                    cumulative_stats[k] += stats[k]
        except Exception as e:
            err_str = str(e).lower()
            is_gated = any(k in err_str for k in ["gated", "401", "403", "restricted", "access", "llama-3.1", "unauthorized"])
            status = "PENDING_ACCESS" if is_gated else "FAILED"
            logger.error(f"Isolated model failure for {model_entry.id} ({status}): {e}")
            
            # Calculate requested count for this model
            req_count = len(scenarios) * len(languages) * 2
            failed_manifest = {
                "model_id": model_entry.id,
                "model_name": model_entry.name,
                "model_family": model_entry.family,
                "category": model_entry.category,
                "requested_judgments": req_count,
                "inference_attempts": 0,
                "successful_inferences": 0,
                "valid_parsed_judgments": 0,
                "inference_failures": 0,
                "parse_failures": 0,
                "skipped_completed": 0,
                "execution_status": status,
                "failure_reason": str(e),
                "included_in_analysis": False,
                "completed_at": None
            }
            model_manifests.append(failed_manifest)
            cumulative_stats["requested_judgments"] += req_count

    # Save Execution Manifest and Model Coverage Table
    manifest_dir = Path("results/tables")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    execution_manifest = {
        "stage": stage_name,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "total_requested": cumulative_stats["requested_judgments"],
        "total_valid": cumulative_stats["valid_parsed_judgments"],
        "total_included_models": sum(1 for m in model_manifests if m.get("included_in_analysis")),
        "total_pending_models": sum(1 for m in model_manifests if m.get("execution_status") == "PENDING_ACCESS"),
        "total_failed_models": sum(1 for m in model_manifests if m.get("execution_status") == "FAILED"),
        "models": model_manifests
    }

    manifest_json_path = manifest_dir / "execution_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(execution_manifest, f, indent=2)
    logger.info(f"Saved execution manifest to {manifest_json_path}")

    # Export Model Coverage CSV
    try:
        import pandas as pd
        df_cov = pd.DataFrame(model_manifests)
        cov_csv_path = manifest_dir / "model_coverage.csv"
        df_cov.to_csv(cov_csv_path, index=False)
        logger.info(f"Saved model coverage table to {cov_csv_path}")
    except Exception as e:
        logger.warning(f"Could not export model_coverage.csv: {e}")

    print("\n" + "=" * 80)
    print(f"BENCHMARK EXECUTION SUMMARY: {stage_name}")
    print("=" * 80)
    print(f"1. Total Requested Judgments : {cumulative_stats['requested_judgments']}")
    print(f"2. Total Inference Attempts  : {cumulative_stats['inference_attempts']}")
    print(f"3. Total Successful Inferences: {cumulative_stats['successful_inferences']}")
    print(f"4. Total Valid Judgments     : {cumulative_stats['valid_parsed_judgments']} ({cumulative_stats['valid_parsed_judgments']/max(cumulative_stats['requested_judgments'], 1)*100:.1f}%)")
    print(f"5. Skipped (Already Valid)   : {cumulative_stats['skipped_completed']}")
    print(f"6. Inference Failures        : {cumulative_stats['inference_failures']}")
    print(f"7. Parse Failures            : {cumulative_stats['parse_failures']}")
    print(f"8. Active Included Models    : {execution_manifest['total_included_models']}/{len(models)}")
    print(f"9. Pending Access Models     : {execution_manifest['total_pending_models']}/{len(models)}")
    print("-" * 80)
    print(f"{'Model ID':<18} | {'Status':<14} | {'Req':<4} | {'Valid':<5} | {'Included':<8} | Notes")
    print("-" * 80)
    for m in model_manifests:
        notes = m.get("failure_reason") or "OK"
        if len(notes) > 30:
            notes = notes[:27] + "..."
        print(f"{m['model_id']:<18} | {m['execution_status']:<14} | {m['requested_judgments']:<4} | {m['valid_parsed_judgments']:<5} | {str(m['included_in_analysis']):<8} | {notes}")
    print("=" * 80)

    if args.smoke_test:
        if cumulative_stats['successful_inferences'] == 2 and cumulative_stats['valid_parsed_judgments'] == 2:
            print("\n>>> SMOKE TEST PASSED: 2 successful inferences and 2 valid parsed allocations recorded.")
            print(">>> Environment is healthy and ready for scientific pilot.\n")
        else:
            print(f"\n>>> SMOKE TEST FAILED: Expected 2 successful inferences and 2 valid parsed allocations, but got {cumulative_stats['successful_inferences']} inferences and {cumulative_stats['valid_parsed_judgments']} valid parsed allocations.")
            sys.exit(1)


if __name__ == "__main__":
    main()

