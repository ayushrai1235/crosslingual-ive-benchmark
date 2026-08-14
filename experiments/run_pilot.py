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
    parser.add_argument("--model", type=str, default=None, help="Run specific model ID.")
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

    # Determine Scenarios, Models, and Languages
    if args.smoke_test:
        scenarios = load_scenarios(count=1)
        if args.model:
            models = [registry.get_model(args.model)]
        else:
            models = registry.list_models(enabled_only=True)[:1]
        languages = ["en"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Environment Smoke Test (1 scenario × {len(languages)} lang × 2 conds × {len(models)} model = {len(scenarios) * len(languages) * 2 * len(models)} judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments/smoke_test"
    elif args.scientific_pilot:
        scenarios = load_scenarios(count=10)
        models = registry.list_models(enabled_only=True)
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = "Stage 2: Scientific Pilot (10 scenarios × 3 langs × 2 conds × 9 models = 540 judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"
    elif args.full_benchmark:
        scenarios = load_scenarios(count=20)
        models = registry.list_models(enabled_only=True)
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = "Stage 3: Full Benchmark (20 scenarios × 3 langs × 2 conds × 9 models = 1,080 judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"
    elif args.model:
        scenarios = load_scenarios()
        models = [registry.get_model(args.model)]
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = f"Single Model Run: {args.model} ({len(scenarios)} scenarios)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"
    else:
        logger.info("No stage flag specified. Defaulting to --scientific-pilot.")
        scenarios = load_scenarios(count=10)
        models = registry.list_models(enabled_only=True)
        languages = ["en", "hi", "es"] if not args.languages else [l.strip() for l in args.languages.split(",")]
        stage_name = "Stage 2: Scientific Pilot (10 scenarios × 3 langs × 2 conds × 9 models = 540 judgments)"
        output_dir = "data/judgments/dry_run" if args.dry_run else "data/judgments"

    logger.info(f"Initiating {stage_name}: {len(scenarios)} scenarios, {len(models)} models, langs={languages}.")

    judge_runner = JudgeRunner(
        prompt_template_path="prompts/judge.txt",
        output_dir=output_dir
    )

    total_judgments = []
    cumulative_stats: Dict[str, int] = {
        "requested_judgments": 0,
        "inference_attempts": 0,
        "successful_inferences": 0,
        "valid_parsed_judgments": 0,
        "inference_failures": 0,
        "parse_failures": 0,
    }

    # Sequential execution: one model at a time with clean unloading
    for i, model_entry in enumerate(models, 1):
        logger.info(f"[{i}/{len(models)}] Running Judge: {model_entry.name} ({model_entry.family}, arch={model_entry.architecture})")
        judgments, stats = judge_runner.run_model_evaluation(
            model_entry=model_entry,
            scenarios=scenarios,
            languages=languages,
            use_mock=args.dry_run
        )
        total_judgments.extend(judgments)
        for k, v in stats.items():
            cumulative_stats[k] += v

    print("\n" + "=" * 80)
    print(f"BENCHMARK EXECUTION SUMMARY: {stage_name}")
    print("=" * 80)
    print(f"1. Requested Judgments     : {cumulative_stats['requested_judgments']}")
    print(f"2. Inference Attempts       : {cumulative_stats['inference_attempts']}")
    print(f"3. Successful Inferences   : {cumulative_stats['successful_inferences']}")
    print(f"4. Valid Parsed Judgments  : {cumulative_stats['valid_parsed_judgments']} ({cumulative_stats['valid_parsed_judgments']/max(cumulative_stats['requested_judgments'], 1)*100:.1f}%)")
    print(f"5. Inference Failures      : {cumulative_stats['inference_failures']}")
    print(f"6. Parse Failures          : {cumulative_stats['parse_failures']}")
    print(f"Output Directory           : {output_dir}")
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

