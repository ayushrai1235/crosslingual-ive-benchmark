"""
End-to-end pipeline integration test.
Executes the full chain in test mode:
generation -> validation -> translation -> audit -> freeze -> mock inference -> parsing -> IVE.
Ensures zero mock leakage to empirical results.
"""

from pathlib import Path
import json
import pytest
from src.scenario_generator import ScenarioGenerator
from src.scenario_validator import ScenarioValidator
from src.translator import ScenarioTranslator
from src.translation_validator import TranslationValidator
from src.dataset_manager import create_dataset_manifest, verify_dataset_integrity
from src.model_registry import ModelRegistry
from src.judge_runner import JudgeRunner
from analysis.compute_ive import compute_paired_ive
from analysis.load_results import load_raw_judgments


def test_full_pipeline_flow(tmp_path):
    scenarios_dir = tmp_path / "scenarios"
    translations_dir = tmp_path / "translations"
    validation_dir = tmp_path / "validation"
    judgments_dir = tmp_path / "judgments"
    manifest_path = tmp_path / "dataset_manifest.json"

    scenarios_dir.mkdir(parents=True)
    translations_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    judgments_dir.mkdir(parents=True)

    # 1. Generate test scenarios
    gen = ScenarioGenerator(output_dir=scenarios_dir)
    scenarios = gen.generate_canonical_seed_scenarios()
    test_scenarios = scenarios[:3]  # Take 3 for test
    assert len(test_scenarios) == 3

    # Save test scenarios to temporary scenarios dir
    for s in test_scenarios:
        with open(scenarios_dir / f"{s.scenario_id}.json", "w", encoding="utf-8") as fp:
            fp.write(s.model_dump_json(indent=2))

    # 2. Validate scenarios
    s_val = ScenarioValidator(validation_dir=validation_dir)
    all_s_pass, s_audits = s_val.validate_all(test_scenarios)
    assert all_s_pass is True

    # 3. Translate scenarios
    trans = ScenarioTranslator(translations_dir=translations_dir)
    translated_scenarios = trans.apply_canonical_translations(test_scenarios)
    for s in translated_scenarios:
        with open(scenarios_dir / f"{s.scenario_id}.json", "w", encoding="utf-8") as fp:
            fp.write(s.model_dump_json(indent=2))

    # 4. Validate translations
    t_val = TranslationValidator(validation_dir=validation_dir)
    all_t_pass, t_audits = t_val.validate_all(translated_scenarios)
    assert all_t_pass is True

    # 5. Freeze dataset
    manifest = create_dataset_manifest(
        scenarios_dir=scenarios_dir,
        translations_dir=translations_dir,
        manifest_path=manifest_path
    )
    is_valid, errors = verify_dataset_integrity(manifest_path)
    assert is_valid is True

    # 6. Run Mock Judge Runner
    registry = ModelRegistry()
    test_model = registry.get_model("llama_3_1_8b")

    judge_runner = JudgeRunner(
        prompt_template_path="prompts/judge.txt",
        output_dir=judgments_dir
    )

    judgments, stats = judge_runner.run_model_evaluation(
        model_entry=test_model,
        scenarios=translated_scenarios,
        languages=["en", "hi", "es"],
        use_mock=True
    )
    assert len(judgments) == 3 * 3 * 2  # 3 scenarios * 3 langs * 2 conditions = 18
    assert stats["valid_parsed_judgments"] == 18

    # 7. Non-leakage check: verify files are named mock_...
    j_files = list(judgments_dir.glob("*.jsonl"))
    assert len(j_files) == 1
    assert j_files[0].name.startswith("mock_")

    # 8. Compute IVE on test mock outputs
    df_raw = load_raw_judgments(judgments_dir, include_mock=True)
    df_paired = compute_paired_ive(df_raw, output_path=tmp_path / "paired.csv")
    assert len(df_paired) == 9  # 3 scenarios * 3 languages
