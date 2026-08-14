"""
Judge Runner module.
Executes standardized evaluation of 9 LLM judges on cross-lingual scenarios.
Streams raw outputs and parsed numeric allocations to JSONL.
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.schemas import Scenario, JudgmentRaw
from src.config import ModelEntry
from src.model_runner import BaseModelRunner, get_model_runner
from src.response_parser import ResponseParser
from src.logging_utils import logger


class JudgeRunner:
    """Executes moral resource allocation benchmark on LLM judges."""

    def __init__(
        self,
        prompt_template_path: str | Path = "prompts/judge.txt",
        output_dir: str | Path = "data/judgments",
        budget: float = 100.0,
        min_allocation: float = 0.0,
        max_allocation: float = 100.0
    ):
        self.prompt_template_path = Path(prompt_template_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.parser = ResponseParser(min_allocation=min_allocation, max_allocation=max_allocation)
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            return f.read()

    def format_prompt(self, scenario_text: str, intervention_cost: float) -> str:
        """Formats the standardized judge prompt."""
        cost_str = str(int(intervention_cost)) if intervention_cost.is_integer() else str(intervention_cost)
        budget_str = str(int(self.budget)) if self.budget.is_integer() else str(self.budget)
        return self.prompt_template.format(
            scenario_text=scenario_text,
            intervention_cost=cost_str,
            budget=budget_str
        )

    def run_model_evaluation(
        self,
        model_entry: ModelEntry,
        scenarios: List[Scenario],
        languages: List[str] | None = None,
        use_mock: bool = False
    ) -> tuple[List[JudgmentRaw], Dict[str, int]]:
        """
        Runs complete factorial evaluation for a given model across scenarios, languages, and conditions.
        Enforces fail-fast ML environment checks before creating any files.
        Streams raw judgments to JSONL and returns judgments along with exact summary metrics.
        """
        if not use_mock:
            try:
                import torch
                import transformers
            except ImportError as e:
                raise RuntimeError(
                    f"ML execution environment check failed: {e}. 'torch' and 'transformers' are required "
                    "for real empirical model evaluations. Empirical inference must be executed on Kaggle GPU. "
                    "Stopping immediately before generating any judgment records."
                ) from e

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU not available. Empirical model evaluation requires an NVIDIA GPU (e.g. Kaggle T4/P100/A100). "
                    "Stopping immediately before generating any judgment records."
                )

        if languages is None:
            languages = ["en", "hi", "es"]

        conditions = ["identifiable", "statistical"]
        runner = get_model_runner(model_entry, use_mock=use_mock)

        # Separate mock outputs to prevent contamination of empirical data
        prefix = "mock_" if use_mock else ""
        out_file = self.output_dir / f"{prefix}{model_entry.id}_judgments.jsonl"

        judgments: List[JudgmentRaw] = []
        run_id = str(uuid.uuid4())[:8]

        stats: Dict[str, int] = {
            "requested_judgments": 0,
            "inference_attempts": 0,
            "successful_inferences": 0,
            "valid_parsed_judgments": 0,
            "inference_failures": 0,
            "parse_failures": 0,
        }

        # Calculate requested judgments upfront
        for scenario in scenarios:
            for lang in languages:
                for cond in conditions:
                    target_dict = getattr(scenario, cond)
                    if lang in target_dict:
                        stats["requested_judgments"] += 1

        logger.info(
            f"Starting evaluation: Model='{model_entry.name}', Scenarios={len(scenarios)}, "
            f"Langs={languages}, Conditions={conditions}, Requested={stats['requested_judgments']}, Output='{out_file.name}'"
        )

        try:
            with open(out_file, "a", encoding="utf-8") as jsonl_file:
                for scenario in scenarios:
                    for lang in languages:
                        for cond in conditions:
                            target_dict = getattr(scenario, cond)
                            if lang not in target_dict:
                                logger.warning(f"Scenario {scenario.scenario_id} missing text for lang '{lang}' and condition '{cond}'.")
                                continue

                            scenario_text = target_dict[lang]
                            prompt = self.format_prompt(scenario_text, scenario.canonical_facts.intervention_cost)
                            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

                            error_msg = None
                            raw_response = ""
                            success = False
                            parsed_val = None
                            parse_method = "rejected"

                            stats["inference_attempts"] += 1
                            try:
                                raw_response = runner.generate(prompt)
                                stats["successful_inferences"] += 1
                                parsed_val, parse_method, rej_reason = self.parser.parse(raw_response)
                                if parsed_val is not None:
                                    success = True
                                    stats["valid_parsed_judgments"] += 1
                                else:
                                    stats["parse_failures"] += 1
                                    error_msg = rej_reason
                            except Exception as e:
                                stats["inference_failures"] += 1
                                error_msg = f"Inference failure: {e}"
                                logger.error(f"Inference error for {model_entry.id} on {scenario.scenario_id} ({lang}, {cond}): {e}")

                            judgment = JudgmentRaw(
                                run_id=run_id,
                                scenario_id=scenario.scenario_id,
                                model_id=model_entry.id,
                                model_family=model_entry.family,
                                category=model_entry.category,
                                language=lang,
                                victim_condition=cond,
                                prompt_hash=prompt_hash,
                                raw_prompt=prompt,
                                raw_response=raw_response,
                                parsed_allocation=parsed_val,
                                parse_method=parse_method,
                                temperature=model_entry.temperature,
                                seed=42,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                hardware="cuda" if not use_mock else "mock_cpu",
                                quantization=model_entry.quantization if not use_mock else "none",
                                success=success,
                                error_message=error_msg
                            )

                            # Stream record only if inference succeeded or was attempted
                            jsonl_file.write(judgment.model_dump_json() + "\n")
                            jsonl_file.flush()
                            judgments.append(judgment)

            logger.info(
                f"Evaluation summary for '{model_entry.name}': "
                f"Requested={stats['requested_judgments']}, "
                f"Attempts={stats['inference_attempts']}, "
                f"Successful={stats['successful_inferences']}, "
                f"ValidParsed={stats['valid_parsed_judgments']}, "
                f"InferenceFailures={stats['inference_failures']}, "
                f"ParseFailures={stats['parse_failures']}"
            )
        finally:
            runner.unload()

        return judgments, stats
