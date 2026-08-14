"""
Data Ingestion and Missingness Analysis module.
Loads raw streaming JSONL judgment logs, validates against Pydantic schema,
and generates parse-rate and missingness audit reports.
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from src.schemas import JudgmentRaw
from src.logging_utils import logger


def load_raw_judgments(judgments_dir: str | Path = "data/judgments", include_mock: bool = False) -> pd.DataFrame:
    """
    Loads all JSONL judgment records from the specified directory.
    Guarantees mock data is excluded by default.
    """
    dir_path = Path(judgments_dir)
    if not dir_path.exists():
        logger.warning(f"Judgments directory '{judgments_dir}' does not exist.")
        return pd.DataFrame()

    jsonl_files = sorted(list(dir_path.glob("*.jsonl")))
    records: List[Dict[str, Any]] = []

    for j_file in jsonl_files:
        is_mock_file = j_file.name.startswith("mock_")
        if is_mock_file and not include_mock:
            logger.info(f"Skipping test mock file: {j_file.name}")
            continue

        with open(j_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    j = JudgmentRaw(**data)
                    records.append(j.model_dump())
                except Exception as e:
                    logger.error(f"Error parsing line {line_idx} in {j_file.name}: {e}")

    if not records:
        logger.warning("No judgment records found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    # Deduplicate: if multiple records exist for (model_id, scenario_id, language, victim_condition), keep the last record
    if not df.empty:
        if "timestamp" in df.columns:
            df = df.sort_values(by="timestamp")
        initial_len = len(df)
        df = df.drop_duplicates(subset=["model_id", "scenario_id", "language", "victim_condition"], keep="last")
        if len(df) < initial_len:
            logger.info(f"Deduplicated judgment records: removed {initial_len - len(df)} duplicate records.")

    logger.info(f"Loaded {len(df)} total raw judgment records from {len(jsonl_files)} files.")
    return df


def generate_missingness_report(df: pd.DataFrame, output_path: str | Path = "results/tables/missingness_report.csv") -> pd.DataFrame:
    """Generates per-model parse success, missingness, and rejection summary table."""
    if df.empty:
        logger.warning("Empty dataframe provided to missingness report.")
        return pd.DataFrame()

    grouped = df.groupby(["model_id", "model_family", "category", "language"]).agg(
        total_trials=("scenario_id", "count"),
        parsed_trials=("parsed_allocation", lambda s: s.notna().sum()),
        rejected_trials=("parsed_allocation", lambda s: s.isna().sum()),
        strict_json_count=("parse_method", lambda s: (s == "strict_json").sum()),
        fenced_json_count=("parse_method", lambda s: (s == "fenced_json").sum()),
        labeled_field_count=("parse_method", lambda s: (s == "labeled_field").sum()),
    ).reset_index()

    grouped["parse_rate_pct"] = (grouped["parsed_trials"] / grouped["total_trials"]) * 100.0
    grouped["missingness_pct"] = (grouped["rejected_trials"] / grouped["total_trials"]) * 100.0

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_file, index=False)
    logger.info(f"Missingness and parse-rate report saved to {out_file}")
    return grouped


def generate_model_coverage_from_judgments(
    df: pd.DataFrame,
    expected_models: Optional[List[str]] = None,
    output_path: str | Path = "results/tables/model_coverage.csv"
) -> pd.DataFrame:
    """Generates a high-level summary of model coverage and inclusion status."""
    if df.empty and not expected_models:
        return pd.DataFrame()

    if expected_models is None:
        expected_models = [
            "llama_3_1_8b", "qwen3_8b", "qwen_2_5_7b", "gemma_3_4b",
            "gemma_3_12b", "aya_expanse_8b", "command_r7b", "bloomz_7b1_mt", "mt0_xl"
        ]

    rows = []
    for mid in expected_models:
        sub = df[df["model_id"] == mid] if not df.empty else pd.DataFrame()
        total_trials = len(sub)
        valid_trials = sub["parsed_allocation"].notna().sum() if total_trials > 0 else 0
        
        if total_trials == 0:
            status = "PENDING_ACCESS" if mid == "llama_3_1_8b" else "NOT_RUN"
            included = False
            notes = "Gated access pending approval" if mid == "llama_3_1_8b" else "No evaluation records found"
        elif valid_trials == total_trials and total_trials > 0:
            status = "COMPLETE"
            included = True
            notes = "All trials validly parsed"
        elif valid_trials > 0:
            status = "PARTIAL"
            included = True
            notes = f"{valid_trials}/{total_trials} valid parsed judgments"
        else:
            status = "FAILED"
            included = False
            notes = "Zero valid allocations parsed"

        rows.append({
            "model_id": mid,
            "total_trials": int(total_trials),
            "valid_trials": int(valid_trials),
            "execution_status": status,
            "included_in_analysis": included,
            "notes": notes
        })

    cov_df = pd.DataFrame(rows)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cov_df.to_csv(out_file, index=False)
    logger.info(f"Model coverage table saved to {out_file}")
    return cov_df

