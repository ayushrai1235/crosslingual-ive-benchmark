"""
Robustness Checks and Sensitivity Analysis module.
Assesses IVE consistency across domains, intervention cost scaling,
and models' language control comprehension metrics.
"""

from pathlib import Path
import json
import pandas as pd
from scipy import stats
from src.logging_utils import logger


def evaluate_domain_robustness(
    df_paired: pd.DataFrame,
    scenarios_metadata_path: str | Path = "data/dataset_manifest.json",
    output_path: str | Path = "results/tables/domain_robustness.csv"
) -> pd.DataFrame:
    """Evaluates IVE variation across the 5 humanitarian domains."""
    if df_paired.empty:
        return pd.DataFrame()

    # Load domain mapping from scenarios
    scenarios_dir = Path("data/scenarios")
    domain_map = {}
    if scenarios_dir.exists():
        for f in scenarios_dir.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                domain_map[data["scenario_id"]] = data.get("domain", "unknown")

    df_copy = df_paired.copy()
    df_copy["domain"] = df_copy["scenario_id"].map(domain_map).fillna("general")

    grouped = df_copy.groupby(["domain", "language"])["ive"].agg(
        n_trials="count",
        mean_ive="mean",
        std_ive="std",
        median_ive="median"
    ).reset_index()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_file, index=False)
    logger.info(f"Domain robustness table saved to {out_file}")

    return grouped


def correlate_with_language_control(
    df_paired: pd.DataFrame,
    control_results_path: str | Path = "results/tables/language_control_results.json",
    output_path: str | Path = "results/tables/language_control_correlation.csv"
) -> pd.DataFrame:
    """Evaluates whether model language control accuracy correlates with cross-lingual IVE attenuation."""
    ctrl_path = Path(control_results_path)
    if not ctrl_path.exists() or df_paired.empty:
        logger.warning("Language control results or paired data missing.")
        return pd.DataFrame()

    with open(ctrl_path, "r", encoding="utf-8") as f:
        ctrl_data = json.load(f)

    ctrl_df = pd.DataFrame(ctrl_data)
    if ctrl_df.empty:
        return pd.DataFrame()

    # Compute comprehension accuracy per model and language
    acc_df = ctrl_df.groupby(["model_id", "language"])["is_correct"].mean().reset_index()
    acc_df.rename(columns={"is_correct": "comprehension_accuracy"}, inplace=True)

    # Compute mean IVE per model and language
    ive_df = df_paired.groupby(["model_id", "language"])["ive"].mean().reset_index()

    merged = pd.merge(acc_df, ive_df, on=["model_id", "language"])

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False)
    logger.info(f"Language control correlation table saved to {out_file}")

    return merged
