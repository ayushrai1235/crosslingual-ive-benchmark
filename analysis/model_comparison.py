"""
Model and Family Comparative Analysis module.
Aggregates IVE by individual model, model family, and architectural category
(general_purpose, multilingual_specialized, reasoning_specialized).
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from scipy import stats
from src.logging_utils import logger


def generate_model_summary(
    df_paired: pd.DataFrame,
    output_path: str | Path = "results/tables/model_summary.csv"
) -> pd.DataFrame:
    """Generates per-model IVE mean, standard deviation, and median across languages."""
    if df_paired.empty:
        return pd.DataFrame()

    grouped = df_paired.groupby(["model_id", "model_family", "category", "language"])["ive"].agg(
        n_trials="count",
        mean_ive="mean",
        std_ive="std",
        median_ive="median",
        iqr_ive=lambda s: s.quantile(0.75) - s.quantile(0.25)
    ).reset_index()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_file, index=False)
    logger.info(f"Model IVE summary saved to {out_file}")
    return grouped


def generate_family_and_category_summary(
    df_paired: pd.DataFrame,
    family_out: str | Path = "results/tables/model_family_summary.csv",
    category_out: str | Path = "results/tables/category_comparison.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates aggregations at the family and category levels."""
    if df_paired.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Family level
    fam_summary = df_paired.groupby(["model_family", "language"])["ive"].agg(
        n_trials="count",
        mean_ive="mean",
        std_ive="std",
        median_ive="median"
    ).reset_index()

    f_out = Path(family_out)
    f_out.parent.mkdir(parents=True, exist_ok=True)
    fam_summary.to_csv(f_out, index=False)

    # Category level
    cat_summary = df_paired.groupby(["category", "language"])["ive"].agg(
        n_trials="count",
        mean_ive="mean",
        std_ive="std",
        median_ive="median"
    ).reset_index()

    c_out = Path(category_out)
    c_out.parent.mkdir(parents=True, exist_ok=True)
    cat_summary.to_csv(c_out, index=False)

    logger.info(f"Family summary saved to {f_out}; Category summary saved to {c_out}")
    return fam_summary, cat_summary
