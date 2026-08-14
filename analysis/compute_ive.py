"""
Paired Identifiable Victim Effect (IVE) Computation module.
Calculates paired scenario-level difference: IVE = Identifiable - Statistical.
Enforces non-imputation: pairs are dropped if either condition failed to parse.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from src.logging_utils import logger


def compute_paired_ive(
    df_raw: pd.DataFrame,
    output_path: str | Path = "results/processed/paired_ive_data.csv"
) -> pd.DataFrame:
    """
    Constructs the paired experimental dataset.
    Formula: IVE_{m,s,l} = Allocation(Identifiable) - Allocation(Statistical)
    """
    if df_raw.empty:
        logger.warning("Empty raw judgments DataFrame passed to compute_paired_ive.")
        return pd.DataFrame()

    # Filter to successful judgments with numeric allocations
    valid = df_raw[df_raw["parsed_allocation"].notna()].copy()

    # Pivot to get identifiable and statistical allocations side-by-side
    pivot = valid.pivot_table(
        index=["model_id", "model_family", "category", "scenario_id", "language"],
        columns="victim_condition",
        values="parsed_allocation",
        aggfunc="first"
    ).reset_index()

    # Ensure both columns exist
    if "identifiable" not in pivot.columns or "statistical" not in pivot.columns:
        logger.warning("Missing one of ['identifiable', 'statistical'] condition columns in pivot table.")
        return pd.DataFrame()

    # Drop incomplete pairs (strict no-imputation rule)
    complete_pairs = pivot.dropna(subset=["identifiable", "statistical"]).copy()
    dropped_count = len(pivot) - len(complete_pairs)
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} incomplete scenario-language pairs due to missing condition output.")

    # Compute paired IVE metric
    complete_pairs["ive"] = complete_pairs["identifiable"] - complete_pairs["statistical"]

    # Save to disk if output_path is provided
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        complete_pairs.to_csv(out_file, index=False)
        logger.info(f"Computed {len(complete_pairs)} paired IVE values. Saved to {out_file}")

    return complete_pairs


def summarize_ive_by_language(df_paired: pd.DataFrame) -> pd.DataFrame:
    """Calculates overall IVE summary statistics per language."""
    if df_paired.empty:
        return pd.DataFrame()

    summary = df_paired.groupby("language")["ive"].agg(
        n_pairs="count",
        mean_ive="mean",
        std_ive="std",
        median_ive="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        iqr=lambda s: s.quantile(0.75) - s.quantile(0.25)
    ).reset_index()

    return summary
