"""
Scenario-Clustered Bootstrap Confidence Interval Estimator.
Performs 10,000 resamples at the scenario level to maintain within-scenario correlation structure.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from src.logging_utils import logger


def scenario_clustered_bootstrap(
    df_paired: pd.DataFrame,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> pd.DataFrame:
    """
    Performs scenario-clustered bootstrap resampling.
    Resamples unique scenario_ids with replacement, evaluating mean IVE and cross-lingual contrasts.
    """
    if df_paired.empty:
        logger.warning("Empty dataframe provided to bootstrap estimator.")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    unique_scenarios = df_paired["scenario_id"].unique()
    n_scenarios = len(unique_scenarios)

    if n_scenarios == 0:
        return pd.DataFrame()

    languages = ["en", "hi", "es"]
    available_langs = [l for l in languages if l in df_paired["language"].unique()]

    # Container for bootstrap replications
    # Mean IVE per language
    boot_means: Dict[str, List[float]] = {l: [] for l in available_langs}
    # Pairwise contrast differences
    boot_diffs: Dict[str, List[float]] = {
        "hi_minus_en": [],
        "es_minus_en": [],
        "es_minus_hi": []
    }

    # Group dataframe by scenario_id for fast lookup
    scenario_groups = {s_id: group for s_id, group in df_paired.groupby("scenario_id")}

    for _ in range(n_resamples):
        # Resample scenario IDs with replacement
        sampled_scenario_ids = rng.choice(unique_scenarios, size=n_scenarios, replace=True)
        # Assemble resampled dataset
        sample_df = pd.concat([scenario_groups[s_id] for s_id in sampled_scenario_ids], ignore_index=True)

        # Compute mean IVE per language
        lang_means = sample_df.groupby("language")["ive"].mean().to_dict()
        for l in available_langs:
            if l in lang_means:
                boot_means[l].append(lang_means[l])

        # Compute pairwise contrasts
        if "hi" in lang_means and "en" in lang_means:
            boot_diffs["hi_minus_en"].append(lang_means["hi"] - lang_means["en"])
        if "es" in lang_means and "en" in lang_means:
            boot_diffs["es_minus_en"].append(lang_means["es"] - lang_means["en"])
        if "es" in lang_means and "hi" in lang_means:
            boot_diffs["es_minus_hi"].append(lang_means["es"] - lang_means["hi"])

    # Calculate Empirical Estimates and Percentile CIs
    alpha = (1.0 - ci_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    results = []

    # Process language means
    for l in available_langs:
        arr = np.array(boot_means[l])
        if len(arr) > 0:
            orig_mean = df_paired[df_paired["language"] == l]["ive"].mean()
            ci_low = np.percentile(arr, lower_pct)
            ci_high = np.percentile(arr, upper_pct)
            results.append({
                "metric_type": "mean_ive",
                "target": l,
                "point_estimate": orig_mean,
                "boot_mean": np.mean(arr),
                "boot_se": np.std(arr, ddof=1),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_resamples": n_resamples
            })

    # Process contrasts
    for contrast_name, diffs in boot_diffs.items():
        if len(diffs) > 0:
            arr = np.array(diffs)
            # Original difference
            l1, l2 = contrast_name.split("_minus_")
            m1 = df_paired[df_paired["language"] == l1]["ive"].mean() if l1 in df_paired["language"].values else np.nan
            m2 = df_paired[df_paired["language"] == l2]["ive"].mean() if l2 in df_paired["language"].values else np.nan
            orig_diff = m1 - m2

            ci_low = np.percentile(arr, lower_pct)
            ci_high = np.percentile(arr, upper_pct)
            results.append({
                "metric_type": "pairwise_contrast",
                "target": contrast_name,
                "point_estimate": orig_diff,
                "boot_mean": np.mean(arr),
                "boot_se": np.std(arr, ddof=1),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_resamples": n_resamples
            })

    res_df = pd.DataFrame(results)
    out_file = Path("results/tables/bootstrap_results.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_file, index=False)
    logger.info(f"Scenario-clustered bootstrap analysis completed ({n_resamples} iterations). Saved to {out_file}")

    return res_df
