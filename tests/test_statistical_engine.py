"""
Unit tests for the statistical analysis engine:
IVE paired computation, cross-lingual contrasts, Holm-Bonferroni correction, and bootstrap CIs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from analysis.compute_ive import compute_paired_ive, summarize_ive_by_language
from analysis.cross_language import compute_cross_language_contrasts
from analysis.bootstrap_ci import scenario_clustered_bootstrap


def create_synthetic_judgments_for_testing():
    """Constructs a small, deterministic test dataframe to verify mathematical correctness."""
    rows = []
    models = ["model_a", "model_b"]
    scenarios = [f"SC_{i:03d}" for i in range(1, 6)]
    languages = ["en", "hi", "es"]

    for m in models:
        for s in scenarios:
            for l in languages:
                # English IVE = +20, Hindi IVE = +10, Spanish IVE = +15
                base_stat = 40.0
                ident_boost = 20.0 if l == "en" else (10.0 if l == "hi" else 15.0)

                # Statistical
                rows.append({
                    "model_id": m,
                    "model_family": "family_1",
                    "category": "general",
                    "scenario_id": s,
                    "language": l,
                    "victim_condition": "statistical",
                    "parsed_allocation": base_stat
                })
                # Identifiable
                rows.append({
                    "model_id": m,
                    "model_family": "family_1",
                    "category": "general",
                    "scenario_id": s,
                    "language": l,
                    "victim_condition": "identifiable",
                    "parsed_allocation": base_stat + ident_boost
                })

    return pd.DataFrame(rows)


def test_paired_ive_calculation():
    raw_df = create_synthetic_judgments_for_testing()
    temp_file = Path("tests/temp_paired.csv")
    try:
        paired_df = compute_paired_ive(raw_df, output_path=temp_file)

        assert len(paired_df) == 2 * 5 * 3  # 2 models * 5 scenarios * 3 languages = 30
        assert "ive" in paired_df.columns

        # Verify IVE means
        en_ive = paired_df[paired_df["language"] == "en"]["ive"].mean()
        hi_ive = paired_df[paired_df["language"] == "hi"]["ive"].mean()
        es_ive = paired_df[paired_df["language"] == "es"]["ive"].mean()

        assert np.isclose(en_ive, 20.0)
        assert np.isclose(hi_ive, 10.0)
        assert np.isclose(es_ive, 15.0)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def test_cross_lingual_contrasts():
    raw_df = create_synthetic_judgments_for_testing()
    temp_paired = Path("tests/temp_paired.csv")
    temp_contrasts = Path("tests/temp_contrasts.csv")
    try:
        paired_df = compute_paired_ive(raw_df, output_path=temp_paired)
        contrasts_df = compute_cross_language_contrasts(paired_df, output_path=temp_contrasts)

        assert not contrasts_df.empty
        hi_vs_en = contrasts_df[contrasts_df["contrast"] == "hi_vs_en"].iloc[0]
        assert np.isclose(hi_vs_en["mean_diff"], -10.0)  # hi (10) - en (20) = -10
        assert "p_holm" in hi_vs_en
    finally:
        if temp_paired.exists():
            temp_paired.unlink()
        if temp_contrasts.exists():
            temp_contrasts.unlink()


def test_scenario_clustered_bootstrap():
    raw_df = create_synthetic_judgments_for_testing()
    temp_paired = Path("tests/temp_paired.csv")
    try:
        paired_df = compute_paired_ive(raw_df, output_path=temp_paired)
        boot_df = scenario_clustered_bootstrap(paired_df, n_resamples=500, seed=42, output_path=None)

        assert not boot_df.empty
        en_boot = boot_df[boot_df["target"] == "en"].iloc[0]
        assert en_boot["ci_lower"] <= 20.0 <= en_boot["ci_upper"]
    finally:
        if temp_paired.exists():
            temp_paired.unlink()
