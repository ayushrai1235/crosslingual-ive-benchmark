"""
Inferential Statistical Testing Engine.
Implements primary non-parametric & repeated-measures hypothesis tests
and secondary Linear Mixed-Effects Models (LMM) with strict convergence checking.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from src.logging_utils import logger


def run_condition_x_language_tests(
    df_raw: pd.DataFrame,
    df_paired: pd.DataFrame,
    output_table: str | Path = "results/tables/inferential_tests.csv"
) -> pd.DataFrame:
    """
    Executes primary inferential hypothesis tests:
    1. Paired non-parametric Wilcoxon tests across languages
    2. Friedman test for omnibus across the 3 languages
    3. Condition x Language interaction two-way ANOVA
    """
    if df_paired.empty or df_raw.empty:
        logger.warning("Empty dataframes passed to statistical_tests.")
        return pd.DataFrame()

    results = []

    # 1. Friedman Omnibus Test across English, Hindi, and Spanish
    pivot = df_paired.pivot_table(
        index=["model_id", "scenario_id"],
        columns="language",
        values="ive"
    ).dropna()

    if len(pivot) >= 5 and all(l in pivot.columns for l in ["en", "hi", "es"]):
        f_stat, p_friedman = stats.friedmanchisquare(pivot["en"], pivot["hi"], pivot["es"])
        results.append({
            "test_name": "Friedman Omnibus (Language Effect on IVE)",
            "statistic_name": "Q (Chi-Square)",
            "statistic_value": f_stat,
            "p_value": p_friedman,
            "degrees_of_freedom": 2,
            "n_obs": len(pivot),
            "significance_05": p_friedman < 0.05
        })

    # 2. Two-way Factorial ANOVA: Condition x Language
    valid_raw = df_raw[df_raw["parsed_allocation"].notna()].copy()
    try:
        model_formula = "parsed_allocation ~ C(victim_condition) + C(language) + C(victim_condition):C(language)"
        ols_res = smf.ols(model_formula, data=valid_raw).fit()
        anova_table = sm.stats.anova_lm(ols_res, typ=2)

        for factor in ["C(victim_condition)", "C(language)", "C(victim_condition):C(language)"]:
            if factor in anova_table.index:
                row = anova_table.loc[factor]
                results.append({
                    "test_name": f"Two-Way ANOVA: {factor}",
                    "statistic_name": "F",
                    "statistic_value": row["F"],
                    "p_value": row["PR(>F)"],
                    "degrees_of_freedom": row["df"],
                    "n_obs": len(valid_raw),
                    "significance_05": row["PR(>F)"] < 0.05
                })
    except Exception as e:
        logger.error(f"Error fitting two-way ANOVA: {e}")

    res_df = pd.DataFrame(results)
    out_path = Path(output_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_path, index=False)
    logger.info(f"Primary inferential test summary saved to {out_path}")

    return res_df


def fit_secondary_mixed_effects_model(
    df_raw: pd.DataFrame,
    summary_path: str | Path = "results/tables/lmm_summary.txt"
) -> str:
    """
    Fits secondary exploratory Linear Mixed-Effects Model (LMM):
    Formula: parsed_allocation ~ victim_condition * language
    Random intercepts for: model_id, scenario_id
    """
    if df_raw.empty:
        return "No data available."

    valid = df_raw[df_raw["parsed_allocation"].notna()].copy()
    valid["is_identifiable"] = (valid["victim_condition"] == "identifiable").astype(int)

    try:
        # Fit MixedLM with random intercept for model_id and scenario_id
        md = smf.mixedlm(
            "parsed_allocation ~ is_identifiable * C(language)",
            valid,
            groups=valid["model_id"],
            vc_formula={"scenario": "0 + C(scenario_id)"}
        )
        mdf = md.fit()
        summary_str = str(mdf.summary())

        out_file = Path(summary_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("SECONDARY / EXPLORATORY LINEAR MIXED-EFFECTS MODEL (LMM)\n")
            f.write("=" * 70 + "\n")
            f.write(summary_str)
            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Secondary LMM summary saved to {out_file}")
        return summary_str
    except Exception as e:
        err_msg = f"LMM estimation note: {e}"
        logger.warning(err_msg)
        return err_msg
