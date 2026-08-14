"""
Cross-Language Pairwise Contrasts and Multiple Testing Correction.
Evaluates differences in IVE between languages (hi-en, es-en, es-hi)
with Holm-Bonferroni and Benjamini-Hochberg (FDR) adjustments.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from src.logging_utils import logger


def compute_cross_language_contrasts(
    df_paired: pd.DataFrame,
    output_path: str | Path = "results/tables/cross_lingual_contrasts.csv"
) -> pd.DataFrame:
    """
    Computes paired within-scenario differences across languages for each model.
    Matches identical (model_id, scenario_id) across languages.
    """
    if df_paired.empty:
        logger.warning("Empty paired dataframe provided to cross_language.")
        return pd.DataFrame()

    # Pivot across languages
    lang_pivot = df_paired.pivot_table(
        index=["model_id", "model_family", "category", "scenario_id"],
        columns="language",
        values="ive",
        aggfunc="first"
    ).reset_index()

    contrasts_def = [
        ("hi_vs_en", "hi", "en"),
        ("es_vs_en", "es", "en"),
        ("es_vs_hi", "es", "hi")
    ]

    results = []
    raw_pvals = []
    contrast_records = []

    for name, l1, l2 in contrasts_def:
        if l1 not in lang_pivot.columns or l2 not in lang_pivot.columns:
            continue

        valid_subset = lang_pivot.dropna(subset=[l1, l2])
        if len(valid_subset) < 3:
            continue

        diff = valid_subset[l1] - valid_subset[l2]
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)
        median_diff = diff.median()
        iqr_diff = diff.quantile(0.75) - diff.quantile(0.25)

        # Paired Student's t-test
        t_stat, p_param = stats.ttest_rel(valid_subset[l1], valid_subset[l2])

        # Paired Wilcoxon signed-rank test
        try:
            w_stat, p_nonparam = stats.wilcoxon(valid_subset[l1], valid_subset[l2], zero_method="wilcox")
            # Rank biserial correlation
            r_biserial = 1.0 - (2.0 * w_stat) / (len(diff) * (len(diff) + 1) / 2.0)
        except Exception:
            w_stat, p_nonparam, r_biserial = np.nan, np.nan, np.nan

        # Cohen's d_z for paired samples
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        record = {
            "contrast": name,
            "lang1": l1,
            "lang2": l2,
            "n_matched_pairs": len(valid_subset),
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "median_diff": median_diff,
            "iqr_diff": iqr_diff,
            "cohens_d": cohens_d,
            "t_stat": t_stat,
            "p_ttest_raw": p_param,
            "wilcoxon_w": w_stat,
            "p_wilcoxon_raw": p_nonparam,
            "rank_biserial_r": r_biserial
        }
        contrast_records.append(record)
        raw_pvals.append(p_nonparam if not np.isnan(p_nonparam) else p_param)

    if not contrast_records:
        return pd.DataFrame()

    # Apply Holm-Bonferroni and FDR (Benjamini-Hochberg) corrections
    if len(raw_pvals) > 0 and not any(np.isnan(p) for p in raw_pvals):
        # Holm-Bonferroni
        _, p_holm, _, _ = multipletests(raw_pvals, method="holm")
        # Benjamini-Hochberg FDR
        _, p_fdr, _, _ = multipletests(raw_pvals, method="fdr_bh")

        for idx, rec in enumerate(contrast_records):
            rec["p_holm"] = p_holm[idx]
            rec["p_fdr_bh"] = p_fdr[idx]
            rec["significant_alpha_05_holm"] = p_holm[idx] < 0.05
            rec["significant_alpha_05_fdr"] = p_fdr[idx] < 0.05

    res_df = pd.DataFrame(contrast_records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_file, index=False)
    logger.info(f"Cross-language contrasts saved to {out_file}")

    return res_df
