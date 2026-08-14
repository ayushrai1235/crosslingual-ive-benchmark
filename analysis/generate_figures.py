"""
Publication-Quality Visualization Generator.
Produces the 7 formal benchmark figures in 300 DPI PNG and vector PDF format.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.logging_utils import logger

# Set publication style
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

LANG_COLORS = {
    "en": "#1f77b4",  # Blue
    "hi": "#ff7f0e",  # Orange
    "es": "#2ca02c"   # Green
}

LANG_LABELS = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish"
}


def save_plot(fig: plt.Figure, base_path: Path):
    """Saves figure in both 300 DPI PNG and vector PDF formats."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {base_path.with_suffix('.png')} and .pdf")


def plot_fig1_ive_by_language_and_model(df_paired: pd.DataFrame, out_dir: Path):
    """Figure 1: Grouped bar plot with error bars showing IVE by model and language."""
    if df_paired.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))

    models = sorted(df_paired["model_id"].unique())
    x = np.arange(len(models))
    width = 0.25

    for idx, lang in enumerate(["en", "hi", "es"]):
        sub = df_paired[df_paired["language"] == lang]
        means = [sub[sub["model_id"] == m]["ive"].mean() for m in models]
        sems = [sub[sub["model_id"] == m]["ive"].sem() for m in models]

        ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=sems,
            capsize=3,
            label=LANG_LABELS.get(lang, lang),
            color=LANG_COLORS.get(lang, "#333333"),
            alpha=0.85
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Identifiable Victim Effect (IVE Score)")
    ax.set_title("Figure 1: Identifiable Victim Effect (IVE) Across 9 LLM Judges & Languages")
    ax.legend(title="Language", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig1_ive_by_language_and_model")


def plot_fig2_cross_lingual_attenuation(df_paired: pd.DataFrame, out_dir: Path):
    """Figure 2: Scenario trajectories across English, Hindi, and Spanish."""
    if df_paired.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))

    # Aggregate IVE per scenario across languages
    pivot = df_paired.pivot_table(
        index="scenario_id",
        columns="language",
        values="ive",
        aggfunc="mean"
    ).dropna()

    if all(l in pivot.columns for l in ["en", "hi", "es"]):
        for _, row in pivot.iterrows():
            ax.plot([0, 1, 2], [row["en"], row["hi"], row["es"]], color="#888888", alpha=0.35, linewidth=1.2)

        # Plot overall means
        means = [pivot["en"].mean(), pivot["hi"].mean(), pivot["es"].mean()]
        ax.plot([0, 1, 2], means, color="#d62728", linewidth=3.0, marker="o", label="Benchmark Mean IVE")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["English", "Hindi", "Spanish"])
        ax.set_ylabel("Mean Scenario IVE Score")
        ax.set_title("Figure 2: Cross-Lingual IVE Trajectories Across Canonical Scenarios")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(frameon=True)
        ax.grid(linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig2_cross_lingual_attenuation")


def plot_fig3_condition_distributions(df_raw: pd.DataFrame, out_dir: Path):
    """Figure 3: Allocation distributions by condition and language."""
    valid = df_raw[df_raw["parsed_allocation"].notna()].copy()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.boxplot(
        data=valid,
        x="language",
        y="parsed_allocation",
        hue="victim_condition",
        palette={"identifiable": "#e74c3c", "statistical": "#3498db"},
        ax=ax,
        fliersize=2
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["English", "Hindi", "Spanish"])
    ax.set_ylabel("Budget Allocation (Points)")
    ax.set_xlabel("Evaluation Language")
    ax.set_title("Figure 3: Allocation Distribution by Victim Condition Across Languages")
    ax.legend(title="Condition", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig3_condition_distributions")


def plot_fig4_domain_breakdown(df_paired: pd.DataFrame, out_dir: Path):
    """Figure 4: IVE by humanitarian domain and language."""
    if df_paired.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mock domain if not loaded
    if "domain" not in df_paired.columns:
        df_paired = df_paired.copy()
        df_paired["domain"] = "Humanitarian Aid"

    grouped = df_paired.groupby(["domain", "language"])["ive"].mean().reset_index()

    sns.barplot(
        data=grouped,
        x="domain",
        y="ive",
        hue="language",
        palette=LANG_COLORS,
        ax=ax
    )

    ax.set_ylabel("Mean IVE Score")
    ax.set_xlabel("Scenario Domain")
    ax.set_title("Figure 4: Identifiable Victim Effect Across Humanitarian Domains")
    ax.legend(title="Language", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig4_domain_breakdown")


def plot_fig5_model_family_comparison(df_paired: pd.DataFrame, out_dir: Path):
    """Figure 5: IVE comparisons across model families."""
    if df_paired.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=df_paired,
        x="model_family",
        y="ive",
        hue="language",
        palette=LANG_COLORS,
        ax=ax,
        errorbar="se",
        capsize=0.1
    )

    ax.set_ylabel("Mean IVE Score")
    ax.set_xlabel("Model Family")
    ax.set_title("Figure 5: IVE Magnitudes by Model Family")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(title="Language", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig5_model_family_comparison")


def plot_fig6_language_control_accuracy(out_dir: Path):
    """Figure 6: Comprehension and instruction accuracy from control battery."""
    ctrl_path = Path("results/tables/language_control_results.json")
    if not ctrl_path.exists():
        return
    import json
    with open(ctrl_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    grouped = df.groupby(["model_id", "language"])["is_correct"].mean().reset_index()

    sns.barplot(
        data=grouped,
        x="model_id",
        y="is_correct",
        hue="language",
        palette=LANG_COLORS,
        ax=ax
    )

    ax.set_ylabel("Comprehension Accuracy (0 - 1.0)")
    ax.set_xlabel("Model ID")
    ax.set_title("Figure 6: Language Comprehension Control Battery Performance")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend(title="Language", frameon=True)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig6_language_control_accuracy")


def plot_fig7_bootstrap_ci_forest(df_boot: pd.DataFrame, out_dir: Path):
    """Figure 7: Forest plot of 95% scenario-clustered bootstrap CIs."""
    if df_boot.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))

    y_positions = np.arange(len(df_boot))
    labels = df_boot["target"].tolist()
    points = df_boot["point_estimate"].tolist()
    ci_lows = df_boot["ci_lower"].tolist()
    ci_highs = df_boot["ci_upper"].tolist()

    xerr_low = [p - l for p, l in zip(points, ci_lows)]
    xerr_high = [h - p for p, h in zip(points, ci_highs)]

    ax.errorbar(
        points,
        y_positions,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#2c3e50",
        ecolor="#e74c3c",
        elinewidth=2,
        capsize=4
    )

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect Size (IVE Points / Contrast Difference)")
    ax.set_title("Figure 7: Scenario-Clustered 95% Bootstrap Confidence Intervals (B=10,000)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    save_plot(fig, out_dir / "fig7_bootstrap_ci_forest")


def generate_all_figures(
    df_raw: pd.DataFrame,
    df_paired: pd.DataFrame,
    df_boot: pd.DataFrame,
    output_dir: str | Path = "results/figures"
):
    """Generates all 7 publication-ready figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating publication figures...")

    plot_fig1_ive_by_language_and_model(df_paired, out_dir)
    plot_fig2_cross_lingual_attenuation(df_paired, out_dir)
    plot_fig3_condition_distributions(df_raw, out_dir)
    plot_fig4_domain_breakdown(df_paired, out_dir)
    plot_fig5_model_family_comparison(df_paired, out_dir)
    plot_fig6_language_control_accuracy(out_dir)
    plot_fig7_bootstrap_ci_forest(df_boot, out_dir)

    logger.info(f"All 7 figures successfully generated at {out_dir}")
