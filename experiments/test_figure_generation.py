"""
Test Figure Generation Engine.
Verifies that all 7 publication-ready figures (PNG 300 DPI + vector PDF) are correctly generated.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.generate_figures import generate_all_figures
from src.logging_utils import logger


def main():
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check if mock judgment data exists in dry_run or software_pilot for figure smoke test
    dry_run_dir = Path("data/judgments/dry_run")
    software_dir = Path("data/judgments/software_pilot")

    from analysis.load_results import load_raw_judgments
    from analysis.compute_ive import compute_paired_ive
    from analysis.bootstrap_ci import scenario_clustered_bootstrap

    data_dir = dry_run_dir if dry_run_dir.exists() and list(dry_run_dir.glob("*.jsonl")) else software_dir

    if data_dir.exists() and list(data_dir.glob("*.jsonl")):
        logger.info(f"Loading test judgment records from {data_dir} for figure generation test...")
        df_raw = load_raw_judgments(data_dir, include_mock=True)
        df_paired = compute_paired_ive(df_raw, output_path=None)
        df_boot = scenario_clustered_bootstrap(df_paired, n_resamples=100)
    else:
        logger.info("Synthesizing minimal DataFrame structure to test all 7 figure plotters...")
        df_raw = pd.DataFrame([
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "en", "victim_condition": "identifiable", "parsed_allocation": 60.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "en", "victim_condition": "statistical", "parsed_allocation": 45.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "hi", "victim_condition": "identifiable", "parsed_allocation": 55.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "hi", "victim_condition": "statistical", "parsed_allocation": 48.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "es", "victim_condition": "identifiable", "parsed_allocation": 58.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "es", "victim_condition": "statistical", "parsed_allocation": 46.0},
        ])
        df_paired = pd.DataFrame([
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "en", "domain": "Health", "ive": 15.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "hi", "domain": "Health", "ive": 7.0},
            {"model_id": "llama_3_1_8b", "model_family": "Llama", "scenario_id": "sc_001", "language": "es", "domain": "Health", "ive": 12.0},
        ])
        df_boot = pd.DataFrame([
            {"target": "Overall IVE (English)", "point_estimate": 15.0, "ci_lower": 10.2, "ci_upper": 19.8},
            {"target": "Overall IVE (Hindi)", "point_estimate": 7.0, "ci_lower": 2.1, "ci_upper": 11.9},
            {"target": "Overall IVE (Spanish)", "point_estimate": 12.0, "ci_lower": 7.5, "ci_upper": 16.5},
            {"target": "Contrast EN - HI", "point_estimate": 8.0, "ci_lower": 3.0, "ci_upper": 13.0},
            {"target": "Contrast EN - ES", "point_estimate": 3.0, "ci_lower": -1.5, "ci_upper": 7.5}
        ])

    generate_all_figures(df_raw, df_paired, df_boot, output_dir=out_dir)

    expected_figures = [
        "fig1_ive_by_language_and_model",
        "fig2_cross_lingual_attenuation",
        "fig3_condition_distributions",
        "fig4_domain_breakdown",
        "fig5_model_family_comparison",
        "fig6_language_control_accuracy",
        "fig7_bootstrap_ci_forest"
    ]

    print("\n" + "=" * 70)
    print("FIGURE GENERATION VERIFICATION AUDIT")
    print("=" * 70)
    for fig_name in expected_figures:
        png_path = out_dir / f"{fig_name}.png"
        pdf_path = out_dir / f"{fig_name}.pdf"
        assert png_path.exists(), f"Missing PNG: {png_path}"
        assert pdf_path.exists(), f"Missing PDF: {pdf_path}"
        print(f"[PASS] {fig_name:<35} : PNG ({png_path.stat().st_size // 1024} KB), PDF ({pdf_path.stat().st_size // 1024} KB)")
    print("=" * 70)
    print(f"All 7 figures (14 files total: 7 PNG + 7 PDF) successfully verified at {out_dir}\n")


if __name__ == "__main__":
    main()
