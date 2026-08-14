"""
One-Command End-to-End Scientific Replication Script.
Usage: python reproduce.py [--include-mock] [--quick-bootstrap]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset_manager import verify_dataset_integrity
from src.reproducibility import set_seed, save_environment_metadata
from src.logging_utils import logger

from analysis.load_results import (
    load_raw_judgments,
    generate_missingness_report,
    generate_model_coverage_from_judgments
)
from analysis.compute_ive import compute_paired_ive, summarize_ive_by_language
from analysis.cross_language import compute_cross_language_contrasts
from analysis.model_comparison import generate_model_summary, generate_family_and_category_summary
from analysis.bootstrap_ci import scenario_clustered_bootstrap
from analysis.statistical_tests import run_condition_x_language_tests, fit_secondary_mixed_effects_model
from analysis.robustness import evaluate_domain_robustness, correlate_with_language_control
from analysis.generate_figures import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description="Reproduce all benchmark tables and figures.")
    parser.add_argument("--include-mock", action="store_true", help="Include test mock runs in analysis (for test pipeline validation only).")
    parser.add_argument("--quick-bootstrap", action="store_true", help="Run 500 bootstrap iterations instead of 10,000 for rapid testing.")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CROSS-LINGUAL IDENTIFIABLE VICTIM EFFECT (IVE) BENCHMARK: REPLICATION PIPELINE")
    print("=" * 80 + "\n")

    # 1. Reproducibility & Integrity Setup
    set_seed(42)
    save_environment_metadata("results/reproducibility_metadata.json")

    manifest_path = Path("data/dataset_manifest.json")
    if manifest_path.exists():
        logger.info("Verifying immutable dataset SHA-256 integrity...")
        is_valid, errors = verify_dataset_integrity(manifest_path)
        if not is_valid:
            logger.error(f"Dataset integrity verification failed: {errors}")
            sys.exit(1)
        logger.info("Dataset SHA-256 integrity check: PASSED.")
    else:
        logger.warning(f"Dataset manifest {manifest_path} not found. Freezing may be required.")

    # 2. Ingest Empirical Judgments
    logger.info("Loading empirical model judgments from data/judgments/...")
    df_raw = load_raw_judgments("data/judgments", include_mock=args.include_mock)

    if df_raw.empty:
        print("\n" + "*" * 80)
        print("EMPIRICAL STATUS: [Results pending]")
        print("No empirical judgment logs found in data/judgments/.")
        print("To collect empirical model judgments, execute:")
        print("  python experiments/run_pilot.py --scientific-pilot")
        print("  python experiments/run_pilot.py --full-benchmark")
        print("*" * 80 + "\n")
        sys.exit(0)

    # 3. Model Coverage & Status Audit
    logger.info("Generating model coverage and execution status audit...")
    cov_df = generate_model_coverage_from_judgments(df_raw, output_path="results/tables/model_coverage.csv")
    generate_missingness_report(df_raw, "results/tables/missingness_report.csv")

    active_models = df_raw["model_id"].unique().tolist()
    pending_models = [m for m in ["llama_3_1_8b"] if m not in active_models]
    print("\n" + "-" * 60)
    print("BENCHMARK PANEL STATUS:")
    print(f"  Active Evaluated Models (N = {len(active_models)}): {', '.join(active_models)}")
    if pending_models:
        print(f"  Pending Access Models  (N = {len(pending_models)}): {', '.join(pending_models)} (Gated / Pending Approval)")
    print("-" * 60)

    # 4. Paired IVE Calculation
    logger.info("Computing paired scenario-level IVE metrics...")
    df_paired = compute_paired_ive(df_raw, "results/processed/paired_ive_data.csv")

    if df_paired.empty:
        logger.warning("No complete (identifiable, statistical) pairs found. Unable to compute IVE.")
        sys.exit(0)

    lang_summary = summarize_ive_by_language(df_paired)
    print("\n" + "-" * 60)
    print("IDENTIFIABLE VICTIM EFFECT (IVE) BY LANGUAGE:")
    print("-" * 60)
    for _, row in lang_summary.iterrows():
        print(f"Language: {row['language'].upper():<4} | N Pairs: {row['n_pairs']:<4} | Mean IVE: {row['mean_ive']:.2f} | Std: {row['std_ive']:.2f} | Median: {row['median_ive']:.2f}")
    print("-" * 60 + "\n")

    # 5. Cross-Language Contrasts & Multiple Testing Adjustments
    logger.info("Computing cross-language contrasts and multiple testing corrections...")
    df_contrasts = compute_cross_language_contrasts(df_paired, "results/tables/cross_lingual_contrasts.csv")

    # 6. Model and Family Comparative Summaries
    logger.info("Generating model, family, and category comparisons...")
    generate_model_summary(df_paired, "results/tables/model_summary.csv")
    generate_family_and_category_summary(df_paired)

    # 7. Scenario-Clustered Bootstrapping (B=10,000)
    n_boot = 500 if args.quick_bootstrap else 10000
    logger.info(f"Running scenario-clustered bootstrap ({n_boot} resamples)...")
    df_boot = scenario_clustered_bootstrap(df_paired, n_resamples=n_boot, seed=42)

    # 8. Primary Inferential Tests & Secondary LMM
    logger.info("Running primary inferential tests and secondary Linear Mixed-Effects Model...")
    run_condition_x_language_tests(df_raw, df_paired, "results/tables/inferential_tests.csv")
    fit_secondary_mixed_effects_model(df_raw, "results/tables/lmm_summary.txt")

    # 9. Domain Robustness & Control Correlation
    logger.info("Evaluating domain robustness...")
    evaluate_domain_robustness(df_paired)
    correlate_with_language_control(df_paired)

    # 10. Generate All 7 Publication Figures
    logger.info("Generating 7 publication figures (PNG 300 DPI + vector PDF)...")
    generate_all_figures(df_raw, df_paired, df_boot, "results/figures")

    print("\n" + "=" * 80)
    print("REPLICATION PIPELINE SUCCESSFULLY EXECUTED")
    print("All processed data, statistical tables, and publication figures have been refreshed.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
