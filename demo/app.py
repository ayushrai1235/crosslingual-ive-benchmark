"""
Gradio Interactive Web Application for the Cross-Lingual IVE Benchmark.
Features:
1. Canonical Stimulus & Translation Browser (Side-by-side English, Hindi, Spanish)
2. Precomputed Empirical Results & Figure Viewer (Strictly non-fabricated)
3. Model Registry & Verification Explorer
"""

import json
from pathlib import Path
import pandas as pd
import gradio as gr


def load_scenarios_data():
    scenarios_dir = Path("data/scenarios")
    if not scenarios_dir.exists():
        return {}
    scenarios = {}
    for f in sorted(list(scenarios_dir.glob("*.json"))):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            scenarios[data["scenario_id"]] = data
    return scenarios


def get_scenario_details(scenario_id: str):
    scenarios = load_scenarios_data()
    if scenario_id not in scenarios:
        return "Scenario not found.", "", "", "", ""

    s = scenarios[scenario_id]

    # Meta
    meta = (
        f"**Scenario ID:** {s['scenario_id']}  \n"
        f"**Domain:** {s.get('domain', 'N/A')}  \n"
        f"**Budget:** {s.get('total_budget', 100)} points | **Cost:** {s.get('intervention_cost', 40)} points  \n"
        f"**Human Reviewed:** {' Yes' if s.get('human_reviewed') else ' No'} ({s.get('human_reviewer_notes', '')})"
    )

    # Identifiable texts
    identifiable_text = (
        f"### English (Canonical)\n{s['identifiable_condition']['text']}\n\n"
        f"### Hindi\n{s['translations'].get('hi', {}).get('identifiable_text', 'N/A')}\n\n"
        f"### Spanish\n{s['translations'].get('es', {}).get('identifiable_text', 'N/A')}"
    )

    # Statistical texts
    statistical_text = (
        f"### English (Canonical)\n{s['statistical_condition']['text']}\n\n"
        f"### Hindi\n{s['translations'].get('hi', {}).get('statistical_text', 'N/A')}\n\n"
        f"### Spanish\n{s['translations'].get('es', {}).get('statistical_text', 'N/A')}"
    )

    # Audit & Back-translation details
    audit_notes = (
        f"**Hindi Audit Notes:**\n"
        f"- Back-translation (Identifiable): {s['translations'].get('hi', {}).get('back_translation_identifiable', 'N/A')}\n"
        f"- Semantic Equivalence: {s['translations'].get('hi', {}).get('semantic_equivalence_score', 1.0)}\n\n"
        f"**Spanish Audit Notes:**\n"
        f"- Back-translation (Identifiable): {s['translations'].get('es', {}).get('back_translation_identifiable', 'N/A')}\n"
        f"- Semantic Equivalence: {s['translations'].get('es', {}).get('semantic_equivalence_score', 1.0)}"
    )

    return meta, identifiable_text, statistical_text, audit_notes


def load_table_if_exists(filepath: str) -> pd.DataFrame:
    p = Path(filepath)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame({"Status": ["Results pending - Run python reproduce.py after experimental inference."]})


def load_figure_paths():
    fig_dir = Path("results/figures")
    if not fig_dir.exists():
        return []
    return sorted(list(fig_dir.glob("*.png")))


def build_app():
    scenarios = load_scenarios_data()
    scenario_ids = list(scenarios.keys()) if scenarios else ["No scenarios available"]

    with gr.Blocks(title="Cross-Lingual IVE Benchmark") as demo:
        gr.Markdown(
            "# 🌍 Cross-Lingual Identifiable Victim Effect (IVE) Benchmark\n"
            "### A Cross-Lingual Study of LLM Moral Allocation Bias across English, Hindi, and Spanish"
        )

        with gr.Tabs():
            with gr.TabItem("📖 Stimulus & Translation Browser"):
                gr.Markdown("Explore the standardized humanitarian scenarios across conditions and languages.")
                scenario_dropdown = gr.Dropdown(choices=scenario_ids, value=scenario_ids[0] if scenario_ids else None, label="Select Scenario")

                meta_markdown = gr.Markdown()
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Identifiable Condition (Named Individual)")
                        identifiable_box = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### Statistical Condition (Group/Statistics)")
                        statistical_box = gr.Markdown()

                with gr.Accordion("Linguistic Audit & Back-Translations", open=False):
                    audit_box = gr.Markdown()

                # Wire event
                if scenario_ids and scenario_ids[0] != "No scenarios available":
                    scenario_dropdown.change(
                        fn=get_scenario_details,
                        inputs=[scenario_dropdown],
                        outputs=[meta_markdown, identifiable_box, statistical_box, audit_box]
                    )
                    # Initial load
                    demo.load(
                        fn=get_scenario_details,
                        inputs=[scenario_dropdown],
                        outputs=[meta_markdown, identifiable_box, statistical_box, audit_box]
                    )

            with gr.TabItem("📊 Empirical Results & Tables"):
                gr.Markdown("### Precomputed Statistical Results")
                with gr.Tabs():
                    with gr.TabItem("Model Summaries"):
                        gr.DataFrame(value=load_table_if_exists("results/tables/model_summary.csv"))
                    with gr.TabItem("Cross-Language Contrasts"):
                        gr.DataFrame(value=load_table_if_exists("results/tables/cross_lingual_contrasts.csv"))
                    with gr.TabItem("Bootstrap 95% CIs"):
                        gr.DataFrame(value=load_table_if_exists("results/tables/bootstrap_results.csv"))
                    with gr.TabItem("Primary Hypothesis Tests"):
                        gr.DataFrame(value=load_table_if_exists("results/tables/inferential_tests.csv"))
                    with gr.TabItem("Missingness & Parse Rates"):
                        gr.DataFrame(value=load_table_if_exists("results/tables/missingness_report.csv"))

            with gr.TabItem("📈 Publication Figures"):
                gr.Markdown("### Publication Figures (PNG 300 DPI / PDF)")
                figures = load_figure_paths()
                if figures:
                    for fig_path in figures:
                        gr.Markdown(f"#### {fig_path.stem.replace('_', ' ').title()}")
                        gr.Image(value=str(fig_path))
                else:
                    gr.Markdown("*Figures will appear here after running `python reproduce.py` on empirical judgment data.*")

            with gr.TabItem("🤖 Judge Model Registry"):
                gr.Markdown("### 9 Registered Independent Open-Weight LLM Judge Families")
                gr.DataFrame(value=load_table_if_exists("results/tables/model_verification.csv"))

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
