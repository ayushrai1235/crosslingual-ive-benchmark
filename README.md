# Does Language Change the Identifiable Victim Effect?
### A Cross-Lingual Study of LLM Moral Allocation Bias

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Reproducibility: 100% Deterministic](https://img.shields.io/badge/Reproducibility-Deterministic-success.svg)](docs/reproducibility.md)

---

## 🌟 Overview

The **Cross-Lingual Identifiable Victim Effect (IVE) Benchmark** is an open-source empirical evaluation framework designed to investigate whether Large Language Models exhibit moral allocation bias when presented with humanitarian resource distribution dilemmas across different languages.

The core experimental comparison investigates:
$$\textbf{IDENTIFIABLE VICTIM} \quad \text{vs.} \quad \textbf{STATISTICAL VICTIMS}$$

Across three distinct languages:
- **English (`en`)**: High-resource canonical baseline.
- **Hindi (`hi`)**: Indo-Aryan language with distinct Devanagari script.
- **Spanish (`es`)**: Romance language serving as a morphological control.

Evaluated by **9 preregistered open-weight LLM judge models across 7 families**:
1. **Llama-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`) — *Llama family*
2. **Qwen3-8B** (`Qwen/Qwen3-8B`) — *Qwen family*
3. **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`) — *Qwen family*
4. **Gemma-3-4B-IT** (`google/gemma-3-4b-it`) — *Gemma family*
5. **Gemma-3-12B-IT** (`google/gemma-3-12b-it`) — *Gemma family*
6. **Aya-Expanse-8B** (`CohereLabs/aya-expanse-8b`) — *Aya family*
7. **Command-R7B** (`CohereLabs/c4ai-command-r7b-12-2024`) — *Command family*
8. **BLOOMZ-7B1-MT** (`bigscience/bloomz-7b1-mt`) — *BLOOMZ family*
9. **mT0-XL** (`bigscience/mt0-xl`) — *mT0 family (Encoder-Decoder Seq2Seq)*

---

## 🔬 Core Scientific Principles

1. **Non-Fabrication Commitment**: All numerical findings in this benchmark are computed solely from actual model outputs. If experiments have not yet been run, all scripts and dashboards display `[Results pending]` rather than generating synthetic metrics.
2. **Cryptographic Immutability**: All scenario stimuli are frozen under SHA-256 integrity hashes in `data/dataset_manifest.json`.
3. **Human-in-the-Loop Validation**: Scenario generation and translations undergo automated structural auditing followed by expert human sign-off.
4. **Statistical Rigor**: Primary inferential focus is on the **Condition $\times$ Language** interaction with paired scenario differences, Holm-Bonferroni/FDR multiple testing corrections, and 10,000 scenario-clustered bootstrap confidence intervals.

---

## 📂 Repository Structure

```
├── config/                  # Experiment, model registry, and language configurations
├── data/
│   ├── scenarios/           # 20 canonical English scenario JSON files
│   ├── translations/        # Verified Hindi and Spanish translations
│   ├── validation/          # Audit results and human review records
│   ├── language_control/    # Language comprehension and instruction battery
│   └── dataset_manifest.json# SHA-256 frozen dataset integrity manifest
├── prompts/                 # Standardized prompt templates for generation & judging
├── src/                     # Core benchmark library (schemas, parser, runners)
├── experiments/             # CLI pipeline scripts (pilot, full run, audits)
├── analysis/                # Statistical testing, bootstrapping, and figure engine
├── results/
│   ├── tables/              # Statistical test summaries, missingness, bootstrap CIs
│   ├── figures/             # 7 publication-ready figures (PNG 300 DPI + PDF)
│   └── processed/           # Paired scenario IVE datasets
├── kaggle/                  # Standalone notebook & requirements for free GPU execution
├── demo/                    # Interactive Gradio dashboard
├── hf/                      # Hugging Face Dataset card and upload utility
├── docs/                    # In-depth methodology and architectural documentation
└── tests/                   # Complete pytest suite
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/your-org/crosslingual-ive-benchmark.git
cd crosslingual-ive-benchmark
pip install -r requirements.txt
```

### 2. Verify Software & Pipeline Tests
```bash
pytest tests/ -v
```

### 3. Run Stage 1 Software Pilot
```bash
python experiments/run_software_pilot.py
```

### 4. Run Stage 2 Scientific Pilot (3 Models)
```bash
python experiments/run_pilot.py --scientific-pilot
```

### 5. Reproduce All Tables & Publication Figures
```bash
python reproduce.py
```

### 6. Launch Interactive Gradio Dashboard
```bash
python demo/app.py
```

---

## 📊 Publication Figures Generated

The benchmark automatically renders 7 publication-quality figures (300 DPI PNG and vector PDF) upon running `python reproduce.py`:
- **Figure 1**: Grouped bar plot of IVE effect sizes across 9 models and 3 languages.
- **Figure 2**: Cross-lingual IVE trajectories across individual canonical scenarios.
- **Figure 3**: Allocation distributions by victim condition across languages.
- **Figure 4**: IVE variation across 5 humanitarian aid domains.
- **Figure 5**: Architectural category comparisons (General vs Multilingual vs Reasoning).
- **Figure 6**: Language comprehension control battery performance.
- **Figure 7**: Forest plot of 95% scenario-clustered bootstrap confidence intervals ($B=10,000$).

---

## 📖 Citation

If you use this benchmark, code, or stimulus dataset in your research, please cite:

```bibtex
@misc{crosslingual_ive_benchmark_2026,
  title={Does Language Change the Identifiable Victim Effect? A Cross-Lingual Study of LLM Moral Allocation Bias},
  author={Cross-Lingual Moral Reasoning Benchmark Initiative},
  year={2026},
  url={https://github.com/your-org/crosslingual-ive-benchmark}
}
```

## 📜 License
- **Code**: [MIT License](LICENSE)
- **Dataset**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
