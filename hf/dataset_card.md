---
language:
- en
- hi
- es
license: cc-by-4.0
task_categories:
- text-generation
tags:
- moral-reasoning
- identifiable-victim-effect
- cross-lingual
- evaluation
- humanitarian-decision-making
size_categories:
- n<1K
---

# Cross-Lingual Identifiable Victim Effect (IVE) Benchmark Dataset

## Dataset Summary
This dataset contains the canonical and multilingual scenario stimuli for the research benchmark:
**“Does Language Change the Identifiable Victim Effect? A Cross-Lingual Study of LLM Moral Allocation Bias”**

The benchmark investigates whether Large Language Models exhibit differential moral bias (specifically the Identifiable Victim Effect, IVE) when presented with strictly controlled humanitarian resource allocation scenarios in **English**, **Hindi**, and **Spanish**.

## Dataset Structure
- `scenario_id`: Unique identifier (`SC_001` - `SC_020`).
- `domain`: Humanitarian domain (`medical`, `disaster_relief`, `education`, `food_security`, `clean_water`).
- `total_budget`: Total budget points available (standardized to 100.0).
- `intervention_cost`: Cost required to execute the intervention (standardized to 40.0).
- `victim_count`: Number of statistical victims (standardized to 50).
- `identifiable_condition`: English canonical text featuring a named individual with personal narrative.
- `statistical_condition`: English canonical text featuring quantified group statistics without individual identifiers.
- `translations`: Hindi and Spanish translations verified through back-translation and expert human linguistic review.
- `audit_metadata`: Semantic equivalence, quantitative integrity, and severity calibration scores.

## Data Splits & Formats
- `data/scenarios/*.json`: Individual scenario JSON records.
- `data/dataset_manifest.json`: Immutable release manifest with SHA-256 cryptographic hashes for full tamper-evidence.

## Languages Covered
1. **English (`en`)**: High-resource canonical baseline.
2. **Hindi (`hi`)**: Indo-Aryan language with distinct Devanagari script.
3. **Spanish (`es`)**: High-resource Romance language serving as a grammatical and morphological control.

## Licensing & Citation
This dataset is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

```bibtex
@misc{crosslingual_ive_benchmark_2026,
  title={Does Language Change the Identifiable Victim Effect? A Cross-Lingual Study of LLM Moral Allocation Bias},
  author={Cross-Lingual Moral Reasoning Benchmark Initiative},
  year={2026},
  publisher={Hugging Face Datasets},
  url={https://huggingface.co/datasets/crosslingual-ive-benchmark}
}
```
