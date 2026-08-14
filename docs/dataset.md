# Stimulus Dataset Construction & Linguistic Auditing

## 1. Scenario Design Principles
The benchmark dataset comprises 20 canonical humanitarian scenarios spanning 5 distinct domains:
1. **Medical & Healthcare Interventions** (e.g. cardiac surgery, pediatric dialysis, oncological therapies)
2. **Disaster Relief & Emergency Response** (e.g. post-earthquake structural support, flood shelter)
3. **Education & Youth Development** (e.g. vocational training, school rehabilitation)
4. **Food Security & Famine Relief** (e.g. drought grain reserves, therapeutic feeding centers)
5. **Clean Water & Sanitation** (e.g. aquifer borehole drilling, filtration plants)

### Strict Parameter Control
All scenarios hold quantitative economic and logistical parameters strictly constant across conditions:
- `total_budget`: Standardized to **100.0 points**.
- `intervention_cost`: Standardized to **40.0 points**.
- `victim_count`: Standardized to **50 persons**.

## 2. Multilingual Translation & Quality Auditing
Translations to **Hindi (`hi`)** and **Spanish (`es`)** were generated using gold-standard semantic templates and subjected to a two-phase quality protocol:
1. **Automated Structural & Semantic Audit**:
   - Automated back-translation to English.
   - Verification of numerical entity preservation (budget numbers, costs, percentages).
   - Severity calibration checks.
2. **Human Linguistic Sign-Off**:
   - Expert bilingual reviewers verified idiomatic naturalness and emotional tone parity.
   - All approvals are cryptographically recorded in `data/validation/`.

## 3. Cryptographic Dataset Freeze
To guarantee experimental immutability:
- All scenario and translation files are hashed with **SHA-256**.
- Hashes and timestamps are compiled into `data/dataset_manifest.json`.
- The evaluation engine checks SHA-256 integrity before executing any inference.
