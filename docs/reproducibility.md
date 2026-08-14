# End-to-End Reproducibility Guide

## 1. Hardware & Environment Specifications
- **Operating Systems**: Linux (Ubuntu 22.04 LTS recommended) / Windows 11 / macOS (Apple Silicon).
- **GPU Accelerator**: NVIDIA T4 (16GB), RTX 3090/4090 (24GB), or A100 (40/80GB).
- **CUDA Version**: CUDA 12.1+ / PyTorch 2.2+.
- **Python Version**: Python 3.10, 3.11, or 3.12.

## 2. One-Command Setup & Verification
Clone the repository and install all dependencies:
```bash
git clone https://github.com/your-org/crosslingual-ive-benchmark.git
cd crosslingual-ive-benchmark
pip install -r requirements.txt
```

Verify test suite and pipeline integrity:
```bash
pytest tests/ -v
```

## 3. Step-by-Step Benchmark Execution Workflow

### Step 3.1: Dataset Integrity Verification
```bash
python experiments/freeze_dataset.py
```

### Step 3.2: Run Objective Language Comprehension Control Battery
```bash
python experiments/run_language_control.py
```

### Step 3.3: Execute Stage 2 Scientific Pilot (3 Models)
```bash
python experiments/run_pilot.py --scientific-pilot
```

### Step 3.4: Execute Stage 3 Full Benchmark (9 Models)
```bash
python experiments/run_pilot.py --full-benchmark
```

### Step 3.5: Run Analysis & Generate Publication Figures
```bash
python reproduce.py
```

## 4. Precomputed Artifacts Location
- Empirical Tables: `results/tables/`
- Publication Figures (PNG/PDF): `results/figures/`
- Processed Datasets: `results/processed/`
- Reproducibility Metadata: `results/reproducibility_metadata.json`
