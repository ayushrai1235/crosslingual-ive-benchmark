# Kaggle Experiment Guide: Cross-Lingual IVE Benchmark

This directory contains the standalone Kaggle notebook and configuration to execute the **Cross-Lingual Identifiable Victim Effect (IVE) Benchmark** on free GPU tiers (NVIDIA T4 x2 or P100).

## Experimental Sizes (Frozen Dataset: 20 Canonical Scenarios)
- **Smoke Test**: 1 scenario × 1 language (`en`) × 2 conditions × 1 model = **2 judgments** (Mandatory environment check)
- **Software Pilot**: 3 scenarios × 3 languages × 2 conditions × 1 model = **18 mock judgments**
- **Scientific Pilot**: 10 scenarios × 3 languages × 2 conditions × 9 models = **540 judgments**
- **Full Benchmark**: 20 scenarios × 3 languages × 2 conditions × 9 models = **1,080 judgments**

## Kaggle Execution Workflow

1. **Create a New Notebook on Kaggle**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code) and create a new notebook.
   - Set Accelerator to **GPU T4 x2** or **GPU P100**.
   - Enable **Internet Access** in Notebook Settings.

2. **Add Hugging Face Token (Required for Gated Models)**:
   - Go to **Add-ons -> Secrets** in the Kaggle notebook.
   - Add a secret named `HF_TOKEN` with your Hugging Face read token (required to access `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-*`).

3. **Install Dependencies**:
   ```bash
   !pip install -q torch transformers accelerate bitsandbytes pydantic scipy statsmodels pandas numpy matplotlib seaborn pyyaml huggingface_hub
   ```

4. **Run Preflight Hardware & Environment Check**:
   ```bash
   !python experiments/kaggle_preflight.py
   ```
   *Verifies Python >= 3.10, PyTorch, CUDA GPU detection, VRAM, Hugging Face login, and SHA-256 dataset integrity.*

5. **Execute Mandatory Stage 0 Smoke Test**:
   ```bash
   !python experiments/run_pilot.py --smoke-test
   ```
   *Executes 2 real model inferences (Identifiable & Statistical) and verifies 2 valid numeric parsed allocations before launching the broader benchmark.*

6. **Run Scientific Pilot or Full Benchmark**:
   - Scientific Pilot (540 judgments):
     ```bash
     !python experiments/run_pilot.py --scientific-pilot
     ```
   - Full Benchmark (1,080 judgments):
     ```bash
     !python experiments/run_pilot.py --full-benchmark
     ```
   - Or single model (e.g. Qwen 2.5 7B):
     ```bash
     !python experiments/run_pilot.py --model qwen_2_5_7b
     ```

7. **Generate Publication Tables & Figures**:
   ```bash
   !python reproduce.py
   ```
   *Generates all 7 publication figures in `results/figures/` and 6 statistical tables in `results/tables/`.*

## Hardware & Memory Architecture
- **Quantization**: 4-bit / 8-bit quantization via `bitsandbytes` allows each 7B–12B model to fit comfortably in **~6–8 GB VRAM**.
- **Lifecycle**: Sequential execution loads one model at a time, executes all trials, and aggressively frees VRAM with `torch.cuda.empty_cache()` and `gc.collect()`.
