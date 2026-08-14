# Preregistered 9-Model Judge Panel & Architecture

## 1. Preregistered Model Registry (9 Models across 7 Families)

To prevent monoculture bias and ensure cross-lingual moral evaluation across diverse architectural paradigms, this benchmark preregisters exactly **9 open-weight LLM judge models across 7 distinct families**:

| Family | Model Name | Hugging Face ID | Architecture | Parameters | Quantization | Context | Officially Documented Languages |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama** | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | CausalLM | 8.0B | 4-bit (NF4) | 128k | en, de, fr, it, pt, hi, es, th |
| **Qwen** | Qwen3-8B | `Qwen/Qwen3-8B` | CausalLM | 8.0B | 4-bit (NF4) | 32k | 29+ languages (en, zh, hi, es, ar, ru, fr, de, ...) |
| **Qwen** | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | CausalLM | 7.6B | 4-bit (NF4) | 128k | 29+ languages (en, zh, hi, es, ar, ru, fr, de, ...) |
| **Gemma** | Gemma-3-4B-IT | `google/gemma-3-4b-it` | CausalLM | 4.0B | 4-bit (NF4) | 128k | 140+ languages (en, hi, es, fr, ar, bn, ...) |
| **Gemma** | Gemma-3-12B-IT | `google/gemma-3-12b-it` | CausalLM | 12.0B | 4-bit (NF4) | 128k | 140+ languages (en, hi, es, fr, ar, bn, ...) |
| **Aya** | Aya-Expanse-8B | `CohereLabs/aya-expanse-8b` | CausalLM | 8.0B | 4-bit (NF4) | 8k | 23 languages (en, hi, es, ar, zh, fr, de, ...) |
| **Command** | Command-R7B | `CohereLabs/c4ai-command-r7b-12-2024` | CausalLM | 7.0B | 4-bit (NF4) | 128k | 23 languages (en, hi, es, ar, zh, fr, de, ...) |
| **BLOOMZ** | BLOOMZ-7B1-MT | `bigscience/bloomz-7b1-mt` | CausalLM | 7.1B | 4-bit (NF4) | 2k | 46 natural languages (en, hi, es, fr, zh, ar, ...) |
| **mT0** | mT0-XL | `bigscience/mt0-xl` | Seq2Seq | 3.7B | 4-bit (NF4) | 2k | 101 languages (en, hi, es, zh, fr, ar, ...) |

---

## 2. Experimental Languages vs. Documented Languages

To uphold research transparency, the benchmark explicitly separates:
1. **Officially Documented Languages (`officially_documented_languages`)**: The full set of natural languages officially supported, evaluated, or documented by the model creators.
2. **Experiment Languages (`experiment_languages`)**: The subset evaluated in this benchmark:
   - **English (`en`)**: High-resource canonical baseline.
   - **Hindi (`hi`)**: Indo-Aryan language with distinct Devanagari script.
   - **Spanish (`es`)**: Romance language serving as a morphological control.

All 9 models have documented native support for English, Hindi, and Spanish.

---

## 3. Dual Runner Architecture

The evaluation pipeline (`src/model_runner.py`) uses a polymorphic runner architecture:

### 3.1 CausalLM Runner (`CausalLMRunner`)
- Utilizes `AutoModelForCausalLM` and `AutoTokenizer`.
- Applies standardized chat templates or fallback instruction formatting.
- **Qwen3 Reasoning Configuration**: Explicitly disables thinking/reasoning traces for primary moral allocation evaluation (`disable_thinking: true`) to maintain direct zero-shot comparability across causal models.

### 3.2 Seq2Seq Runner (`Seq2SeqRunner`)
- Utilizes `AutoModelForSeq2SeqLM` for encoder-decoder models (`bigscience/mt0-xl`).
- Feeds full prompt into the encoder and decodes directly without autoregressive chat templates.

---

## 4. Quantization & GPU Memory Optimization

- **Precision**: 4-bit NormalFloat (NF4) with double quantization and `bfloat16` compute dtype via `bitsandbytes`.
- **Sequential Execution**: Models are executed one at a time. After evaluation, Python garbage collection and `torch.cuda.empty_cache()` are invoked to prevent VRAM accumulation on 16GB GPUs (e.g. NVIDIA T4 / P100).
- **Batch Size**: `batch_size = 1` for deterministic single-stream judgment generation at `temperature = 0.0`.
