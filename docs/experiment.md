# Experimental Protocol & Inference Pipeline

## 1. Multi-Stage Execution Plan
The benchmark follows a strict 3-stage validation process:

1. **Stage 1: Software Verification Pilot**
   - Evaluates 3 scenarios on a test runner.
   - Validates the 3-tier response parser, prompt formatter, memory lifecycle, and JSONL logger.
   - Outputs are strictly isolated from the empirical results directory.
2. **Stage 2: Scientific Pilot**
   - Evaluates 5 scenarios across 3 diverse model families (`llama_3_1_8b`, `qwen_2_5_7b`, `aya_expanse_8b`) across all 3 languages (English, Hindi, Spanish) and both conditions (90 total inferences).
   - Validates live model loading, memory management, and baseline IVE variance.
3. **Stage 3: Full Benchmark**
   - Evaluates all 20 canonical scenarios across all 9 models, 3 languages, and 2 conditions (1,080 total inferences).

## 2. Deterministic Inference & 3-Tier Parsing
- **Hyperparameters**: `temperature = 0.0`, `do_sample = False`, `max_new_tokens = 512`.
- **Chat Templates**: Model-specific native chat templates are applied via `tokenizer.apply_chat_template`.
- **Response Parser**:
  1. *Tier 1*: Strict JSON deserialization (`json.loads`).
  2. *Tier 2*: Markdown fenced JSON extraction (regex ````json ... ````).
  3. *Tier 3*: Labeled field extraction (`"allocation": <num>`).
  - **Ambiguity Guard**: Responses returning numerical ranges (e.g. `40-60`) or multiple conflicting numbers are strictly rejected and recorded as unparsed.
