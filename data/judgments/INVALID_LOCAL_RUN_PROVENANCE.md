# Provenance Log: Invalid Local Environment Run Record

**Date**: 2026-08-14 / 2026-08-15  
**Execution Environment**: Local Windows OS (Development / Code Environment)  
**Status**: **INVALID / FAILED ENVIRONMENT RUN (EXCLUDED FROM EMPIRICAL ANALYSIS)**

### Reason for Invalidation
1. Real model inference was initiated on a local development environment without ML execution dependencies (`torch`, `transformers`, `accelerate`, `bitsandbytes`) and without an NVIDIA GPU.
2. The execution resulted in `ImportError: No module named 'torch'` across all prompts with 0.0% parse rate and 0 valid inferences.
3. In accordance with the scientific evaluation protocol, all 0-byte/failed local judgment records have been purged from `data/judgments/`. No local mock or failed records may enter the empirical evaluation directory.

### Canonical Empirical Architecture
- **Development & Codebase**: Local Antigravity Workspace
- **Version Control & Dataset Manifest**: GitHub
- **Canonical GPU Inference Runtime**: Kaggle GPU (NVIDIA T4 16GB / P100 16GB / A100)
- **Model Weights**: Hugging Face Hub
- **Statistical Analysis & Figure Generation**: Offline Python execution (`reproduce.py`) after importing valid empirical judgments from Kaggle.
