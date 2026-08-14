"""
Kaggle GPU Preflight Verification Script for Cross-Lingual IVE Benchmark.
Audits the execution runtime before running empirical LLM evaluations:
1. Python runtime & dependencies (torch, transformers, accelerate, bitsandbytes).
2. NVIDIA CUDA GPU detection & VRAM measurement.
3. Hugging Face authentication & gated model access check.
4. Dataset SHA-256 cryptographic integrity verification.

Usage:
  python experiments/kaggle_preflight.py
"""

import sys
import os
import platform
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_manager import verify_dataset_integrity


def run_preflight() -> bool:
    print("=" * 80)
    print("KAGGLE GPU PREFLIGHT ENVIRONMENT & HARDWARE AUDIT")
    print("=" * 80)

    all_passed = True

    # 1. Python Version
    py_ver = platform.python_version()
    py_major_minor = tuple(map(int, py_ver.split(".")[:2]))
    if py_major_minor >= (3, 10):
        print(f"[PASS] Python Version: {py_ver} (>= 3.10 requirement met)")
    else:
        print(f"[FAIL] Python Version: {py_ver} (Python 3.10+ required)")
        all_passed = False

    # 2. PyTorch & CUDA Check
    try:
        import torch
        print(f"[PASS] PyTorch Installed: v{torch.__version__}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)
            print(f"[PASS] NVIDIA CUDA Available: True (Devices: {device_count})")
            print(f"[PASS] Primary GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
            if vram_gb < 12.0:
                print(f"[WARN] VRAM is {vram_gb:.2f} GB. Models may require 4-bit / 8-bit quantization.")
        else:
            print("[FAIL] NVIDIA CUDA is NOT available! PyTorch cannot detect GPU.")
            print("       -> On Kaggle: Navigate to Notebook Settings -> Accelerator -> select 'GPU T4 x2' or 'GPU P100'.")
            all_passed = False
    except ImportError:
        print("[FAIL] PyTorch is not installed in the environment.")
        all_passed = False

    # 3. Required ML Packages
    ml_packages = [
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("pydantic", "pydantic"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("yaml", "pyyaml")
    ]
    for mod_name, pkg_name in ml_packages:
        try:
            __import__(mod_name)
            print(f"[PASS] Dependency: {pkg_name} installed")
        except ImportError:
            print(f"[FAIL] Missing dependency: {pkg_name}. Run `pip install -q {pkg_name}`.")
            all_passed = False

    # 4. Hugging Face Authentication & Gated Models Access Check
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("[PASS] Hugging Face Token: Detected in environment")
            try:
                user_info = api.whoami(token=hf_token)
                print(f"[PASS] Authenticated Hugging Face User: {user_info.get('name', 'Unknown')}")
            except Exception as e:
                print(f"[WARN] Hugging Face token verification warning: {e}")

            # Check individual judge model access status
            print("\n--- Hugging Face Judge Model Access Status ---")
            models_to_check = [
                ("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", True),
                ("Qwen3-8B", "Qwen/Qwen3-8B", False),
                ("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", False),
                ("Gemma-3-4B-IT", "google/gemma-3-4b-it", True),
                ("Gemma-3-12B-IT", "google/gemma-3-12b-it", True),
                ("Aya-Expanse-8B", "CohereLabs/aya-expanse-8b", False),
                ("Command-R7B", "CohereLabs/c4ai-command-r7b-12-2024", True),
                ("BLOOMZ-7B1-MT", "bigscience/bloomz-7b1-mt", False),
                ("mT0-XL", "bigscience/mt0-xl", False)
            ]
            for m_name, m_repo, is_gated in models_to_check:
                try:
                    api.model_info(m_repo, token=hf_token)
                    print(f"  [PASS] {m_name:<24} ({m_repo}): Access CONFIRMED")
                except Exception as e:
                    err_str = str(e).lower()
                    if "gated" in err_str or "forbidden" in err_str or "403" in err_str or "unauthorized" in err_str or "access" in err_str:
                        print(f"  [PENDING] {m_name:<24} ({m_repo}): Access PENDING approval.")
                        print(f"            -> Do NOT attempt to download/run until approved. Never substitute.")
                    else:
                        print(f"  [WARN] {m_name:<24} ({m_repo}): Could not verify ({e})")
            print("-" * 46 + "\n")
        else:
            print("[WARN] No HF_TOKEN detected in environment.")
            print("       -> Gated models (Llama 3.1, Gemma 3, Command R) require a Hugging Face token.")
            print("       -> Add your token in Kaggle Secrets as 'HF_TOKEN'.")
    except ImportError:
        print("[WARN] huggingface_hub not installed.")

    # 5. Dataset Freeze Manifest Cryptographic Integrity
    manifest_path = Path("data/dataset_manifest.json")
    if manifest_path.exists():
        is_valid, errors = verify_dataset_integrity(manifest_path)
        if is_valid:
            print("[PASS] Dataset Freeze SHA-256 Integrity: PASSED (All 40 stimulus files match manifest)")
        else:
            print(f"[FAIL] Dataset Integrity Check Failed: {errors}")
            all_passed = False
    else:
        print(f"[FAIL] Manifest not found at {manifest_path}")
        all_passed = False

    print("=" * 80)
    if all_passed:
        print("PREFLIGHT STATUS: ALL CHECKS PASSED (Ready for model smoke test)")
        print("=" * 80)
        return True
    else:
        print("PREFLIGHT STATUS: FAILED (Resolve reported errors before proceeding)")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_preflight()
    sys.exit(0 if success else 1)
