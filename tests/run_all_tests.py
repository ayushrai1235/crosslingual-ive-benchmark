"""
Standalone Test Runner for Cross-Lingual IVE Benchmark.
Executes all unit and integration test suites without requiring external pytest installation.
"""

import sys
import inspect
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

test_files = [
    "test_schemas",
    "test_parser",
    "test_dataset_manager",
    "test_validators",
    "test_statistical_engine",
    "test_model_registry_and_runners",
    "test_pipeline_integration"
]

def main():
    print("=" * 80)
    print("RUNNING CROSS-LINGUAL IVE BENCHMARK TEST SUITE")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for mod_name in test_files:
        print(f"\n--- Testing module: tests.{mod_name} ---")
        try:
            mod = __import__(f"tests.{mod_name}", fromlist=["*"])
        except Exception as e:
            print(f"[ERROR] Failed to import tests.{mod_name}: {e}")
            traceback.print_exc()
            failed_tests.append((f"tests.{mod_name}.__import__", str(e)))
            continue

        for attr_name, attr_val in inspect.getmembers(mod):
            if attr_name.startswith("test_") and callable(attr_val):
                total_tests += 1
                sig = inspect.signature(attr_val)
                try:
                    kwargs = {}
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        path_tmp = Path(tmp_dir)
                        if "tmp_path" in sig.parameters:
                            kwargs["tmp_path"] = path_tmp
                        if "temp_dataset" in sig.parameters and hasattr(mod, "temp_dataset"):
                            fixture_func = getattr(mod.temp_dataset, "__wrapped__", mod.temp_dataset)
                            kwargs["temp_dataset"] = fixture_func(path_tmp)
                        
                        attr_val(**kwargs)
                    print(f"  [PASS] {attr_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  [FAIL] {attr_name}: {e}")
                    traceback.print_exc()
                    failed_tests.append((f"{mod_name}.{attr_name}", str(e)))

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} passed ({passed_tests/max(total_tests, 1)*100:.1f}%)")
    if failed_tests:
        print(f"FAILED TESTS ({len(failed_tests)}):")
        for name, err in failed_tests:
            print(f"  - {name}: {err}")
        print("=" * 80 + "\n")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED SUCCESSFULLY.")
        print("=" * 80 + "\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
