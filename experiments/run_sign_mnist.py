from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_ga.benchmarks import main as benchmarks_main

# Ensure default problem is 'sign_mnist' when not specified by user
if __name__ == "__main__":
    argv = sys.argv
    if not any(a.startswith("--problem") for a in argv[1:]):
        sys.argv.insert(1, "--problem")
        sys.argv.insert(2, "sign_mnist")
    benchmarks_main()
