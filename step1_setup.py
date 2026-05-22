"""
Step 1 — Environment Setup
Installs all required libraries for the Sales Data Pipeline project.
Run this once before executing any other steps.
"""

import subprocess
import sys

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "pyspark",
    "pyarrow",       # Parquet support
    "fastparquet",   # Alternative Parquet engine
    "faker",         # Realistic fake data generation
    "joblib",        # Model persistence
]

def install_packages(packages: list[str]) -> None:
    print("=" * 60)
    print("  Sales Data Pipeline — Step 1: Environment Setup")
    print("=" * 60)

    for package in packages:
        print(f"\n📦 Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"   ✅ {package} installed successfully.")
        else:
            print(f"   ❌ Failed to install {package}.")
            print(f"      Error: {result.stderr.strip()}")

    print("\n" + "=" * 60)
    print("  All packages processed. Environment is ready!")
    print("=" * 60)


def verify_imports() -> None:
    """Quick smoke-test to confirm key libraries are importable."""
    print("\n🔍 Verifying imports...")
    checks = {
        "pandas": "pd",
        "numpy": "np",
        "matplotlib": "matplotlib",
        "seaborn": "sns",
        "sklearn": "sklearn",
        "pyspark": "pyspark",
        "faker": "Faker",
    }
    for module, alias in checks.items():
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} — not importable, check installation.")


if __name__ == "__main__":
    install_packages(REQUIRED_PACKAGES)
    verify_imports()
