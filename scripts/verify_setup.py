"""Verify that all required dependencies are installed and importable.

Run with:
    python scripts/verify_setup.py
"""

import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("librosa", "librosa"),
    ("ripser", "ripser"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("pydantic", "pydantic"),
]

OPTIONAL = [
    ("parselmouth", "praat-parselmouth", "jitter/shimmer/formant features"),
    ("gudhi", "gudhi", "cubical complex PH"),
    ("gtda", "giotto-tda", "persistence images/landscapes"),
    ("persim", "persim", "persistence images (lightweight)"),
    ("seaborn", "seaborn", "enhanced visualization"),
]


def check(module: str, package: str, label: str = "") -> bool:
    try:
        __import__(module)
        print(f"  [OK]  {package}{' — ' + label if label else ''}")
        return True
    except ImportError:
        tag = "MISSING" if not label else "OPTIONAL"
        print(f"  [{tag}]  {package}{' — ' + label if label else ''}")
        return False


def main() -> None:
    print("\n=== TDA Audio Deepfake Detection — Setup Verification ===\n")

    print("Required dependencies:")
    missing = [pkg for mod, pkg in REQUIRED if not check(mod, pkg)]

    print("\nOptional dependencies:")
    for mod, pkg, label in OPTIONAL:
        check(mod, pkg, label)

    print()
    if missing:
        print(f"ERROR: {len(missing)} required package(s) missing: {', '.join(missing)}")
        print("Install with:  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("All required dependencies are installed.")
        print("Install optional packages as needed for extended features.")


if __name__ == "__main__":
    main()
