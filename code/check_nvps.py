"""
Quick audit: What NVPS-related assessment names exist in clif_patient_assessments,
and which ones have actual values?
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_clif_table

print("Loading patient_assessments...")
pa = load_clif_table("patient_assessments")

# Find all assessment names containing "NVPS" or "NONVERBAL" or "PAIN SCALE"
mask = pa["assessment_name"].str.upper().str.contains(
    "NVPS|NONVERBAL|PAIN.?SCALE", na=False
)
nvps = pa[mask].copy()
print(f"\nTotal rows matching NVPS/NONVERBAL/PAIN SCALE: {len(nvps):,}")

print("\n" + "=" * 80)
print("ALL MATCHING ASSESSMENT NAMES:")
print("=" * 80)
for name in sorted(nvps["assessment_name"].unique()):
    subset = nvps[nvps["assessment_name"] == name]
    n_rows = len(subset)

    # Check which value columns have data
    has_numerical = subset["numerical_value"].notna().sum()
    has_categorical = subset["categorical_value"].notna().sum()
    has_text = subset["text_value"].notna().sum()

    print(f"\n  {name}")
    print(f"    Rows: {n_rows:,}")
    print(f"    numerical_value non-null: {has_numerical:,}")
    print(f"    categorical_value non-null: {has_categorical:,}")
    print(f"    text_value non-null: {has_text:,}")

    if has_numerical > 0:
        vals = subset["numerical_value"].dropna()
        print(f"    numerical_value distribution:")
        print(f"      {vals.describe().to_string()}")
        print(f"      unique values: {sorted(vals.unique())}")

    if has_categorical > 0:
        cats = subset["categorical_value"].value_counts().head(10)
        print(f"    categorical_value top values:")
        for v, c in cats.items():
            print(f"      {v}: {c:,}")

    if has_text > 0:
        texts = subset["text_value"].value_counts().head(10)
        print(f"    text_value top values:")
        for v, c in texts.items():
            print(f"      {v}: {c:,}")

print("\n" + "=" * 80)
print("DONE")
