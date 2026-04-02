"""Check which column contains NVPS scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_clif_table

pa = load_clif_table("patient_assessments", filters=[("assessment_category", "==", "NVPS")])

print(f"All NVPS rows: {len(pa):,}")
print(f"\nassessment_name values:\n{pa['assessment_name'].value_counts().to_string()}")
print()

for name in pa["assessment_name"].unique():
    subset = pa[pa["assessment_name"] == name]
    print(f"--- {name} ---")
    print(f"  Rows: {len(subset):,}")
    for col in ["numerical_value", "categorical_value", "text_value"]:
        n_notna = subset[col].notna().sum()
        print(f"  {col}: {n_notna:,} non-null")
        if n_notna > 0:
            vals = subset[col].dropna()
            print(f"    unique values (up to 20): {sorted(vals.unique())[:20]}")
    print()
