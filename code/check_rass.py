"""Check RASS data structure and values."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_clif_table, load_intermediate

pa = load_clif_table("patient_assessments", filters=[("assessment_category", "==", "RASS")])
cohort = load_intermediate("cohort")
cohort_ids = set(cohort["hospitalization_id"])

print(f"All RASS rows: {len(pa):,}")
print(f"RASS in cohort: {pa['hospitalization_id'].isin(cohort_ids).sum():,}")
print(f"\nassessment_name values:\n{pa['assessment_name'].value_counts().to_string()}")
print()

for col in ["numerical_value", "categorical_value", "text_value"]:
    n_notna = pa[col].notna().sum()
    print(f"{col}: {n_notna:,} non-null")
    if n_notna > 0:
        vals = pa[col].dropna()
        print(f"  unique values: {sorted(vals.unique())[:30]}")
        print(f"  value_counts:\n{vals.value_counts().head(15).to_string()}")
    print()
