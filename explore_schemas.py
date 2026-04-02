"""
Explore CLIF 2.1.0 table schemas and category value counts.
Only prints aggregated summaries — no individual patient records.
"""
import pandas as pd
import os

DATA_DIR = "/Users/shanguleria/Desktop/Research/CLIF/CLIF Data/CLIF_2.1.0"

def show_schema(name, df):
    print(f"\n{'='*70}")
    print(f"TABLE: {name}")
    print(f"{'='*70}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    print("COLUMNS & DTYPES:")
    for col in df.columns:
        print(f"  {col:40s} {str(df[col].dtype)}")
    print()

def show_value_counts(df, col):
    print(f"VALUE COUNTS for '{col}':")
    vc = df[col].value_counts(dropna=False)
    for val, cnt in vc.items():
        print(f"  {str(val):50s} {cnt:>10,}")
    print()

def show_unique(df, col):
    vals = sorted(df[col].dropna().unique().tolist())
    print(f"UNIQUE VALUES for '{col}' ({len(vals)} unique):")
    for v in vals:
        print(f"  {v}")
    print()

# 1. clif_patient_assessments
fp = os.path.join(DATA_DIR, "clif_patient_assessments.parquet")
df = pd.read_parquet(fp)
show_schema("clif_patient_assessments", df)
show_value_counts(df, "assessment_category")

# 2. clif_medication_admin_continuous
fp = os.path.join(DATA_DIR, "clif_medication_admin_continuous.parquet")
df = pd.read_parquet(fp)
show_schema("clif_medication_admin_continuous", df)
show_value_counts(df, "med_category")
show_unique(df, "med_route_category")
show_unique(df, "med_dose_unit")

# 3. clif_medication_admin_intermittent
fp = os.path.join(DATA_DIR, "clif_medication_admin_intermittent.parquet")
df = pd.read_parquet(fp)
show_schema("clif_medication_admin_intermittent", df)
show_value_counts(df, "med_category")
show_unique(df, "med_route_category")
show_unique(df, "med_dose_unit")

# 4. clif_respiratory_support
fp = os.path.join(DATA_DIR, "clif_respiratory_support.parquet")
df = pd.read_parquet(fp)
show_schema("clif_respiratory_support", df)
show_value_counts(df, "device_category")

# 5. clif_hospitalization
fp = os.path.join(DATA_DIR, "clif_hospitalization.parquet")
df = pd.read_parquet(fp)
show_schema("clif_hospitalization", df)

print("\nDone.")
