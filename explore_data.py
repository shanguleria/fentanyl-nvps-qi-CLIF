"""Explore NVPS, fentanyl, and IMV data characteristics for study planning."""
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

with open('config.json') as f:
    config = json.load(f)
DATA = config['clif_data_path']

# 1. NVPS in patient_assessments
pa = pd.read_parquet(f"{DATA}/clif_patient_assessments.parquet")
nvps = pa[pa['assessment_category'] == 'NVPS']
print("=== NVPS ===")
print(f"Total rows: {len(nvps):,}")
print(f"Unique hospitalizations: {nvps['hospitalization_id'].nunique():,}")
print(f"\nassessment_name value_counts:\n{nvps['assessment_name'].value_counts().to_string()}")
print(f"\nassessment_group value_counts:\n{nvps['assessment_group'].value_counts().to_string()}")
print(f"\nnumerical_value describe:\n{nvps['numerical_value'].describe().to_string()}")
cat_vals = nvps['categorical_value'].dropna().unique()
print(f"\ncategorical_value unique ({len(cat_vals)}): {cat_vals[:20]}")
del pa

# 2. Fentanyl continuous
mc = pd.read_parquet(f"{DATA}/clif_medication_admin_continuous.parquet")
fent_c = mc[mc['med_category'] == 'fentanyl']
print("\n=== FENTANYL CONTINUOUS ===")
print(f"Total rows: {len(fent_c):,}")
print(f"Unique hospitalizations: {fent_c['hospitalization_id'].nunique():,}")
print(f"\nmed_name value_counts:\n{fent_c['med_name'].value_counts().to_string()}")
print(f"\nmed_dose_unit value_counts:\n{fent_c['med_dose_unit'].value_counts().to_string()}")
print(f"\nmed_dose describe:\n{fent_c['med_dose'].describe().to_string()}")
print(f"\nmar_action_category value_counts:\n{fent_c['mar_action_category'].value_counts().to_string()}")
fent_c_ids = set(fent_c['hospitalization_id'].unique())
del mc

# 3. Fentanyl intermittent
mi = pd.read_parquet(f"{DATA}/clif_medication_admin_intermittent.parquet")
fent_i = mi[mi['med_category'] == 'fentanyl']
print("\n=== FENTANYL INTERMITTENT ===")
print(f"Total rows: {len(fent_i):,}")
print(f"Unique hospitalizations: {fent_i['hospitalization_id'].nunique():,}")
print(f"\nmed_name value_counts:\n{fent_i['med_name'].value_counts().to_string()}")
print(f"\nmed_dose_unit value_counts:\n{fent_i['med_dose_unit'].value_counts().to_string()}")
print(f"\nmed_dose describe:\n{fent_i['med_dose'].describe().to_string()}")
print(f"\nmed_route_category value_counts:\n{fent_i['med_route_category'].value_counts().to_string()}")
print(f"\nmar_action_category value_counts:\n{fent_i['mar_action_category'].value_counts().to_string()}")
fent_i_ids = set(fent_i['hospitalization_id'].unique())
del mi

# 4. Overlap
fent_all_ids = fent_c_ids | fent_i_ids
nvps_ids = set(nvps['hospitalization_id'].unique())
print(f"\n=== OVERLAP ===")
print(f"Any fentanyl: {len(fent_all_ids):,}")
print(f"Any NVPS: {len(nvps_ids):,}")
print(f"Both fentanyl AND NVPS: {len(fent_all_ids & nvps_ids):,}")
del nvps

# 5. IMV overlap
rs = pd.read_parquet(f"{DATA}/clif_respiratory_support.parquet")
imv = rs[rs['device_category'] == 'IMV']
imv_ids = set(imv['hospitalization_id'].unique())
print(f"\nIMV hospitalizations: {len(imv_ids):,}")
print(f"IMV + any fentanyl: {len(imv_ids & fent_all_ids):,}")
print(f"IMV + fentanyl continuous only: {len(imv_ids & fent_c_ids):,}")
print(f"IMV + NVPS: {len(imv_ids & nvps_ids):,}")
print(f"IMV + fentanyl + NVPS: {len(imv_ids & fent_all_ids & nvps_ids):,}")
