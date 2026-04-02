"""
01_build_cohort.py
==================
Build the analytic cohort:
  1. Compute MV episodes from respiratory_support (first episode per hospitalization)
  2. Require fentanyl administration during MV
  3. Merge demographics, comorbidities, concurrent sedatives
  4. Compute VFD-28, mortality flags

Outputs:
  - intermediate/mv_episodes.parquet
  - intermediate/cohort.parquet
  - tables/table1_cohort.csv
  - tables/cohort_flow.csv  (exclusion counts)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from tableone import TableOne
from clifpy import compute_sofa_polars

# Add project code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_clif_table,
    load_config,
    save_intermediate,
    save_table,
    FENTANYL_EXCLUDE_NAMES,
    CONCURRENT_SEDATIVES,
    MV_GAP_THRESHOLD_HOURS,
    MV_MIN_DURATION_HOURS,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Step 1: Compute MV episodes
# ──────────────────────────────────────────────
def compute_mv_episodes(gap_hours=MV_GAP_THRESHOLD_HOURS):
    """Identify MV episodes from respiratory_support IMV rows."""
    print("Step 1: Computing MV episodes...")
    rs = load_clif_table(
        "respiratory_support",
        columns=["hospitalization_id", "recorded_dttm", "device_category"],
        filters=[("device_category", "==", "IMV")],
    )
    rs["recorded_dttm"] = pd.to_datetime(rs["recorded_dttm"])
    rs = rs.sort_values(["hospitalization_id", "recorded_dttm"])

    # Detect episode boundaries: gap > threshold between consecutive IMV rows
    rs["prev_dttm"] = rs.groupby("hospitalization_id")["recorded_dttm"].shift(1)
    rs["gap_hours"] = (
        rs["recorded_dttm"] - rs["prev_dttm"]
    ).dt.total_seconds() / 3600
    rs["new_episode"] = (rs["gap_hours"] > gap_hours) | rs["gap_hours"].isna()
    rs["episode_num"] = rs.groupby("hospitalization_id")["new_episode"].cumsum()

    # Summarize episodes
    episodes = (
        rs.groupby(["hospitalization_id", "episode_num"])
        .agg(
            mv_start=("recorded_dttm", "min"),
            mv_end=("recorded_dttm", "max"),
            n_records=("recorded_dttm", "count"),
        )
        .reset_index()
    )
    episodes["mv_duration_hours"] = (
        episodes["mv_end"] - episodes["mv_start"]
    ).dt.total_seconds() / 3600

    print(f"  Total MV episodes: {len(episodes):,}")
    print(f"  Unique hospitalizations with MV: {episodes['hospitalization_id'].nunique():,}")

    # Restrict to first episode per hospitalization
    episodes = episodes.sort_values(["hospitalization_id", "mv_start"])
    first_episodes = episodes.groupby("hospitalization_id").first().reset_index()
    first_episodes = first_episodes.drop(columns=["episode_num"])

    print(f"  First episodes: {len(first_episodes):,}")
    return first_episodes


# ──────────────────────────────────────────────
# Step 2: Apply MV duration filter
# ──────────────────────────────────────────────
def filter_mv_duration(episodes, min_hours=MV_MIN_DURATION_HOURS):
    """Exclude very short MV episodes."""
    n_before = len(episodes)
    episodes = episodes[episodes["mv_duration_hours"] >= min_hours].copy()
    n_after = len(episodes)
    print(f"  Excluded {n_before - n_after:,} episodes with MV < {min_hours}h")
    print(f"  Remaining: {n_after:,}")
    return episodes


# ──────────────────────────────────────────────
# Step 3: Require fentanyl during MV
# ──────────────────────────────────────────────
def filter_fentanyl_during_mv(episodes):
    """Keep only hospitalizations with fentanyl (continuous or intermittent) during MV."""
    print("Step 3: Filtering to patients with fentanyl during MV...")

    # Continuous fentanyl
    fent_c = load_clif_table(
        "medication_admin_continuous",
        columns=["hospitalization_id", "admin_dttm", "med_category", "med_name"],
        filters=[("med_category", "==", "fentanyl")],
    )
    fent_c["admin_dttm"] = pd.to_datetime(fent_c["admin_dttm"])
    # Exclude non-fentanyl drugs
    mask = ~fent_c["med_name"].str.upper().str.contains(
        "|".join(FENTANYL_EXCLUDE_NAMES), na=False
    )
    fent_c = fent_c[mask]

    # Intermittent fentanyl
    fent_i = load_clif_table(
        "medication_admin_intermittent",
        columns=[
            "hospitalization_id", "admin_dttm", "med_category", "med_name",
            "mar_action_category",
        ],
        filters=[("med_category", "==", "fentanyl")],
    )
    fent_i["admin_dttm"] = pd.to_datetime(fent_i["admin_dttm"])
    mask = ~fent_i["med_name"].str.upper().str.contains(
        "|".join(FENTANYL_EXCLUDE_NAMES), na=False
    )
    fent_i = fent_i[mask]
    # Only actually given doses
    fent_i = fent_i[fent_i["mar_action_category"].isin(["given", "bolus"])]

    # Merge with MV episodes to check overlap
    fent_all = pd.concat(
        [
            fent_c[["hospitalization_id", "admin_dttm"]],
            fent_i[["hospitalization_id", "admin_dttm"]],
        ]
    )
    merged = fent_all.merge(episodes, on="hospitalization_id")
    # Fentanyl must occur during MV window
    in_window = merged[
        (merged["admin_dttm"] >= merged["mv_start"])
        & (merged["admin_dttm"] <= merged["mv_end"])
    ]
    fent_hosp_ids = set(in_window["hospitalization_id"].unique())

    episodes_fent = episodes[episodes["hospitalization_id"].isin(fent_hosp_ids)].copy()
    print(f"  Hospitalizations with fentanyl during MV: {len(episodes_fent):,}")

    # Flag fentanyl type
    c_ids = set(
        in_window[in_window["hospitalization_id"].isin(
            set(fent_c["hospitalization_id"])
        )]["hospitalization_id"]
    )
    # Re-check which IDs had continuous vs intermittent within MV window
    fent_c_mv = fent_c.merge(episodes, on="hospitalization_id")
    fent_c_mv = fent_c_mv[
        (fent_c_mv["admin_dttm"] >= fent_c_mv["mv_start"])
        & (fent_c_mv["admin_dttm"] <= fent_c_mv["mv_end"])
    ]
    fent_i_mv = fent_i.merge(episodes, on="hospitalization_id")
    fent_i_mv = fent_i_mv[
        (fent_i_mv["admin_dttm"] >= fent_i_mv["mv_start"])
        & (fent_i_mv["admin_dttm"] <= fent_i_mv["mv_end"])
    ]
    c_ids = set(fent_c_mv["hospitalization_id"].unique())
    i_ids = set(fent_i_mv["hospitalization_id"].unique())

    def fent_type(hid):
        in_c = hid in c_ids
        in_i = hid in i_ids
        if in_c and in_i:
            return "both"
        elif in_c:
            return "continuous"
        else:
            return "intermittent"

    episodes_fent["fentanyl_type"] = episodes_fent["hospitalization_id"].apply(fent_type)
    print(f"  Fentanyl type distribution:")
    print(f"    {episodes_fent['fentanyl_type'].value_counts().to_string()}")

    return episodes_fent


# ──────────────────────────────────────────────
# Step 4: Merge demographics
# ──────────────────────────────────────────────
def merge_demographics(episodes):
    """Add age, sex, race, admission/discharge info, death_dttm."""
    print("Step 4: Merging demographics...")
    hosp = load_clif_table("hospitalization")
    hosp["admission_dttm"] = pd.to_datetime(hosp["admission_dttm"])
    hosp["discharge_dttm"] = pd.to_datetime(hosp["discharge_dttm"])

    # Keep relevant columns
    hosp_cols = [
        "hospitalization_id", "patient_id", "age_at_admission",
        "admission_dttm", "discharge_dttm",
    ]
    # Add columns that exist (schema may vary)
    for col in ["admission_type_category", "discharge_category",
                "sex_category", "race_category", "ethnicity_category"]:
        if col in hosp.columns:
            hosp_cols.append(col)
    hosp = hosp[hosp_cols]

    cohort = episodes.merge(hosp, on="hospitalization_id", how="left")

    # Patient-level data (death)
    patient = load_clif_table("patient", columns=["patient_id", "death_dttm"])
    patient["death_dttm"] = pd.to_datetime(patient["death_dttm"])
    cohort = cohort.merge(patient, on="patient_id", how="left")

    # If sex/race not in hospitalization, check patient table
    if "sex_category" not in cohort.columns:
        pt_demo = load_clif_table("patient")
        for col in ["sex_category", "race_category", "ethnicity_category"]:
            if col in pt_demo.columns and col not in cohort.columns:
                cohort = cohort.merge(
                    pt_demo[["patient_id", col]].drop_duplicates(),
                    on="patient_id", how="left"
                )

    # Adults only
    n_before = len(cohort)
    cohort = cohort[cohort["age_at_admission"] >= 18].copy()
    print(f"  Excluded {n_before - len(cohort):,} pediatric patients")
    print(f"  Cohort size: {len(cohort):,}")

    return cohort


# ──────────────────────────────────────────────
# Step 5: SOFA scores (admission, first 24h of MV)
# ──────────────────────────────────────────────
def add_sofa_scores(cohort):
    """Compute admission SOFA (first 24h of MV) using clifpy."""
    print("Step 5: Computing admission SOFA scores (first 24h of MV)...")
    data_path, _ = load_config()

    # Build Polars cohort with 24h window from MV start
    cohort_pl = pl.DataFrame({
        "hospitalization_id": cohort["hospitalization_id"].tolist(),
        "start_dttm": cohort["mv_start"].tolist(),
        "end_dttm": (cohort["mv_start"] + pd.Timedelta(hours=24)).tolist(),
    })

    sofa = compute_sofa_polars(
        data_directory=data_path,
        cohort_df=cohort_pl,
        filetype="parquet",
        id_name="hospitalization_id",
        timezone="US/Central",
        fill_na_scores_with_zero=True,
        remove_outliers=True,
    )

    # Convert to pandas and merge
    sofa_pd = sofa.to_pandas()
    sofa_cols = ["hospitalization_id", "sofa_total",
                 "sofa_resp", "sofa_coag", "sofa_liver",
                 "sofa_cv_97", "sofa_cns", "sofa_renal"]
    sofa_cols = [c for c in sofa_cols if c in sofa_pd.columns]
    cohort = cohort.merge(sofa_pd[sofa_cols], on="hospitalization_id", how="left")

    n_missing = cohort["sofa_total"].isna().sum()
    print(f"  SOFA scores computed for {len(cohort) - n_missing:,}/{len(cohort):,} patients")
    print(f"  SOFA total — median: {cohort['sofa_total'].median():.0f}, "
          f"IQR: [{cohort['sofa_total'].quantile(0.25):.0f}-{cohort['sofa_total'].quantile(0.75):.0f}]")

    return cohort


# ──────────────────────────────────────────────
# Step 6: Concurrent sedatives
# ──────────────────────────────────────────────
def add_concurrent_sedatives(cohort):
    """Flag propofol, dexmedetomidine, midazolam use during MV."""
    print("Step 6: Checking concurrent sedatives...")
    meds = load_clif_table(
        "medication_admin_continuous",
        columns=["hospitalization_id", "admin_dttm", "med_category"],
    )
    meds["admin_dttm"] = pd.to_datetime(meds["admin_dttm"])

    for sed in CONCURRENT_SEDATIVES:
        sed_df = meds[meds["med_category"] == sed].copy()
        sed_mv = sed_df.merge(
            cohort[["hospitalization_id", "mv_start", "mv_end"]],
            on="hospitalization_id",
        )
        sed_mv = sed_mv[
            (sed_mv["admin_dttm"] >= sed_mv["mv_start"])
            & (sed_mv["admin_dttm"] <= sed_mv["mv_end"])
        ]
        ids_with_sed = set(sed_mv["hospitalization_id"].unique())
        cohort[f"received_{sed}"] = cohort["hospitalization_id"].isin(ids_with_sed)
        n = cohort[f"received_{sed}"].sum()
        print(f"  {sed}: {n:,} patients ({100*n/len(cohort):.1f}%)")

    return cohort


# ──────────────────────────────────────────────
# Step 7: Mortality and VFD-28
# ──────────────────────────────────────────────
def add_outcomes(cohort):
    """Compute mortality flags and ventilator-free days at 28 days."""
    print("Step 7: Computing outcomes (mortality, VFD-28)...")

    # In-hospital mortality
    cohort["in_hospital_death"] = cohort["death_dttm"].notna() & (
        cohort["death_dttm"] <= cohort["discharge_dttm"] + pd.Timedelta(hours=24)
    )

    # 28-day mortality (from MV start)
    day28 = cohort["mv_start"] + pd.Timedelta(days=28)
    cohort["death_within_28d"] = cohort["death_dttm"].notna() & (
        cohort["death_dttm"] <= day28
    )

    # MV duration in days
    cohort["mv_duration_days"] = cohort["mv_duration_hours"] / 24

    # VFD-28: 28 - MV_days if alive at 28 days, else 0
    cohort["vfd_28"] = np.where(
        cohort["death_within_28d"],
        0,
        np.maximum(0, 28 - cohort["mv_duration_days"]),
    )

    print(f"  In-hospital mortality: {cohort['in_hospital_death'].sum():,} ({100*cohort['in_hospital_death'].mean():.1f}%)")
    print(f"  28-day mortality: {cohort['death_within_28d'].sum():,} ({100*cohort['death_within_28d'].mean():.1f}%)")
    print(f"  VFD-28 median: {cohort['vfd_28'].median():.1f} days")

    return cohort


# ──────────────────────────────────────────────
# Step 8: Table 1
# ──────────────────────────────────────────────
def create_table1(cohort):
    """Generate Table 1 summary statistics."""
    print("Step 8: Creating Table 1...")

    columns_for_table1 = [
        "age_at_admission", "mv_duration_hours", "mv_duration_days",
        "sofa_total", "vfd_28", "fentanyl_type",
        "in_hospital_death", "death_within_28d",
    ]
    # Add demographics if available
    for col in ["sex_category", "race_category", "ethnicity_category"]:
        if col in cohort.columns:
            columns_for_table1.append(col)
    # Add concurrent sedatives
    for sed in CONCURRENT_SEDATIVES:
        col = f"received_{sed}"
        if col in cohort.columns:
            columns_for_table1.append(col)

    # Filter to columns that exist
    columns_for_table1 = [c for c in columns_for_table1 if c in cohort.columns]

    categorical = [
        c for c in columns_for_table1
        if cohort[c].dtype == "object" or cohort[c].dtype == "bool"
        or c in ["fentanyl_type", "sex_category", "race_category", "ethnicity_category"]
    ]
    nonnormal = ["mv_duration_hours", "mv_duration_days", "sofa_total", "vfd_28"]
    nonnormal = [c for c in nonnormal if c in columns_for_table1]

    table1 = TableOne(
        cohort,
        columns=columns_for_table1,
        categorical=categorical,
        nonnormal=nonnormal,
        missing=True,
    )

    table1_df = table1.tableone
    save_table(table1_df.reset_index(), "table1_cohort")
    print(table1)
    return table1


# ──────────────────────────────────────────────
# Step 9: Cohort flow diagram
# ──────────────────────────────────────────────
def save_cohort_flow(flow_steps):
    """Save the cohort exclusion flow as a table."""
    flow_df = pd.DataFrame(flow_steps, columns=["Step", "N"])
    save_table(flow_df, "cohort_flow")
    print("\n  Cohort flow:")
    for _, row in flow_df.iterrows():
        print(f"    {row['Step']}: {row['N']:,}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("01_build_cohort.py — Building analytic cohort")
    print("=" * 60)

    flow = []

    # Step 1: MV episodes
    episodes = compute_mv_episodes()
    flow.append(("All first MV episodes", len(episodes)))

    # Step 2: Duration filter
    print(f"\nStep 2: Applying MV duration filter (>= {MV_MIN_DURATION_HOURS}h)...")
    episodes = filter_mv_duration(episodes)
    flow.append((f"MV >= {MV_MIN_DURATION_HOURS} hours", len(episodes)))

    # Step 3: Fentanyl filter
    episodes = filter_fentanyl_during_mv(episodes)
    flow.append(("Received fentanyl during MV", len(episodes)))

    # Step 4: Demographics
    cohort = merge_demographics(episodes)
    flow.append(("Adults (age >= 18)", len(cohort)))

    # Step 5: SOFA scores
    cohort = add_sofa_scores(cohort)

    # Step 6: Concurrent sedatives
    cohort = add_concurrent_sedatives(cohort)

    # Step 7: Outcomes
    cohort = add_outcomes(cohort)

    # Save MV episodes and cohort
    save_intermediate(
        cohort[["hospitalization_id", "mv_start", "mv_end",
                "mv_duration_hours", "n_records"]],
        "mv_episodes",
    )
    save_intermediate(cohort, "cohort")

    # Step 8: Table 1
    create_table1(cohort)

    # Step 9: Cohort flow
    save_cohort_flow(flow)

    print("\n" + "=" * 60)
    print(f"DONE. Final cohort: {len(cohort):,} hospitalizations")
    print("=" * 60)


if __name__ == "__main__":
    main()
