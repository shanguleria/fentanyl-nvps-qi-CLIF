"""
10_propofol_dosing.py
=====================
Propofol dosing characterization during mechanical ventilation.
Mirrors 02_aim1_fentanyl.py but for propofol (continuous only, no bolus analysis).

Requires: 01_build_cohort.py outputs

Outputs:
  - tables/table17_propofol_summary.csv
  - figures/fig24_propofol_trajectory.pdf/.png
  - figures/fig25_propofol_dose_distribution.pdf/.png
  - intermediate/propofol_hourly.parquet
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_clif_table,
    load_intermediate,
    save_intermediate,
    save_table,
    save_figure,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

PROPOFOL_MAX_DOSE = 200  # mcg/kg/min — outlier threshold


# ──────────────────────────────────────────────
# Step 1: Load cohort and propofol data
# ──────────────────────────────────────────────
def load_data():
    """Load cohort and propofol continuous data during MV."""
    print("Step 1: Loading cohort and propofol data...")
    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])
    mv_windows = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()
    print(f"  Cohort: {len(cohort):,} patients")

    # Load continuous propofol
    prop = load_clif_table(
        "medication_admin_continuous",
        filters=[("med_category", "==", "propofol")],
    )
    prop["admin_dttm"] = pd.to_datetime(prop["admin_dttm"])

    # Restrict to MV windows
    prop = prop.merge(mv_windows, on="hospitalization_id")
    prop = prop[
        (prop["admin_dttm"] >= prop["mv_start"])
        & (prop["admin_dttm"] <= prop["mv_end"])
    ]
    prop["hours_from_mv_start"] = (
        prop["admin_dttm"] - prop["mv_start"]
    ).dt.total_seconds() / 3600

    print(f"  Propofol records during MV: {len(prop):,}")
    print(f"  Patients with propofol: {prop['hospitalization_id'].nunique():,}")

    return cohort, mv_windows, prop


# ──────────────────────────────────────────────
# Step 2: Standardize propofol doses
# ──────────────────────────────────────────────
def standardize_doses(prop):
    """Verify units and clean propofol doses (expected: mcg/kg/min)."""
    print("\nStep 2: Standardizing propofol doses...")
    print(f"  Dose units:")
    print(f"    {prop['med_dose_unit'].value_counts().to_string()}")

    # All propofol should be mcg/kg/min — use as-is
    prop["dose_mcg_kg_min"] = prop["med_dose"].copy()

    # Handle stop actions
    prop.loc[prop["mar_action_category"] == "stop", "dose_mcg_kg_min"] = 0

    # Remove missing/outlier doses
    n_before = len(prop)
    prop = prop[prop["dose_mcg_kg_min"].notna()].copy()
    n_outlier = (prop["dose_mcg_kg_min"] > PROPOFOL_MAX_DOSE).sum()
    prop = prop[
        (prop["dose_mcg_kg_min"] >= 0) & (prop["dose_mcg_kg_min"] <= PROPOFOL_MAX_DOSE)
    ].copy()

    if n_outlier > 0:
        print(f"  Removed {n_outlier:,} records with dose > {PROPOFOL_MAX_DOSE} mcg/kg/min")
    print(f"  Remaining records: {len(prop):,}")
    print(f"  Dose (mcg/kg/min) summary:")
    print(f"    {prop['dose_mcg_kg_min'].describe().to_string()}")

    return prop


# ──────────────────────────────────────────────
# Step 3: Hourly trajectory
# ──────────────────────────────────────────────
def compute_hourly_trajectory(prop, max_hours=168):
    """Compute hourly propofol dose trajectory using LOCF."""
    print(f"\nStep 3: Computing hourly trajectory (up to {max_hours}h)...")

    prop["hour_bin"] = prop["hours_from_mv_start"].astype(int)
    prop = prop[prop["hour_bin"] < max_hours]

    # Last observation per hour per patient
    last_per_hour = (
        prop.sort_values("admin_dttm")
        .groupby(["hospitalization_id", "hour_bin"])
        .last()
        .reset_index()
    )

    # Full hour grid per patient with forward-fill
    patients = last_per_hour["hospitalization_id"].unique()
    all_hours = []
    for pid in patients:
        pt_data = last_per_hour[last_per_hour["hospitalization_id"] == pid][
            ["hour_bin", "dose_mcg_kg_min"]
        ].set_index("hour_bin")
        max_h = min(int(pt_data.index.max()), max_hours - 1)
        full_range = pd.DataFrame({"hour_bin": range(0, max_h + 1)}).set_index("hour_bin")
        filled = full_range.join(pt_data).ffill()
        filled["hospitalization_id"] = pid
        all_hours.append(filled.reset_index())

    hourly_df = pd.concat(all_hours, ignore_index=True)

    # Summary by hour
    hourly_summary = (
        hourly_df.groupby("hour_bin")["dose_mcg_kg_min"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
        .reset_index()
    )
    hourly_summary.columns = ["hour", "median", "q25", "q75", "n_patients"]

    # Only show hours where >= 25% of initial patients remain
    initial_n = hourly_summary["n_patients"].iloc[0] if len(hourly_summary) > 0 else 0
    hourly_summary = hourly_summary[hourly_summary["n_patients"] >= initial_n * 0.25]

    print(f"  Trajectory computed for {len(patients):,} patients")
    print(f"  Hours shown (>=25% patients): {len(hourly_summary)}")

    return hourly_df, hourly_summary


# ──────────────────────────────────────────────
# Step 4: Per-patient summaries
# ──────────────────────────────────────────────
def compute_patient_summaries(prop, cohort):
    """Compute per-patient propofol summary statistics."""
    print("\nStep 4: Computing per-patient propofol summaries...")

    # Only patients who received propofol
    prop_ids = set(prop["hospitalization_id"].unique())
    summaries = cohort[cohort["hospitalization_id"].isin(prop_ids)][
        ["hospitalization_id", "mv_duration_hours"]
    ].copy()

    cont_stats = (
        prop.groupby("hospitalization_id")
        .agg(
            first_dose=("dose_mcg_kg_min", "first"),
            peak_dose=("dose_mcg_kg_min", "max"),
            mean_dose=("dose_mcg_kg_min", "mean"),
            median_dose=("dose_mcg_kg_min", "median"),
            n_records=("dose_mcg_kg_min", "count"),
        )
        .reset_index()
    )

    # Infusion duration
    duration = (
        prop.groupby("hospitalization_id")["admin_dttm"]
        .agg(["min", "max"])
        .reset_index()
    )
    duration["infusion_duration_hours"] = (
        duration["max"] - duration["min"]
    ).dt.total_seconds() / 3600
    cont_stats = cont_stats.merge(
        duration[["hospitalization_id", "infusion_duration_hours"]],
        on="hospitalization_id",
        how="left",
    )

    summaries = summaries.merge(cont_stats, on="hospitalization_id", how="left")
    return summaries


# ──────────────────────────────────────────────
# Step 5: Figures
# ──────────────────────────────────────────────
def plot_trajectory(hourly_summary):
    """Plot propofol dose trajectory over MV course."""
    print("\nStep 5a: Plotting propofol trajectory...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_summary["hour"], hourly_summary["median"], color="darkorange", linewidth=2)
    ax.fill_between(
        hourly_summary["hour"],
        hourly_summary["q25"],
        hourly_summary["q75"],
        alpha=0.3,
        color="darkorange",
        label="IQR",
    )
    ax.set_xlabel("Hours from MV Start")
    ax.set_ylabel("Propofol Dose (mcg/kg/min)")
    ax.set_title("Continuous Propofol Dose Trajectory During Mechanical Ventilation")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(
        hourly_summary["hour"], hourly_summary["n_patients"],
        color="gray", alpha=0.4, linestyle="--", linewidth=1,
    )
    ax2.set_ylabel("N patients still on infusion", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    save_figure(fig, "fig24_propofol_trajectory")
    plt.close(fig)


def plot_dose_distribution(summaries):
    """Plot distribution of starting and peak propofol doses."""
    print("Step 5b: Plotting dose distributions...")
    cont_patients = summaries[summaries["first_dose"] > 0]

    if len(cont_patients) == 0:
        print("  No propofol data for dose distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(cont_patients["first_dose"], bins=50, color="darkorange", edgecolor="white")
    axes[0].set_xlabel("Dose (mcg/kg/min)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Initial Propofol Infusion Dose")

    axes[1].hist(cont_patients["peak_dose"], bins=50, color="coral", edgecolor="white")
    axes[1].set_xlabel("Dose (mcg/kg/min)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Peak Propofol Infusion Dose")

    fig.tight_layout()
    save_figure(fig, "fig25_propofol_dose_distribution")
    plt.close(fig)


# ──────────────────────────────────────────────
# Step 6: Summary table
# ──────────────────────────────────────────────
def create_summary_table(summaries, cohort):
    """Create propofol dosing summary table."""
    print("\nStep 6: Creating propofol summary table...")

    n_cohort = len(cohort)
    n_propofol = len(summaries)

    rows = []
    rows.append(("Total cohort", f"{n_cohort:,}", ""))
    rows.append(("Received propofol during MV", f"{n_propofol:,}", f"{100*n_propofol/n_cohort:.1f}%"))

    cont = summaries[summaries["first_dose"] > 0]
    if len(cont) > 0:
        rows.append(("", "", ""))
        rows.append(("Propofol infusion", "", ""))
        for label, col in [
            ("Starting dose (mcg/kg/min)", "first_dose"),
            ("Peak dose (mcg/kg/min)", "peak_dose"),
            ("Mean dose (mcg/kg/min)", "mean_dose"),
            ("Infusion duration (hours)", "infusion_duration_hours"),
        ]:
            med = cont[col].median()
            q25 = cont[col].quantile(0.25)
            q75 = cont[col].quantile(0.75)
            rows.append((label, f"{med:.1f}", f"[{q25:.1f} - {q75:.1f}]"))

    summary_df = pd.DataFrame(rows, columns=["Variable", "Median", "IQR"])
    save_table(summary_df, "table17_propofol_summary")
    print(summary_df.to_string(index=False))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("10_propofol_dosing.py — Propofol Dosing Characterization")
    print("=" * 60)

    cohort, mv_windows, prop = load_data()
    prop = standardize_doses(prop)

    hourly_df, hourly_summary = compute_hourly_trajectory(prop)
    save_intermediate(
        hourly_df[["hospitalization_id", "hour_bin", "dose_mcg_kg_min"]],
        "propofol_hourly",
    )

    summaries = compute_patient_summaries(prop, cohort)

    plot_trajectory(hourly_summary)
    plot_dose_distribution(summaries)
    create_summary_table(summaries, cohort)

    print("\n" + "=" * 60)
    print("DONE. Propofol dosing characterization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
