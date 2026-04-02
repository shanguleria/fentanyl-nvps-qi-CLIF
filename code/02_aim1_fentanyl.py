"""
02_aim1_fentanyl.py
===================
Aim 1: Characterize fentanyl dosing over the course of mechanical ventilation.

Requires: 01_build_cohort.py outputs (intermediate/cohort.parquet, intermediate/mv_episodes.parquet)

Outputs:
  - tables/table2_fentanyl_summary.csv
  - figures/fig1_fentanyl_trajectory.pdf/.png
  - figures/fig2_fentanyl_dose_distribution.pdf/.png
  - figures/fig3_fentanyl_bolus_pattern.pdf/.png
  - intermediate/fentanyl_hourly.parquet
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
    FENTANYL_EXCLUDE_NAMES,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ──────────────────────────────────────────────
# Step 1: Load cohort and fentanyl data
# ──────────────────────────────────────────────
def load_data():
    """Load cohort and fentanyl medication data."""
    print("Step 1: Loading cohort and fentanyl data...")
    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])
    mv_windows = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()
    print(f"  Cohort: {len(cohort):,} patients")
    return cohort, mv_windows


def load_fentanyl_continuous(mv_windows):
    """Load and filter continuous fentanyl to MV windows."""
    fent = load_clif_table(
        "medication_admin_continuous",
        filters=[("med_category", "==", "fentanyl")],
    )
    fent["admin_dttm"] = pd.to_datetime(fent["admin_dttm"])

    # Exclude non-fentanyl
    mask = ~fent["med_name"].str.upper().str.contains(
        "|".join(FENTANYL_EXCLUDE_NAMES), na=False
    )
    fent = fent[mask]

    # Restrict to MV windows
    fent = fent.merge(mv_windows, on="hospitalization_id")
    fent = fent[(fent["admin_dttm"] >= fent["mv_start"]) & (fent["admin_dttm"] <= fent["mv_end"])]
    fent["hours_from_mv_start"] = (fent["admin_dttm"] - fent["mv_start"]).dt.total_seconds() / 3600

    print(f"  Continuous fentanyl records during MV: {len(fent):,}")
    print(f"  Patients with continuous: {fent['hospitalization_id'].nunique():,}")
    return fent


def load_fentanyl_intermittent(mv_windows):
    """Load and filter intermittent fentanyl to MV windows."""
    fent = load_clif_table(
        "medication_admin_intermittent",
        filters=[("med_category", "==", "fentanyl")],
    )
    fent["admin_dttm"] = pd.to_datetime(fent["admin_dttm"])

    # Exclude non-fentanyl
    mask = ~fent["med_name"].str.upper().str.contains(
        "|".join(FENTANYL_EXCLUDE_NAMES), na=False
    )
    fent = fent[mask]

    # Only actually given doses
    fent = fent[fent["mar_action_category"].isin(["given", "bolus"])]

    # Restrict to MV windows
    fent = fent.merge(mv_windows, on="hospitalization_id")
    fent = fent[(fent["admin_dttm"] >= fent["mv_start"]) & (fent["admin_dttm"] <= fent["mv_end"])]
    fent["hours_from_mv_start"] = (fent["admin_dttm"] - fent["mv_start"]).dt.total_seconds() / 3600

    print(f"  Intermittent fentanyl records during MV: {len(fent):,}")
    print(f"  Patients with intermittent: {fent['hospitalization_id'].nunique():,}")
    return fent


# ──────────────────────────────────────────────
# Step 2: Standardize continuous fentanyl doses
# ──────────────────────────────────────────────
def standardize_continuous_doses(fent_c):
    """Convert all continuous fentanyl doses to mcg/hr."""
    print("\nStep 2: Standardizing continuous fentanyl doses...")
    print(f"  Dose units before conversion:")
    print(f"    {fent_c['med_dose_unit'].value_counts().to_string()}")

    # For mcg/kg/hr, we need patient weight
    needs_weight = fent_c["med_dose_unit"] == "mcg/kg/hr"
    n_needs = needs_weight.sum()

    if n_needs > 0:
        print(f"  {n_needs:,} records need weight-based conversion...")
        # Get weight from vitals
        weight_ids = fent_c.loc[needs_weight, "hospitalization_id"].unique()
        vitals = load_clif_table(
            "vitals",
            columns=["hospitalization_id", "recorded_dttm", "vital_category", "vital_value"],
            filters=[("vital_category", "==", "weight_kg")],
        )
        vitals = vitals[vitals["hospitalization_id"].isin(weight_ids)]
        vitals["vital_value"] = pd.to_numeric(vitals["vital_value"], errors="coerce")

        # Use median weight per patient
        weight_by_patient = (
            vitals.groupby("hospitalization_id")["vital_value"]
            .median()
            .reset_index()
            .rename(columns={"vital_value": "weight_kg"})
        )
        fent_c = fent_c.merge(weight_by_patient, on="hospitalization_id", how="left")

        # Recompute mask after merge (index may have changed)
        needs_weight = fent_c["med_dose_unit"] == "mcg/kg/hr"

        # Fill missing weights with population median (80 kg)
        n_missing_weight = fent_c.loc[needs_weight, "weight_kg"].isna().sum()
        if n_missing_weight > 0:
            print(f"  WARNING: {n_missing_weight:,} records missing weight, using 80 kg default")
        fent_c["weight_kg"] = fent_c["weight_kg"].fillna(80.0)

        # Convert
        fent_c["dose_mcg_hr"] = np.where(
            fent_c["med_dose_unit"] == "mcg/kg/hr",
            fent_c["med_dose"] * fent_c["weight_kg"],
            fent_c["med_dose"],
        )
    else:
        fent_c["dose_mcg_hr"] = fent_c["med_dose"]

    # Handle stop actions → dose = 0
    fent_c.loc[fent_c["mar_action_category"] == "stop", "dose_mcg_hr"] = 0

    # Remove implausible values
    n_before = len(fent_c)
    fent_c = fent_c[fent_c["dose_mcg_hr"].notna()].copy()
    n_missing = n_before - len(fent_c)
    n_outlier = (fent_c["dose_mcg_hr"] > 500).sum()
    fent_c = fent_c[(fent_c["dose_mcg_hr"] >= 0) & (fent_c["dose_mcg_hr"] <= 500)].copy()
    n_after = len(fent_c)
    if n_missing > 0:
        print(f"  Removed {n_missing:,} records with missing doses")
    if n_outlier > 0:
        print(f"  Removed {n_outlier:,} records with dose > 500 mcg/hr (outliers)")
    print(f"  Remaining records: {n_after:,}")

    print(f"  Dose (mcg/hr) after standardization:")
    print(f"    {fent_c['dose_mcg_hr'].describe().to_string()}")

    return fent_c


# ──────────────────────────────────────────────
# Step 3: Hourly trajectory (continuous)
# ──────────────────────────────────────────────
def compute_hourly_trajectory(fent_c, max_hours=168):
    """Compute hourly fentanyl dose trajectory using LOCF."""
    print(f"\nStep 3: Computing hourly trajectory (up to {max_hours}h)...")

    # Bin into hours
    fent_c["hour_bin"] = fent_c["hours_from_mv_start"].astype(int)
    fent_c = fent_c[fent_c["hour_bin"] < max_hours]

    # Last observation per hour per patient
    last_per_hour = (
        fent_c.sort_values("admin_dttm")
        .groupby(["hospitalization_id", "hour_bin"])
        .last()
        .reset_index()
    )

    # Create full hour grid per patient and forward-fill
    patients = last_per_hour["hospitalization_id"].unique()
    all_hours = []
    for pid in patients:
        pt_data = last_per_hour[last_per_hour["hospitalization_id"] == pid][
            ["hour_bin", "dose_mcg_hr"]
        ].set_index("hour_bin")
        max_h = min(int(pt_data.index.max()), max_hours - 1)
        full_range = pd.DataFrame({"hour_bin": range(0, max_h + 1)}).set_index("hour_bin")
        filled = full_range.join(pt_data).ffill()
        filled["hospitalization_id"] = pid
        all_hours.append(filled.reset_index())

    hourly_df = pd.concat(all_hours, ignore_index=True)

    # Summary statistics by hour
    hourly_summary = (
        hourly_df.groupby("hour_bin")["dose_mcg_hr"]
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
# Step 4: Per-patient summary statistics
# ──────────────────────────────────────────────
def compute_patient_summaries(fent_c, fent_i, cohort):
    """Compute per-patient fentanyl summary statistics."""
    print("\nStep 4: Computing per-patient fentanyl summaries...")

    summaries = cohort[["hospitalization_id", "fentanyl_type", "mv_duration_hours"]].copy()

    # Continuous summaries
    if len(fent_c) > 0:
        cont_stats = (
            fent_c.groupby("hospitalization_id")
            .agg(
                first_dose_mcg_hr=("dose_mcg_hr", "first"),
                peak_dose_mcg_hr=("dose_mcg_hr", "max"),
                mean_dose_mcg_hr=("dose_mcg_hr", "mean"),
                median_dose_mcg_hr=("dose_mcg_hr", "median"),
                n_continuous_records=("dose_mcg_hr", "count"),
            )
            .reset_index()
        )
        # Duration of continuous infusion (first to last record)
        infusion_duration = (
            fent_c.groupby("hospitalization_id")["admin_dttm"]
            .agg(["min", "max"])
            .reset_index()
        )
        infusion_duration["infusion_duration_hours"] = (
            infusion_duration["max"] - infusion_duration["min"]
        ).dt.total_seconds() / 3600
        cont_stats = cont_stats.merge(
            infusion_duration[["hospitalization_id", "infusion_duration_hours"]],
            on="hospitalization_id",
            how="left",
        )
        summaries = summaries.merge(cont_stats, on="hospitalization_id", how="left")

    # Intermittent summaries
    if len(fent_i) > 0:
        int_stats = (
            fent_i.groupby("hospitalization_id")
            .agg(
                n_boluses=("med_dose", "count"),
                total_bolus_dose_mcg=("med_dose", "sum"),
                mean_bolus_dose_mcg=("med_dose", "mean"),
            )
            .reset_index()
        )
        summaries = summaries.merge(int_stats, on="hospitalization_id", how="left")

    # Fill NaN for patients without continuous or intermittent
    for col in summaries.columns:
        if col not in ["hospitalization_id", "fentanyl_type", "mv_duration_hours"]:
            summaries[col] = summaries[col].fillna(0)

    return summaries


# ──────────────────────────────────────────────
# Step 5: Figures
# ──────────────────────────────────────────────
def plot_trajectory(hourly_summary):
    """Plot fentanyl dose trajectory over MV course."""
    print("\nStep 5a: Plotting fentanyl trajectory...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_summary["hour"], hourly_summary["median"], color="steelblue", linewidth=2)
    ax.fill_between(
        hourly_summary["hour"],
        hourly_summary["q25"],
        hourly_summary["q75"],
        alpha=0.3,
        color="steelblue",
        label="IQR",
    )
    ax.set_xlabel("Hours from MV Start")
    ax.set_ylabel("Fentanyl Dose (mcg/hr)")
    ax.set_title("Continuous Fentanyl Dose Trajectory During Mechanical Ventilation")
    ax.legend()

    # Add patient count on secondary axis
    ax2 = ax.twinx()
    ax2.plot(
        hourly_summary["hour"], hourly_summary["n_patients"],
        color="gray", alpha=0.4, linestyle="--", linewidth=1,
    )
    ax2.set_ylabel("N patients still on infusion", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.tight_layout()
    save_figure(fig, "fig1_fentanyl_trajectory")
    plt.close(fig)


def plot_dose_distribution(summaries):
    """Plot distribution of starting and peak fentanyl doses."""
    print("Step 5b: Plotting dose distributions...")
    has_cont = summaries["peak_dose_mcg_hr"].sum() > 0

    if has_cont:
        cont_patients = summaries[summaries["first_dose_mcg_hr"] > 0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(cont_patients["first_dose_mcg_hr"], bins=50, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("Dose (mcg/hr)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Initial Fentanyl Infusion Dose")

        axes[1].hist(cont_patients["peak_dose_mcg_hr"], bins=50, color="coral", edgecolor="white")
        axes[1].set_xlabel("Dose (mcg/hr)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Peak Fentanyl Infusion Dose")

        fig.tight_layout()
        save_figure(fig, "fig2_fentanyl_dose_distribution")
        plt.close(fig)
    else:
        print("  No continuous fentanyl data for dose distribution plot.")


def plot_bolus_pattern(fent_i, mv_windows):
    """Plot intermittent fentanyl bolus rate per patient-hour and dose distribution."""
    print("Step 5c: Plotting bolus pattern...")
    if len(fent_i) == 0:
        print("  No intermittent fentanyl data.")
        return

    max_hours = 168

    # Compute MV duration in hours per patient for at-risk denominator
    mv = mv_windows.copy()
    mv["mv_duration_h"] = (mv["mv_end"] - mv["mv_start"]).dt.total_seconds() / 3600
    mv["mv_duration_h"] = mv["mv_duration_h"].clip(upper=max_hours)

    # Patients at risk at each hour = patients whose MV duration > that hour
    hours = np.arange(0, max_hours)
    n_at_risk = np.array([(mv["mv_duration_h"] > h).sum() for h in hours])

    # Boluses per hour
    fent_i["hour_bin"] = fent_i["hours_from_mv_start"].astype(int)
    bolus_by_hour = fent_i[fent_i["hour_bin"] < max_hours].groupby("hour_bin").size()
    bolus_counts = np.zeros(max_hours)
    bolus_counts[bolus_by_hour.index] = bolus_by_hour.values

    # Rate per 100 patients per hour (avoid division by zero)
    rate = np.where(n_at_risk > 0, bolus_counts / n_at_risk * 100, 0)

    # Smooth with 4-hour rolling average for readability
    rate_smooth = pd.Series(rate).rolling(4, min_periods=1, center=True).mean().values

    # Only show hours where >= 25% of initial patients remain
    initial_n = n_at_risk[0]
    mask = n_at_risk >= initial_n * 0.25
    hours_show = hours[mask]
    rate_show = rate_smooth[mask]
    n_at_risk_show = n_at_risk[mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: bolus rate per 100 patients per hour with N at risk
    axes[0].plot(hours_show, rate_show, color="teal", linewidth=2)
    axes[0].set_xlabel("Hours from MV Start")
    axes[0].set_ylabel("Boluses per 100 Patients per Hour", color="teal")
    axes[0].tick_params(axis="y", labelcolor="teal")
    axes[0].set_title("Fentanyl Bolus Rate During Mechanical Ventilation")

    ax_risk = axes[0].twinx()
    ax_risk.plot(hours_show, n_at_risk_show, color="gray", alpha=0.4, linestyle="--", linewidth=1)
    ax_risk.set_ylabel("N Patients at Risk", color="gray")
    ax_risk.tick_params(axis="y", labelcolor="gray")

    # Right panel: bolus dose distribution (unchanged)
    valid_doses = fent_i[fent_i["med_dose"].between(1, 500)]
    axes[1].hist(valid_doses["med_dose"], bins=50, color="teal", edgecolor="white")
    axes[1].set_xlabel("Bolus Dose (mcg)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Fentanyl Bolus Dose Distribution")

    fig.tight_layout()
    save_figure(fig, "fig3_fentanyl_bolus_pattern")
    plt.close(fig)


# ──────────────────────────────────────────────
# Step 6: Summary table
# ──────────────────────────────────────────────
def create_summary_table(summaries, cohort):
    """Create Table 2: fentanyl dosing summary."""
    print("\nStep 6: Creating fentanyl summary table...")

    rows = []
    n_total = len(summaries)
    n_cont = (summaries["fentanyl_type"].isin(["continuous", "both"])).sum()
    n_int = (summaries["fentanyl_type"].isin(["intermittent", "both"])).sum()

    rows.append(("Total patients", f"{n_total:,}", ""))
    rows.append(("Continuous fentanyl", f"{n_cont:,}", f"{100*n_cont/n_total:.1f}%"))
    rows.append(("Intermittent fentanyl", f"{n_int:,}", f"{100*n_int/n_total:.1f}%"))

    # Continuous stats
    cont = summaries[summaries["first_dose_mcg_hr"] > 0]
    if len(cont) > 0:
        rows.append(("", "", ""))
        rows.append(("Continuous infusion", "", ""))
        for label, col in [
            ("Starting dose (mcg/hr)", "first_dose_mcg_hr"),
            ("Peak dose (mcg/hr)", "peak_dose_mcg_hr"),
            ("Mean dose (mcg/hr)", "mean_dose_mcg_hr"),
            ("Infusion duration (hours)", "infusion_duration_hours"),
        ]:
            med = cont[col].median()
            q25 = cont[col].quantile(0.25)
            q75 = cont[col].quantile(0.75)
            rows.append((label, f"{med:.0f}", f"[{q25:.0f} - {q75:.0f}]"))

    # Intermittent stats
    bolus = summaries[summaries["n_boluses"] > 0]
    if len(bolus) > 0:
        rows.append(("", "", ""))
        rows.append(("Intermittent boluses", "", ""))
        for label, col in [
            ("Number of boluses per patient", "n_boluses"),
            ("Total bolus dose (mcg)", "total_bolus_dose_mcg"),
            ("Mean single bolus dose (mcg)", "mean_bolus_dose_mcg"),
        ]:
            med = bolus[col].median()
            q25 = bolus[col].quantile(0.25)
            q75 = bolus[col].quantile(0.75)
            rows.append((label, f"{med:.0f}", f"[{q25:.0f} - {q75:.0f}]"))

    summary_df = pd.DataFrame(rows, columns=["Variable", "Median", "IQR"])
    save_table(summary_df, "table2_fentanyl_summary")
    print(summary_df.to_string(index=False))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("02_aim1_fentanyl.py — Fentanyl Dosing Characterization")
    print("=" * 60)

    cohort, mv_windows = load_data()

    # Load fentanyl data
    fent_c = load_fentanyl_continuous(mv_windows)
    fent_i = load_fentanyl_intermittent(mv_windows)

    # Standardize continuous doses
    fent_c = standardize_continuous_doses(fent_c)

    # Hourly trajectory
    hourly_df, hourly_summary = compute_hourly_trajectory(fent_c)
    save_intermediate(hourly_df[["hospitalization_id", "hour_bin", "dose_mcg_hr"]], "fentanyl_hourly")

    # Per-patient summaries
    summaries = compute_patient_summaries(fent_c, fent_i, cohort)

    # Figures
    plot_trajectory(hourly_summary)
    plot_dose_distribution(summaries)
    plot_bolus_pattern(fent_i, mv_windows)

    # Summary table
    create_summary_table(summaries, cohort)

    print("\n" + "=" * 60)
    print("DONE. Aim 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
