"""
09_dose_increase_rass.py
========================
QI Analysis: Are fentanyl dose increases driven by RASS (sedation depth)
rather than NVPS (pain)?

Hypothesis: Nurses may be titrating fentanyl to RASS instead of NVPS.
For each fentanyl rate increase (continuous infusion), we look back
60 minutes for the most recent RASS score and categorize:
  - RASS > 0:   agitated — dose increase justified for sedation
  - RASS -1, 0: light sedation — near target
  - RASS -2,-3: moderate sedation — at typical target
  - RASS -4,-5: deep sedation — already deeply sedated
  - No RASS:    no assessment within 60 min

Requires: 01_build_cohort.py outputs

Outputs:
  - tables/table15_dose_increase_rass_justification.csv
  - tables/table16_dose_increase_rass_by_icu.csv
  - figures/fig20_dose_increase_rass_justification.pdf/.png
  - figures/fig21_dose_increase_rass_by_icu.pdf/.png
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
    save_table,
    save_figure,
    FENTANYL_EXCLUDE_NAMES,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

LOOKBACK_MINUTES = 60
RASS_JUSTIFIED_THRESHOLD = 0  # dose increase justified when RASS > this value


# ──────────────────────────────────────────────
# Step 1: Load and standardize fentanyl continuous data
# ──────────────────────────────────────────────
def load_fentanyl_continuous():
    """Load continuous fentanyl during MV, standardize to mcg/hr."""
    print("Step 1: Loading continuous fentanyl data...")
    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])
    mv = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()

    fent_c = load_clif_table(
        "medication_admin_continuous",
        filters=[("med_category", "==", "fentanyl")],
    )
    fent_c["admin_dttm"] = pd.to_datetime(fent_c["admin_dttm"])

    # Exclude sufentanil/remifentanil
    mask = ~fent_c["med_name"].str.upper().str.contains(
        "|".join(FENTANYL_EXCLUDE_NAMES), na=False
    )
    fent_c = fent_c[mask]

    # Restrict to MV windows
    fent_c = fent_c.merge(mv, on="hospitalization_id")
    fent_c = fent_c[
        (fent_c["admin_dttm"] >= fent_c["mv_start"])
        & (fent_c["admin_dttm"] <= fent_c["mv_end"])
    ]

    # Standardize doses to mcg/hr
    needs_weight = fent_c["med_dose_unit"] == "mcg/kg/hr"
    if needs_weight.any():
        weight_ids = fent_c.loc[needs_weight, "hospitalization_id"].unique()
        vitals = load_clif_table(
            "vitals",
            columns=["hospitalization_id", "vital_category", "vital_value"],
            filters=[("vital_category", "==", "weight_kg")],
        )
        vitals = vitals[vitals["hospitalization_id"].isin(weight_ids)]
        vitals["vital_value"] = pd.to_numeric(vitals["vital_value"], errors="coerce")
        weight_by_patient = (
            vitals.groupby("hospitalization_id")["vital_value"]
            .median()
            .reset_index()
            .rename(columns={"vital_value": "weight_kg"})
        )
        fent_c = fent_c.merge(weight_by_patient, on="hospitalization_id", how="left")
        needs_weight = fent_c["med_dose_unit"] == "mcg/kg/hr"
        fent_c["weight_kg"] = fent_c["weight_kg"].fillna(80.0)
        fent_c["dose_mcg_hr"] = np.where(
            needs_weight,
            fent_c["med_dose"] * fent_c["weight_kg"],
            fent_c["med_dose"],
        )
    else:
        fent_c["dose_mcg_hr"] = fent_c["med_dose"]

    # Handle stops as dose = 0
    fent_c.loc[fent_c["mar_action_category"] == "stop", "dose_mcg_hr"] = 0

    # Remove missing/outlier doses
    fent_c = fent_c[fent_c["dose_mcg_hr"].notna()].copy()
    fent_c = fent_c[(fent_c["dose_mcg_hr"] >= 0) & (fent_c["dose_mcg_hr"] <= 500)].copy()

    print(f"  Continuous fentanyl records during MV: {len(fent_c):,}")
    print(f"  Patients: {fent_c['hospitalization_id'].nunique():,}")

    return fent_c, mv


# ──────────────────────────────────────────────
# Step 2: Identify dose increases
# ──────────────────────────────────────────────
def identify_dose_increases(fent_c):
    """Find all instances where fentanyl rate increased from the previous record."""
    print("\nStep 2: Identifying dose increases...")

    fent_c = fent_c.sort_values(["hospitalization_id", "admin_dttm"]).copy()
    fent_c["prev_dose"] = fent_c.groupby("hospitalization_id")["dose_mcg_hr"].shift(1)

    # Dose increase = current > previous, and previous was not missing
    increases = fent_c[
        (fent_c["prev_dose"].notna())
        & (fent_c["dose_mcg_hr"] > fent_c["prev_dose"])
        & (fent_c["prev_dose"] > 0)  # Exclude restarts from 0
    ].copy()

    increases["dose_change_mcg_hr"] = increases["dose_mcg_hr"] - increases["prev_dose"]

    n_increases = len(increases)
    n_patients = increases["hospitalization_id"].nunique()
    print(f"  Total dose increases: {n_increases:,}")
    print(f"  Patients with at least one increase: {n_patients:,}")
    print(f"  Median increase magnitude: {increases['dose_change_mcg_hr'].median():.0f} mcg/hr")
    print(f"  IQR: [{increases['dose_change_mcg_hr'].quantile(0.25):.0f} - "
          f"{increases['dose_change_mcg_hr'].quantile(0.75):.0f}] mcg/hr")

    return increases


# ──────────────────────────────────────────────
# Step 3: Match dose increases to preceding RASS
# ──────────────────────────────────────────────
def match_rass_to_increases(increases, mv):
    """For each dose increase, find most recent RASS within lookback window."""
    print(f"\nStep 3: Matching dose increases to RASS within {LOOKBACK_MINUTES} min lookback...")

    # Load RASS during MV
    rass = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "RASS")],
    )
    rass["recorded_dttm"] = pd.to_datetime(rass["recorded_dttm"])
    rass["rass_value"] = pd.to_numeric(rass["numerical_value"], errors="coerce")
    rass = rass.dropna(subset=["rass_value"])

    # Restrict to MV windows
    rass = rass.merge(mv, on="hospitalization_id")
    rass = rass[
        (rass["recorded_dttm"] >= rass["mv_start"])
        & (rass["recorded_dttm"] <= rass["mv_end"])
    ]
    rass = rass[["hospitalization_id", "recorded_dttm", "rass_value"]].copy()
    rass = rass.sort_values(["hospitalization_id", "recorded_dttm"])
    print(f"  RASS records during MV: {len(rass):,}")

    # For each dose increase, find most recent RASS within lookback
    rass_scores = []

    for hid, inc_grp in increases.groupby("hospitalization_id"):
        pt_rass = rass[rass["hospitalization_id"] == hid]
        if len(pt_rass) == 0:
            rass_scores.extend([np.nan] * len(inc_grp))
            continue

        rass_times = pt_rass["recorded_dttm"].values
        rass_vals = pt_rass["rass_value"].values

        for _, row in inc_grp.iterrows():
            inc_time = row["admin_dttm"]
            diffs = (rass_times - np.datetime64(inc_time)) / np.timedelta64(1, "m")
            in_window = (diffs >= -LOOKBACK_MINUTES) & (diffs <= 0)

            if np.any(in_window):
                window_diffs = diffs[in_window]
                window_vals = rass_vals[in_window]
                most_recent_idx = np.argmax(window_diffs)  # closest to 0
                rass_scores.append(window_vals[most_recent_idx])
            else:
                rass_scores.append(np.nan)

    increases = increases.copy()
    increases["rass_score"] = rass_scores

    # Categorize using standard RASS buckets
    def categorize(score):
        if np.isnan(score):
            return f"No RASS within {LOOKBACK_MINUTES} min"
        elif score > RASS_JUSTIFIED_THRESHOLD:
            return "Agitated (RASS > 0) — justified"
        elif score >= -1:
            return "Light sedation (RASS -1, 0)"
        elif score >= -3:
            return "Moderate sedation (RASS -2, -3)"
        else:
            return "Deep sedation (RASS -4, -5)"

    increases["justification"] = increases["rass_score"].apply(categorize)

    # Print summary
    print(f"\n  RASS justification breakdown:")
    print(f"  {'─' * 55}")
    for cat, n in increases["justification"].value_counts().items():
        pct = 100 * n / len(increases)
        print(f"  {cat}: {n:,} ({pct:.1f}%)")
    print(f"  {'─' * 55}")

    return increases


# ──────────────────────────────────────────────
# Step 4: Summary table and figure
# ──────────────────────────────────────────────
CAT_ORDER = [
    "Agitated (RASS > 0) — justified",
    "Light sedation (RASS -1, 0)",
    "Moderate sedation (RASS -2, -3)",
    "Deep sedation (RASS -4, -5)",
    f"No RASS within {LOOKBACK_MINUTES} min",
]

CAT_COLORS = {
    "Agitated (RASS > 0) — justified": "#2e7d32",
    "Light sedation (RASS -1, 0)": "#8bc34a",
    "Moderate sedation (RASS -2, -3)": "#f57c00",
    "Deep sedation (RASS -4, -5)": "#d32f2f",
    f"No RASS within {LOOKBACK_MINUTES} min": "#78909c",
}


def create_summary(increases):
    """Create overall RASS justification summary table and figure."""
    print("\nStep 4: Creating summary table and figure...")

    rows = []
    for cat in CAT_ORDER:
        n = (increases["justification"] == cat).sum()
        pct = 100 * n / len(increases)
        rows.append({"Category": cat, "N Dose Increases": n, "%": f"{pct:.1f}%"})

    rows.append({"Category": "Total", "N Dose Increases": len(increases), "%": "100%"})
    table = pd.DataFrame(rows)
    save_table(table, "table15_dose_increase_rass_justification")
    print(table.to_string(index=False))

    # Figure: horizontal bar
    plot_data = table[table["Category"] != "Total"].copy()
    counts = [plot_data.iloc[i]["N Dose Increases"] for i in range(len(plot_data))]
    pcts = [100 * c / len(increases) for c in counts]
    labels = plot_data["Category"].tolist()
    colors = [CAT_COLORS[cat] for cat in labels]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels[::-1], pcts[::-1], color=colors[::-1], edgecolor="white", height=0.6)

    for bar, pct, count in zip(bars, pcts[::-1], counts[::-1]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% (n={count:,})",
            va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("% of Fentanyl Dose Increases")
    ax.set_title(
        f"RASS at Time of Fentanyl Dose Increase\n"
        f"({LOOKBACK_MINUTES}-min lookback, 'justified' = RASS > {RASS_JUSTIFIED_THRESHOLD})"
    )
    ax.set_xlim(0, max(pcts) * 1.3)
    fig.tight_layout()
    save_figure(fig, "fig20_dose_increase_rass_justification")
    plt.close(fig)


# ──────────────────────────────────────────────
# Step 5: Stratify by ICU type
# ──────────────────────────────────────────────
def stratify_by_icu_type(increases):
    """Break down RASS justification by ICU type."""
    print("\nStep 5: Stratifying by ICU type...")

    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])

    # Assign ICU type via ADT (same logic as scripts 06/08)
    adt = load_clif_table(
        "adt",
        columns=["hospitalization_id", "in_dttm", "out_dttm",
                 "location_category", "location_type"],
    )
    adt["in_dttm"] = pd.to_datetime(adt["in_dttm"])
    adt["out_dttm"] = pd.to_datetime(adt["out_dttm"])
    adt_icu = adt[adt["location_category"] == "icu"].copy()

    mv = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()
    merged = adt_icu.merge(mv, on="hospitalization_id")
    merged["overlap_start"] = merged[["in_dttm", "mv_start"]].max(axis=1)
    merged["overlap_end"] = merged[["out_dttm", "mv_end"]].min(axis=1)
    merged = merged[merged["overlap_start"] < merged["overlap_end"]].copy()
    merged["overlap_hours"] = (
        merged["overlap_end"] - merged["overlap_start"]
    ).dt.total_seconds() / 3600

    dominant = (
        merged.sort_values("overlap_hours", ascending=False)
        .groupby("hospitalization_id")
        .first()
        .reset_index()[["hospitalization_id", "location_type"]]
    )
    dominant = dominant.rename(columns={"location_type": "icu_type"})

    # Merge ICU type onto increases
    inc = increases.merge(dominant, on="hospitalization_id", how="left")
    inc["icu_type"] = inc["icu_type"].fillna("unknown")
    inc = inc[inc["icu_type"] != "unknown"]

    rows = []
    for icu_type, grp in inc.groupby("icu_type"):
        if len(grp) < 20:
            continue
        n_total = len(grp)
        row = {"ICU Type": icu_type, "N Increases": n_total}
        for cat in CAT_ORDER:
            n = (grp["justification"] == cat).sum()
            pct = 100 * n / n_total
            short_label = cat.split("—")[0].strip() if "—" in cat else cat
            row[short_label] = f"{n:,} ({pct:.1f}%)"
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("N Increases", ascending=False)
    save_table(table, "table16_dose_increase_rass_by_icu")
    print(table.to_string(index=False))

    # Figure: stacked horizontal bar by ICU type
    icu_types_sorted = []
    for icu_type, grp in inc.groupby("icu_type"):
        if len(grp) < 20:
            continue
        n_justified = (grp["justification"] == CAT_ORDER[0]).sum()
        icu_types_sorted.append({
            "icu_type": icu_type,
            "n": len(grp),
            "pct_justified": 100 * n_justified / len(grp),
        })

    icu_df = pd.DataFrame(icu_types_sorted).sort_values("pct_justified", ascending=True)

    if len(icu_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(icu_df))
        left = np.zeros(len(icu_df))

        for cat in CAT_ORDER:
            vals = []
            for _, row in icu_df.iterrows():
                grp = inc[inc["icu_type"] == row["icu_type"]]
                pct = 100 * (grp["justification"] == cat).sum() / len(grp)
                vals.append(pct)
            bars = ax.barh(
                y_pos, vals, left=left, color=CAT_COLORS[cat],
                label=cat, edgecolor="white", height=0.6,
            )
            left += np.array(vals)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(icu_df["icu_type"].tolist())
        ax.set_xlabel("% of Dose Increases")
        ax.set_title("RASS at Time of Fentanyl Dose Increase by ICU Type")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(0, 105)

        # Add n labels
        for i, (_, row) in enumerate(icu_df.iterrows()):
            ax.text(102, i, f"n={row['n']:,}", va="center", fontsize=9, color="gray")

        fig.tight_layout()
        save_figure(fig, "fig21_dose_increase_rass_by_icu")
        plt.close(fig)


# ──────────────────────────────────────────────
# Step 6: % Justified over time (by year)
# ──────────────────────────────────────────────
def plot_justification_by_year(increases):
    """Plot % RASS-justified dose increases by admission year."""
    print("\nStep 6: Plotting justification trend by year...")

    cohort = load_intermediate("cohort")
    cohort["admission_dttm"] = pd.to_datetime(cohort["admission_dttm"])
    cohort["admission_year"] = cohort["admission_dttm"].dt.year

    inc = increases.merge(
        cohort[["hospitalization_id", "admission_year"]],
        on="hospitalization_id",
        how="left",
    )
    inc = inc.dropna(subset=["admission_year"])
    inc["admission_year"] = inc["admission_year"].astype(int)

    justified_label = "Agitated (RASS > 0) — justified"

    rows = []
    for year, grp in inc.groupby("admission_year"):
        n_total = len(grp)
        n_justified = (grp["justification"] == justified_label).sum()
        pct = 100 * n_justified / n_total
        rows.append({"Year": year, "N Increases": n_total, "N Justified": n_justified, "% Justified": pct})

    yr_df = pd.DataFrame(rows)
    print(yr_df.to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar: N dose increases (background)
    ax2 = ax1.twinx()
    ax2.bar(yr_df["Year"], yr_df["N Increases"], color="#e0e0e0", width=0.6, zorder=1, label="N increases")
    ax2.set_ylabel("N Dose Increases", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Line: % justified
    ax1.plot(yr_df["Year"], yr_df["% Justified"], "o-", color="#2e7d32", linewidth=2.5,
             markersize=8, zorder=3, label="% Agitated (RASS > 0)")
    for _, row in yr_df.iterrows():
        ax1.annotate(f"{row['% Justified']:.1f}%",
                     (row["Year"], row["% Justified"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=10, fontweight="bold", color="#2e7d32")

    ax1.set_xlabel("Admission Year")
    ax1.set_ylabel("% of Dose Increases Justified")
    ax1.set_title(f"RASS-Justified Fentanyl Dose Increases Over Time\n(RASS > {RASS_JUSTIFIED_THRESHOLD} within {LOOKBACK_MINUTES} min)")
    ax1.set_ylim(0, max(yr_df["% Justified"]) * 1.5)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.set_xticks(yr_df["Year"])

    fig.tight_layout()
    save_figure(fig, "fig23_rass_justification_by_year")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("09_dose_increase_rass.py")
    print("Are fentanyl dose increases driven by RASS (sedation depth)?")
    print(f"'Justified' = RASS > {RASS_JUSTIFIED_THRESHOLD} (agitated)")
    print(f"Lookback window: {LOOKBACK_MINUTES} minutes")
    print("=" * 60)

    fent_c, mv = load_fentanyl_continuous()
    increases = identify_dose_increases(fent_c)
    increases = match_rass_to_increases(increases, mv)
    create_summary(increases)
    stratify_by_icu_type(increases)
    plot_justification_by_year(increases)

    print("\n" + "=" * 60)
    print("DONE. RASS-based dose increase justification analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
