"""
06_unit_level_nvps.py
=====================
Unit-level NVPS analyses: stratify NVPS documentation and scoring
patterns by ICU type using ADT data.

Requires: 01_build_cohort.py, 03_aim2_nvps.py outputs

Analyses:
  1. NVPS scoring patterns by ICU type (non-zero rate, score distribution)
  2. Q4H documentation compliance by ICU type
  3. NVPS-RASS concordance by ICU type (NVPS=0 in lightly sedated patients)

Outputs:
  - tables/table10_nvps_by_icu_type.csv
  - tables/table11_compliance_by_icu_type.csv
  - tables/table12_nvps_rass_by_icu_type.csv
  - figures/fig14_nonzero_rate_by_icu_type.pdf/.png
  - figures/fig15_compliance_by_icu_type.pdf/.png
  - figures/fig16_nvps_rass_by_icu_type.pdf/.png
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
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ──────────────────────────────────────────────
# Step 1: Link patients to ICU type via ADT
# ──────────────────────────────────────────────
def assign_icu_type(cohort):
    """Assign each patient's dominant ICU type during MV using ADT data."""
    print("Step 1: Assigning ICU type from ADT data...")
    adt = load_clif_table(
        "adt",
        columns=["hospitalization_id", "in_dttm", "out_dttm",
                 "location_category", "location_type"],
    )
    adt["in_dttm"] = pd.to_datetime(adt["in_dttm"])
    adt["out_dttm"] = pd.to_datetime(adt["out_dttm"])

    # Keep only ICU locations
    adt_icu = adt[adt["location_category"] == "icu"].copy()
    print(f"  ICU ADT records: {len(adt_icu):,}")

    # Merge with cohort MV windows
    mv = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()
    merged = adt_icu.merge(mv, on="hospitalization_id")

    # Find overlapping ICU stays during MV
    merged["overlap_start"] = merged[["in_dttm", "mv_start"]].max(axis=1)
    merged["overlap_end"] = merged[["out_dttm", "mv_end"]].min(axis=1)
    merged = merged[merged["overlap_start"] < merged["overlap_end"]].copy()
    merged["overlap_hours"] = (
        merged["overlap_end"] - merged["overlap_start"]
    ).dt.total_seconds() / 3600

    print(f"  Patients with ICU overlap during MV: {merged['hospitalization_id'].nunique():,}")

    # Assign dominant ICU type (longest overlap)
    dominant = (
        merged.sort_values("overlap_hours", ascending=False)
        .groupby("hospitalization_id")
        .first()
        .reset_index()[["hospitalization_id", "location_type"]]
    )
    dominant = dominant.rename(columns={"location_type": "icu_type"})

    cohort = cohort.merge(dominant, on="hospitalization_id", how="left")
    cohort["icu_type"] = cohort["icu_type"].fillna("unknown")

    print(f"  ICU type distribution:")
    for icu, n in cohort["icu_type"].value_counts().items():
        print(f"    {icu}: {n:,} ({100*n/len(cohort):.1f}%)")

    return cohort


# ──────────────────────────────────────────────
# Step 2: Load NVPS during MV with ICU type
# ──────────────────────────────────────────────
def load_nvps_with_icu_type(cohort):
    """Load NVPS TOTAL scores during MV, tagged with ICU type."""
    print("\nStep 2: Loading NVPS data with ICU type...")
    mv = cohort[["hospitalization_id", "mv_start", "mv_end", "icu_type",
                 "mv_duration_hours"]].copy()

    nvps = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "NVPS")],
    )
    nvps["recorded_dttm"] = pd.to_datetime(nvps["recorded_dttm"])
    nvps = nvps[nvps["assessment_name"].str.contains("TOTAL", case=False, na=False)]

    nvps = nvps.merge(mv, on="hospitalization_id")
    nvps = nvps[
        (nvps["recorded_dttm"] >= nvps["mv_start"])
        & (nvps["recorded_dttm"] <= nvps["mv_end"])
    ]
    nvps["numerical_value"] = pd.to_numeric(nvps["numerical_value"], errors="coerce")
    nvps["hours_from_mv_start"] = (
        nvps["recorded_dttm"] - nvps["mv_start"]
    ).dt.total_seconds() / 3600

    print(f"  NVPS records with ICU type: {len(nvps):,}")
    return nvps, mv


# ──────────────────────────────────────────────
# Analysis 1: NVPS scoring patterns by ICU type
# ──────────────────────────────────────────────
def analysis1_scoring_by_icu_type(nvps):
    """Compare non-zero NVPS rates and score distributions across ICU types."""
    print("\n" + "=" * 60)
    print("Analysis 1: NVPS scoring patterns by ICU type")
    print("=" * 60)

    # Exclude unknown and types with very few assessments
    nvps_known = nvps[nvps["icu_type"] != "unknown"].copy()
    scores = nvps_known.dropna(subset=["numerical_value"])

    rows = []
    for icu_type, grp in scores.groupby("icu_type"):
        n_total = len(grp)
        if n_total < 50:
            continue
        n_patients = grp["hospitalization_id"].nunique()
        n_zero = (grp["numerical_value"] == 0).sum()
        n_nonzero = n_total - n_zero
        pct_nonzero = 100 * n_nonzero / n_total

        rows.append({
            "ICU Type": icu_type,
            "N Patients": n_patients,
            "N Assessments": n_total,
            "N Score=0": n_zero,
            "N Score>0": n_nonzero,
            "% Non-Zero": round(pct_nonzero, 1),
        })

    table = pd.DataFrame(rows).sort_values("N Patients", ascending=False)
    save_table(table, "table10_nvps_by_icu_type")
    print(table.to_string(index=False))

    # Figure: non-zero rate by ICU type
    if len(table) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        table_sorted = table.sort_values("% Non-Zero", ascending=True)
        bars = ax.barh(
            table_sorted["ICU Type"], table_sorted["% Non-Zero"],
            color="coral", edgecolor="white",
        )

        # Add patient count labels
        for bar, (_, row) in zip(bars, table_sorted.iterrows()):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"n={row['N Patients']:,} ({row['N Assessments']:,} scores)",
                va="center", fontsize=9, color="gray",
            )

        # Overall reference line
        overall_nonzero = 100 * (scores["numerical_value"] > 0).sum() / len(scores)
        ax.axvline(x=overall_nonzero, color="red", linestyle="--", linewidth=1.5,
                   label=f"Overall: {overall_nonzero:.1f}%")

        ax.set_xlabel("% Non-Zero NVPS Scores")
        ax.set_title("Non-Zero NVPS Rate by ICU Type")
        ax.legend(loc="lower right")
        ax.set_xlim(0, max(table["% Non-Zero"].max() * 1.5, 15))
        fig.tight_layout()
        save_figure(fig, "fig14_nonzero_rate_by_icu_type")
        plt.close(fig)

    return table


# ──────────────────────────────────────────────
# Analysis 2: Documentation compliance by ICU type
# ──────────────────────────────────────────────
def analysis2_compliance_by_icu_type(nvps, mv):
    """Compare Q4H documentation compliance across ICU types."""
    print("\n" + "=" * 60)
    print("Analysis 2: Documentation compliance by ICU type")
    print("=" * 60)

    mv_known = mv[mv["icu_type"] != "unknown"].copy()

    compliance_list = []
    for _, row in mv_known.iterrows():
        hid = row["hospitalization_id"]
        mv_dur = row["mv_duration_hours"]
        icu_type = row["icu_type"]

        pt_nvps = nvps[nvps["hospitalization_id"] == hid]
        n_assessments = len(pt_nvps)
        n_blocks = max(int(np.ceil(mv_dur / 4)), 1)

        if n_assessments > 0:
            block_indices = (pt_nvps["hours_from_mv_start"] // 4).astype(int)
            # Only count blocks within the valid range [0, n_blocks-1]
            valid_blocks = block_indices[(block_indices >= 0) & (block_indices < n_blocks)]
            n_compliant = valid_blocks.nunique()
            compliance = n_compliant / n_blocks
            assessments_per_day = n_assessments / max(mv_dur / 24, 1 / 24)
        else:
            compliance = 0.0
            assessments_per_day = 0.0

        compliance_list.append({
            "hospitalization_id": hid,
            "icu_type": icu_type,
            "compliance_rate": compliance,
            "assessments_per_day": assessments_per_day,
            "has_nvps": n_assessments > 0,
        })

    comp = pd.DataFrame(compliance_list)

    # Summary by ICU type
    rows = []
    for icu_type, grp in comp.groupby("icu_type"):
        n = len(grp)
        if n < 20:
            continue
        has = grp[grp["has_nvps"]]
        rows.append({
            "ICU Type": icu_type,
            "N Patients": n,
            "% With Any NVPS": round(100 * len(has) / n, 1),
            "Median Compliance": round(has["compliance_rate"].median(), 2) if len(has) > 0 else 0,
            "IQR Compliance": (f"[{has['compliance_rate'].quantile(0.25):.2f}-"
                               f"{has['compliance_rate'].quantile(0.75):.2f}]") if len(has) > 0 else "",
            "Median Assessments/Day": round(has["assessments_per_day"].median(), 1) if len(has) > 0 else 0,
        })

    table = pd.DataFrame(rows).sort_values("N Patients", ascending=False)
    save_table(table, "table11_compliance_by_icu_type")
    print(table.to_string(index=False))

    # Figure: compliance boxplot by ICU type
    has_nvps = comp[comp["has_nvps"] & (comp["icu_type"] != "unknown")]
    icu_types_ordered = (
        has_nvps.groupby("icu_type")["compliance_rate"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    # Only plot types with enough patients
    icu_counts = has_nvps["icu_type"].value_counts()
    icu_types_ordered = [t for t in icu_types_ordered if icu_counts.get(t, 0) >= 20]

    if len(icu_types_ordered) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = has_nvps[has_nvps["icu_type"].isin(icu_types_ordered)]
        sns.boxplot(
            data=plot_data, y="icu_type", x="compliance_rate",
            order=icu_types_ordered, color="steelblue", ax=ax,
        )
        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="100% compliance")
        ax.set_xlabel("Q4H Block Compliance Rate")
        ax.set_ylabel("")
        ax.set_title("NVPS Documentation Compliance by ICU Type")

        # Add n labels
        for i, icu in enumerate(icu_types_ordered):
            n = icu_counts[icu]
            ax.text(1.02, i, f"n={n:,}", va="center", fontsize=9, color="gray")

        ax.legend(loc="lower left")
        fig.tight_layout()
        save_figure(fig, "fig15_compliance_by_icu_type")
        plt.close(fig)

    return comp


# ──────────────────────────────────────────────
# Analysis 3: NVPS-RASS concordance by ICU type
# ──────────────────────────────────────────────
def analysis3_nvps_rass_by_icu_type(nvps, cohort):
    """NVPS=0 rate among lightly sedated patients, stratified by ICU type."""
    print("\n" + "=" * 60)
    print("Analysis 3: NVPS-RASS concordance by ICU type")
    print("=" * 60)

    mv = cohort[["hospitalization_id", "mv_start", "mv_end", "icu_type"]].copy()

    # Load RASS during MV
    rass = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "RASS")],
    )
    rass["recorded_dttm"] = pd.to_datetime(rass["recorded_dttm"])
    rass["rass_value"] = pd.to_numeric(rass["numerical_value"], errors="coerce")
    rass = rass.dropna(subset=["rass_value"])
    rass = rass.merge(mv, on="hospitalization_id")
    rass = rass[
        (rass["recorded_dttm"] >= rass["mv_start"])
        & (rass["recorded_dttm"] <= rass["mv_end"])
    ]
    print(f"  RASS during MV: {len(rass):,}")

    # Pair NVPS with nearest RASS within +/- 2 hours
    print("  Pairing NVPS with nearest RASS (+-2h)... this may take several minutes...")

    pairs = []
    for hid, nvps_grp in nvps.groupby("hospitalization_id"):
        pt_rass = rass[rass["hospitalization_id"] == hid]
        if len(pt_rass) == 0:
            continue

        icu_type = nvps_grp["icu_type"].iloc[0]
        rass_times = pt_rass["recorded_dttm"].values
        rass_vals = pt_rass["rass_value"].values

        for _, row in nvps_grp.iterrows():
            nvps_time = row["recorded_dttm"]
            diffs_hr = (rass_times - np.datetime64(nvps_time)) / np.timedelta64(1, "h")
            nearest_idx = np.argmin(np.abs(diffs_hr))

            if abs(diffs_hr[nearest_idx]) <= 2:
                pairs.append({
                    "icu_type": icu_type,
                    "nvps_score": row["numerical_value"],
                    "rass_value": rass_vals[nearest_idx],
                })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Paired NVPS-RASS: {len(pairs_df):,}")

    # Categorize RASS
    def rass_cat(val):
        if val <= -4:
            return "Deep (-5,-4)"
        elif val <= -2:
            return "Moderate (-3,-2)"
        elif val <= 0:
            return "Light (-1,0)"
        else:
            return "Agitated (+1 to +3)"

    pairs_df["rass_category"] = pairs_df["rass_value"].apply(rass_cat)

    # Focus on lightly sedated patients (RASS -1 to 0) — this is where
    # NVPS=0 is most clinically implausible
    light = pairs_df[pairs_df["rass_category"] == "Light (-1,0)"]
    print(f"\n  Lightly sedated (RASS -1 to 0): {len(light):,} paired assessments")

    rows = []
    for icu_type, grp in light.groupby("icu_type"):
        if icu_type == "unknown" or len(grp) < 20:
            continue
        n = len(grp)
        n_zero = (grp["nvps_score"] == 0).sum()
        pct_zero = 100 * n_zero / n
        rows.append({
            "ICU Type": icu_type,
            "N Paired (Light Sedation)": n,
            "NVPS=0": n_zero,
            "NVPS>0": n - n_zero,
            "% NVPS=0": round(pct_zero, 1),
        })

    table = pd.DataFrame(rows).sort_values("N Paired (Light Sedation)", ascending=False)
    save_table(table, "table12_nvps_rass_by_icu_type")
    print("\n  NVPS=0 rate in lightly sedated patients by ICU type:")
    print(table.to_string(index=False))

    # Figure: grouped bar — % NVPS=0 by ICU type and RASS category
    rass_order = ["Deep (-5,-4)", "Moderate (-3,-2)", "Light (-1,0)", "Agitated (+1 to +3)"]
    icu_types = pairs_df[pairs_df["icu_type"] != "unknown"]["icu_type"].value_counts()
    icu_types = icu_types[icu_types >= 50].index.tolist()

    if len(icu_types) > 0:
        plot_data = pairs_df[pairs_df["icu_type"].isin(icu_types)].copy()
        plot_data["nvps_zero"] = (plot_data["nvps_score"] == 0).astype(int)

        summary = (
            plot_data.groupby(["icu_type", "rass_category"])
            .agg(pct_zero=("nvps_zero", "mean"), n=("nvps_zero", "count"))
            .reset_index()
        )
        summary["pct_zero"] = summary["pct_zero"] * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(icu_types))
        width = 0.2
        colors = {"Deep (-5,-4)": "#2c3e50", "Moderate (-3,-2)": "#4a90d9",
                  "Light (-1,0)": "#f39c12", "Agitated (+1 to +3)": "#e74c3c"}

        for i, rcat in enumerate(rass_order):
            vals = []
            for icu in icu_types:
                row = summary[(summary["icu_type"] == icu) & (summary["rass_category"] == rcat)]
                vals.append(row["pct_zero"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=rcat, color=colors[rcat], edgecolor="white")

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(icu_types, rotation=45, ha="right")
        ax.set_ylabel("% NVPS = 0")
        ax.set_title("NVPS = 0 Rate by ICU Type and Sedation Depth")
        ax.legend(title="RASS Category", loc="upper right")
        ax.set_ylim(0, 105)
        fig.tight_layout()
        save_figure(fig, "fig16_nvps_rass_by_icu_type")
        plt.close(fig)

    return pairs_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("06_unit_level_nvps.py — NVPS Patterns by ICU Type")
    print("=" * 60)

    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])

    # Assign ICU type
    cohort = assign_icu_type(cohort)

    # Load NVPS with ICU type
    nvps, mv = load_nvps_with_icu_type(cohort)

    # Analysis 1: Scoring by ICU type
    analysis1_scoring_by_icu_type(nvps)

    # Analysis 2: Compliance by ICU type
    analysis2_compliance_by_icu_type(nvps, mv)

    # Analysis 3: NVPS-RASS concordance by ICU type
    analysis3_nvps_rass_by_icu_type(nvps, cohort)

    print("\n" + "=" * 60)
    print("DONE. Unit-level NVPS analyses complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
