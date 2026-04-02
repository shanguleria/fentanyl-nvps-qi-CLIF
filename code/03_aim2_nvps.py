"""
03_aim2_nvps.py
===============
Aim 2: Characterize NVPS documentation patterns during mechanical ventilation.

Requires: 01_build_cohort.py outputs

Outputs:
  - tables/table3_nvps_documentation.csv
  - figures/fig4_nvps_gap_distribution.pdf/.png
  - figures/fig5_nvps_by_mv_hour.pdf/.png
  - figures/fig6_nvps_by_time_of_day.pdf/.png
  - figures/fig7_nvps_score_distribution.pdf/.png
  - intermediate/nvps_metrics.parquet
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


# ──────────────────────────────────────────────
# Step 1: Load NVPS data restricted to MV windows
# ──────────────────────────────────────────────
def load_nvps_during_mv():
    """Load NVPS TOTAL scores during MV episodes."""
    print("Step 1: Loading NVPS data during MV...")
    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])
    mv_windows = cohort[["hospitalization_id", "mv_start", "mv_end", "mv_duration_hours"]].copy()

    # Load NVPS TOTAL scores (0-10 scale)
    nvps = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "NVPS")],
    )
    nvps["recorded_dttm"] = pd.to_datetime(nvps["recorded_dttm"])
    nvps = nvps[nvps["assessment_name"].str.contains("TOTAL", case=False, na=False)]

    # Restrict to MV windows
    nvps = nvps.merge(mv_windows, on="hospitalization_id")
    nvps = nvps[
        (nvps["recorded_dttm"] >= nvps["mv_start"])
        & (nvps["recorded_dttm"] <= nvps["mv_end"])
    ]
    nvps["hours_from_mv_start"] = (
        nvps["recorded_dttm"] - nvps["mv_start"]
    ).dt.total_seconds() / 3600
    nvps["numerical_value"] = pd.to_numeric(nvps["numerical_value"], errors="coerce")

    n_patients = nvps["hospitalization_id"].nunique()
    n_total = len(mv_windows)
    print(f"  NVPS records during MV: {len(nvps):,}")
    print(f"  Patients with NVPS during MV: {n_patients:,} / {n_total:,} ({100*n_patients/n_total:.1f}%)")

    return nvps, mv_windows


# ──────────────────────────────────────────────
# Step 2: Per-episode documentation metrics
# ──────────────────────────────────────────────
def compute_documentation_metrics(nvps, mv_windows):
    """Compute NVPS documentation frequency and regularity per MV episode."""
    print("\nStep 2: Computing per-episode documentation metrics...")

    metrics_list = []

    for _, row in mv_windows.iterrows():
        hid = row["hospitalization_id"]
        mv_dur = row["mv_duration_hours"]
        pt_nvps = nvps[nvps["hospitalization_id"] == hid].sort_values("recorded_dttm")

        n_assessments = len(pt_nvps)
        assessments_per_day = n_assessments / max(mv_dur / 24, 1 / 24)

        # Inter-assessment gaps
        if n_assessments >= 2:
            gaps = pt_nvps["recorded_dttm"].diff().dt.total_seconds() / 3600
            gaps = gaps.dropna()
            median_gap = gaps.median()
            max_gap = gaps.max()
            cv_gap = gaps.std() / gaps.mean() if gaps.mean() > 0 else np.nan
        else:
            median_gap = np.nan
            max_gap = np.nan
            cv_gap = np.nan

        # Time to first NVPS
        if n_assessments > 0:
            time_to_first = pt_nvps["hours_from_mv_start"].min()
        else:
            time_to_first = np.nan

        # 4-hour block compliance
        n_blocks = max(int(np.ceil(mv_dur / 4)), 1)
        if n_assessments > 0:
            block_indices = (pt_nvps["hours_from_mv_start"] // 4).astype(int)
            valid_blocks = block_indices[(block_indices >= 0) & (block_indices < n_blocks)]
            n_compliant_blocks = valid_blocks.nunique()
            compliance_rate = n_compliant_blocks / n_blocks
        else:
            n_compliant_blocks = 0
            compliance_rate = 0.0

        # 72h landmark metrics (for Aim 3)
        if mv_dur >= 72:
            nvps_72h = pt_nvps[pt_nvps["hours_from_mv_start"] <= 72]
            n_blocks_72h = 18  # 72 / 4
            if len(nvps_72h) > 0:
                blocks_72h = (nvps_72h["hours_from_mv_start"] // 4).astype(int)
                valid_72h = blocks_72h[(blocks_72h >= 0) & (blocks_72h < n_blocks_72h)]
                compliance_72h = valid_72h.nunique() / n_blocks_72h
                assessments_per_day_72h = len(nvps_72h) / 3
            else:
                compliance_72h = 0.0
                assessments_per_day_72h = 0.0
        else:
            compliance_72h = np.nan
            assessments_per_day_72h = np.nan

        metrics_list.append({
            "hospitalization_id": hid,
            "n_nvps": n_assessments,
            "assessments_per_day": assessments_per_day,
            "median_gap_hours": median_gap,
            "max_gap_hours": max_gap,
            "cv_gap": cv_gap,
            "time_to_first_nvps_hours": time_to_first,
            "n_4h_blocks": n_blocks,
            "n_compliant_blocks": n_compliant_blocks,
            "compliance_rate": compliance_rate,
            "compliance_72h": compliance_72h,
            "assessments_per_day_72h": assessments_per_day_72h,
            "has_nvps": n_assessments > 0,
        })

    metrics = pd.DataFrame(metrics_list)

    print(f"  Patients with any NVPS: {metrics['has_nvps'].sum():,} / {len(metrics):,}")
    print(f"  Median assessments/day: {metrics.loc[metrics['has_nvps'], 'assessments_per_day'].median():.1f}")
    print(f"  Median 4h-block compliance: {metrics.loc[metrics['has_nvps'], 'compliance_rate'].median():.2f}")
    print(f"  Median time to first NVPS: {metrics['time_to_first_nvps_hours'].median():.1f} hours")

    return metrics


# ──────────────────────────────────────────────
# Step 3: NVPS score distribution
# ──────────────────────────────────────────────
def analyze_score_distribution(nvps):
    """Analyze distribution of NVPS scores."""
    print("\nStep 3: Analyzing NVPS score distribution...")
    scores = nvps["numerical_value"].dropna()
    n_total = len(scores)
    n_zero = (scores == 0).sum()
    n_nonzero = n_total - n_zero

    print(f"  Total scored assessments: {n_total:,}")
    print(f"  Score = 0: {n_zero:,} ({100*n_zero/n_total:.1f}%)")
    print(f"  Score > 0: {n_nonzero:,} ({100*n_nonzero/n_total:.1f}%)")
    if n_nonzero > 0:
        print(f"  Non-zero scores distribution:")
        print(f"    {scores[scores > 0].describe().to_string()}")

    return {"n_total": n_total, "n_zero": n_zero, "pct_zero": 100 * n_zero / n_total}


# ──────────────────────────────────────────────
# Step 4: Temporal patterns
# ──────────────────────────────────────────────
def analyze_temporal_patterns(nvps):
    """Analyze when NVPS is documented (hour of day, MV course)."""
    print("\nStep 4: Analyzing temporal patterns...")

    # Hour of day
    nvps["hour_of_day"] = nvps["recorded_dttm"].dt.hour
    hourly_counts = nvps.groupby("hour_of_day").size()

    # Day vs night
    nvps["is_day"] = nvps["hour_of_day"].between(7, 18)  # 7am-7pm
    day_count = nvps["is_day"].sum()
    night_count = (~nvps["is_day"]).sum()
    day_hours = 12
    night_hours = 12
    print(f"  Day (7am-7pm) assessments: {day_count:,} ({day_count/day_hours:.0f}/hr)")
    print(f"  Night (7pm-7am) assessments: {night_count:,} ({night_count/night_hours:.0f}/hr)")

    # Documentation rate by MV day
    nvps["mv_day"] = (nvps["hours_from_mv_start"] // 24).astype(int)
    daily_rate = (
        nvps.groupby(["hospitalization_id", "mv_day"]).size()
        .reset_index(name="n_per_day")
    )

    return hourly_counts, daily_rate


# ──────────────────────────────────────────────
# Step 5: Figures
# ──────────────────────────────────────────────
def plot_gap_distribution(metrics):
    """Plot distribution of inter-assessment gaps."""
    print("\nStep 5a: Plotting gap distribution...")
    has_gaps = metrics[metrics["median_gap_hours"].notna()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        has_gaps["median_gap_hours"].clip(upper=24), bins=50,
        color="steelblue", edgecolor="white",
    )
    axes[0].axvline(x=4, color="red", linestyle="--", label="Q4h target")
    axes[0].set_xlabel("Median Gap Between NVPS Assessments (hours)")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_title("Median Inter-Assessment Gap")
    axes[0].legend()

    axes[1].hist(
        has_gaps["compliance_rate"], bins=50,
        color="teal", edgecolor="white",
    )
    axes[1].set_xlabel("4-Hour Block Compliance Rate")
    axes[1].set_ylabel("Number of Patients")
    axes[1].set_title("NVPS Q4H Documentation Compliance")

    fig.tight_layout()
    save_figure(fig, "fig4_nvps_gap_distribution")
    plt.close(fig)


def plot_nvps_by_mv_hour(daily_rate):
    """Plot NVPS documentation rate over MV course by day."""
    print("Step 5b: Plotting NVPS by MV day...")
    daily_summary = (
        daily_rate[daily_rate["mv_day"] <= 14]
        .groupby("mv_day")["n_per_day"]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
        .reset_index()
    )
    daily_summary.columns = ["mv_day", "median", "q25", "q75", "n_patients"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(daily_summary["mv_day"], daily_summary["median"], color="steelblue", alpha=0.8)
    ax.errorbar(
        daily_summary["mv_day"], daily_summary["median"],
        yerr=[
            daily_summary["median"] - daily_summary["q25"],
            daily_summary["q75"] - daily_summary["median"],
        ],
        fmt="none", color="black", capsize=3,
    )
    ax.axhline(y=6, color="red", linestyle="--", label="Q4H target (6/day)")
    ax.set_xlabel("MV Day")
    ax.set_ylabel("NVPS Assessments per Day (median)")
    ax.set_title("NVPS Documentation Rate Over MV Course")
    ax.legend()

    # Patient count annotation
    for _, row in daily_summary.iterrows():
        ax.text(
            row["mv_day"], -0.5, f"n={int(row['n_patients'])}",
            ha="center", va="top", fontsize=7, color="gray",
        )

    fig.tight_layout()
    save_figure(fig, "fig5_nvps_by_mv_day")
    plt.close(fig)


def plot_nvps_by_time_of_day(hourly_counts):
    """Plot NVPS documentation by hour of day."""
    print("Step 5c: Plotting NVPS by time of day...")
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = range(24)
    counts = [hourly_counts.get(h, 0) for h in hours]
    colors = ["#4a90d9" if 7 <= h <= 18 else "#2c3e50" for h in hours]

    ax.bar(hours, counts, color=colors, edgecolor="white")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total NVPS Assessments")
    ax.set_title("NVPS Documentation by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)

    # Add shift labels
    ax.axvspan(7, 19, alpha=0.05, color="yellow", label="Day shift (7am-7pm)")
    ax.axvspan(-0.5, 7, alpha=0.05, color="gray")
    ax.axvspan(19, 23.5, alpha=0.05, color="gray", label="Night shift")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, "fig6_nvps_by_time_of_day")
    plt.close(fig)


def plot_score_distribution(nvps):
    """Plot NVPS score distribution."""
    print("Step 5d: Plotting NVPS score distribution...")
    scores = nvps["numerical_value"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All scores
    score_counts = scores.value_counts().sort_index()
    axes[0].bar(score_counts.index, score_counts.values, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("NVPS Total Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("NVPS Score Distribution (All Assessments)")

    # Annotate % zero
    pct_zero = 100 * (scores == 0).sum() / len(scores)
    axes[0].text(
        0.95, 0.95, f"{pct_zero:.1f}% scored 0",
        transform=axes[0].transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Non-zero scores only
    nonzero = scores[scores > 0]
    if len(nonzero) > 0:
        nz_counts = nonzero.value_counts().sort_index()
        axes[1].bar(nz_counts.index, nz_counts.values, color="coral", edgecolor="white")
        axes[1].set_xlabel("NVPS Total Score")
        axes[1].set_ylabel("Count")
        axes[1].set_title("NVPS Score Distribution (Non-Zero Only)")
    else:
        axes[1].text(0.5, 0.5, "No non-zero scores", ha="center", va="center",
                     transform=axes[1].transAxes)

    fig.tight_layout()
    save_figure(fig, "fig7_nvps_score_distribution")
    plt.close(fig)


# ──────────────────────────────────────────────
# Step 6: Summary table
# ──────────────────────────────────────────────
def create_summary_table(metrics, score_stats):
    """Create Table 3: NVPS documentation summary."""
    print("\nStep 6: Creating NVPS documentation summary table...")

    has = metrics[metrics["has_nvps"]]
    rows = []

    rows.append(("Patients with any NVPS during MV", f"{len(has):,}", f"{100*len(has)/len(metrics):.1f}%"))
    rows.append(("Patients without any NVPS during MV", f"{(~metrics['has_nvps']).sum():,}", ""))
    rows.append(("", "", ""))

    rows.append(("Among patients with NVPS:", "", ""))
    for label, col in [
        ("Total NVPS assessments per patient", "n_nvps"),
        ("Assessments per day", "assessments_per_day"),
        ("Median inter-assessment gap (hours)", "median_gap_hours"),
        ("Maximum inter-assessment gap (hours)", "max_gap_hours"),
        ("Time to first NVPS from MV start (hours)", "time_to_first_nvps_hours"),
        ("4-hour block compliance rate", "compliance_rate"),
    ]:
        vals = has[col].dropna()
        med = vals.median()
        q25 = vals.quantile(0.25)
        q75 = vals.quantile(0.75)
        if col == "compliance_rate":
            rows.append((label, f"{med:.2f}", f"[{q25:.2f} - {q75:.2f}]"))
        else:
            rows.append((label, f"{med:.1f}", f"[{q25:.1f} - {q75:.1f}]"))

    rows.append(("", "", ""))
    rows.append(("NVPS Score Distribution", "", ""))
    rows.append(("Score = 0", f"{score_stats['n_zero']:,}", f"{score_stats['pct_zero']:.1f}%"))
    rows.append(("Score > 0", f"{score_stats['n_total'] - score_stats['n_zero']:,}",
                 f"{100 - score_stats['pct_zero']:.1f}%"))

    summary_df = pd.DataFrame(rows, columns=["Variable", "Value", "Detail"])
    save_table(summary_df, "table3_nvps_documentation")
    print(summary_df.to_string(index=False))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("03_aim2_nvps.py — NVPS Documentation Patterns")
    print("=" * 60)

    nvps, mv_windows = load_nvps_during_mv()

    # Documentation metrics
    metrics = compute_documentation_metrics(nvps, mv_windows)
    save_intermediate(metrics, "nvps_metrics")

    # Score distribution
    score_stats = analyze_score_distribution(nvps)

    # Temporal patterns
    hourly_counts, daily_rate = analyze_temporal_patterns(nvps)

    # Figures
    plot_gap_distribution(metrics)
    plot_nvps_by_mv_hour(daily_rate)
    plot_nvps_by_time_of_day(hourly_counts)
    plot_score_distribution(nvps)

    # Summary table
    create_summary_table(metrics, score_stats)

    print("\n" + "=" * 60)
    print("DONE. Aim 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
