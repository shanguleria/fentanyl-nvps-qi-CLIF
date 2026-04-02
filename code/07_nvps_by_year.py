"""
07_nvps_by_year.py
==================
NVPS score distribution stratified by year of hospitalization.

Recreates Fig7 (score distribution) but with one panel per year,
showing how non-zero NVPS scoring patterns change over time.

Requires: 01_build_cohort.py outputs

Outputs:
  - figures/fig17_nvps_scores_by_year.pdf/.png
  - tables/table_nvps_scores_by_year.csv
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_clif_table, load_intermediate, save_table, save_figure

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_nvps_with_year():
    """Load NVPS TOTAL scores during MV, annotated with hospitalization year."""
    print("Step 1: Loading cohort and NVPS data...")
    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])
    cohort["admission_dttm"] = pd.to_datetime(cohort["admission_dttm"])
    cohort["admission_year"] = cohort["admission_dttm"].dt.year

    mv_windows = cohort[["hospitalization_id", "mv_start", "mv_end", "admission_year"]].copy()

    # Load NVPS TOTAL scores (0-10 scale)
    nvps = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "NVPS")],
    )
    nvps["recorded_dttm"] = pd.to_datetime(nvps["recorded_dttm"])
    nvps = nvps[nvps["assessment_name"].str.contains("TOTAL", case=False, na=False)]

    # Restrict to MV windows and attach year
    nvps = nvps.merge(mv_windows, on="hospitalization_id")
    nvps = nvps[
        (nvps["recorded_dttm"] >= nvps["mv_start"])
        & (nvps["recorded_dttm"] <= nvps["mv_end"])
    ]
    nvps["numerical_value"] = pd.to_numeric(nvps["numerical_value"], errors="coerce")

    years = sorted(nvps["admission_year"].unique())
    print(f"  NVPS records during MV: {len(nvps):,}")
    print(f"  Years represented: {years}")
    for yr in years:
        yr_data = nvps[nvps["admission_year"] == yr]
        n_pts = yr_data["hospitalization_id"].nunique()
        n_scores = yr_data["numerical_value"].dropna().shape[0]
        print(f"    {yr}: {n_pts:,} patients, {n_scores:,} scored assessments")

    return nvps, mv_windows, years


def plot_scores_by_year(nvps, years):
    """Create multi-panel figure: NVPS score distribution by year (non-zero focus)."""
    print("\nStep 2: Plotting NVPS score distribution by year...")

    n_years = len(years)
    fig, axes = plt.subplots(2, n_years, figsize=(4 * n_years, 9),
                             squeeze=False)

    for i, yr in enumerate(years):
        yr_scores = nvps.loc[nvps["admission_year"] == yr, "numerical_value"].dropna()

        # Top row: all scores
        ax_all = axes[0, i]
        score_counts = yr_scores.value_counts().sort_index()
        ax_all.bar(score_counts.index, score_counts.values, color="steelblue", edgecolor="white")
        ax_all.set_xlabel("NVPS Total Score")
        ax_all.set_ylabel("Count" if i == 0 else "")
        ax_all.set_title(f"{yr}\n(n={len(yr_scores):,} assessments)")

        pct_zero = 100 * (yr_scores == 0).sum() / len(yr_scores) if len(yr_scores) > 0 else 0
        ax_all.text(
            0.95, 0.95, f"{pct_zero:.1f}% = 0",
            transform=ax_all.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        # Bottom row: non-zero scores only
        ax_nz = axes[1, i]
        nonzero = yr_scores[yr_scores > 0]
        if len(nonzero) > 0:
            nz_counts = nonzero.value_counts().sort_index()
            ax_nz.bar(nz_counts.index, nz_counts.values, color="coral", edgecolor="white")
            ax_nz.set_xlabel("NVPS Total Score")
            ax_nz.set_ylabel("Count" if i == 0 else "")
            ax_nz.set_title(f"Non-Zero Only (n={len(nonzero):,})")
        else:
            ax_nz.text(0.5, 0.5, "No non-zero scores", ha="center", va="center",
                       transform=ax_nz.transAxes)
            ax_nz.set_title("Non-Zero Only")

    fig.suptitle("NVPS Score Distribution by Year of Hospitalization", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig17_nvps_scores_by_year")
    plt.close(fig)


def create_year_summary_table(nvps, years):
    """Create summary table of NVPS scores by year."""
    print("\nStep 3: Creating summary table...")

    rows = []
    for yr in years:
        yr_data = nvps[nvps["admission_year"] == yr]
        yr_scores = yr_data["numerical_value"].dropna()
        n_patients = yr_data["hospitalization_id"].nunique()
        n_assessments = len(yr_scores)
        n_zero = (yr_scores == 0).sum()
        n_nonzero = n_assessments - n_zero
        pct_zero = 100 * n_zero / n_assessments if n_assessments > 0 else 0
        pct_nonzero = 100 * n_nonzero / n_assessments if n_assessments > 0 else 0

        # Non-zero score breakdown by range
        score_1_3 = ((yr_scores >= 1) & (yr_scores <= 3)).sum()
        score_4_6 = ((yr_scores >= 4) & (yr_scores <= 6)).sum()
        score_7_10 = ((yr_scores >= 7) & (yr_scores <= 10)).sum()
        mean_nonzero = yr_scores[yr_scores > 0].mean() if n_nonzero > 0 else 0

        rows.append({
            "Year": yr,
            "N Patients": n_patients,
            "N Assessments": n_assessments,
            "N Zero (%)": f"{n_zero:,} ({pct_zero:.1f}%)",
            "N Non-Zero (%)": f"{n_nonzero:,} ({pct_nonzero:.1f}%)",
            "Score 1-3": score_1_3,
            "Score 4-6": score_4_6,
            "Score 7-10": score_7_10,
            "Mean Non-Zero": round(mean_nonzero, 1),
        })

    df = pd.DataFrame(rows)
    save_table(df, "table_nvps_scores_by_year")
    print(df.to_string(index=False))
    return df


def main():
    print("=" * 60)
    print("07_nvps_by_year.py — NVPS Scores by Year")
    print("=" * 60)

    nvps, _, years = load_nvps_with_year()
    plot_scores_by_year(nvps, years)
    create_year_summary_table(nvps, years)

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
