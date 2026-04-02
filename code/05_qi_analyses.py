"""
05_qi_analyses.py
=================
Quality improvement analyses testing whether NVPS is used meaningfully:

  Analysis 1: Does a non-zero NVPS lead to fentanyl dose changes?
  Analysis 2: Is NVPS documented before fentanyl boluses?
  Analysis 3: NVPS-RASS concordance — are zeros plausible given sedation depth?

Requires: 01_build_cohort.py outputs

Outputs:
  - tables/table7_nvps_action.csv
  - tables/table8_bolus_assessment.csv
  - tables/table9_nvps_rass_concordance.csv
  - figures/fig11_nvps_action_comparison.pdf/.png
  - figures/fig12_bolus_nvps_timing.pdf/.png
  - figures/fig13_nvps_by_rass.pdf/.png
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


# ──────────────────────────────────────────────
# Shared data loading
# ──────────────────────────────────────────────
def load_shared_data():
    """Load cohort, NVPS, fentanyl continuous, and fentanyl intermittent during MV."""
    print("Loading shared data...")

    # Cohort
    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    cohort["mv_end"] = pd.to_datetime(cohort["mv_end"])
    mv = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()
    print(f"  Cohort: {len(cohort):,} patients")

    # NVPS TOTAL (0-10 scale)
    nvps = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "NVPS")],
    )
    nvps["recorded_dttm"] = pd.to_datetime(nvps["recorded_dttm"])
    nvps = nvps[nvps["assessment_name"].str.contains("TOTAL", case=False, na=False)]
    nvps["numerical_value"] = pd.to_numeric(nvps["numerical_value"], errors="coerce")
    nvps = nvps.merge(mv, on="hospitalization_id")
    nvps = nvps[(nvps["recorded_dttm"] >= nvps["mv_start"]) & (nvps["recorded_dttm"] <= nvps["mv_end"])]
    print(f"  NVPS during MV: {len(nvps):,} assessments")
    print(f"  Non-zero NVPS: {(nvps['numerical_value'] > 0).sum():,} ({100*(nvps['numerical_value'] > 0).mean():.1f}%)")

    # Fentanyl continuous
    fent_c = load_clif_table(
        "medication_admin_continuous",
        filters=[("med_category", "==", "fentanyl")],
    )
    fent_c["admin_dttm"] = pd.to_datetime(fent_c["admin_dttm"])
    mask = ~fent_c["med_name"].str.upper().str.contains("|".join(FENTANYL_EXCLUDE_NAMES), na=False)
    fent_c = fent_c[mask]
    fent_c = fent_c.merge(mv, on="hospitalization_id")
    fent_c = fent_c[(fent_c["admin_dttm"] >= fent_c["mv_start"]) & (fent_c["admin_dttm"] <= fent_c["mv_end"])]
    print(f"  Fentanyl continuous during MV: {len(fent_c):,} records")

    # Fentanyl intermittent
    fent_i = load_clif_table(
        "medication_admin_intermittent",
        filters=[("med_category", "==", "fentanyl")],
    )
    fent_i["admin_dttm"] = pd.to_datetime(fent_i["admin_dttm"])
    mask = ~fent_i["med_name"].str.upper().str.contains("|".join(FENTANYL_EXCLUDE_NAMES), na=False)
    fent_i = fent_i[mask]
    fent_i = fent_i[fent_i["mar_action_category"].isin(["given", "bolus"])]
    fent_i = fent_i.merge(mv, on="hospitalization_id")
    fent_i = fent_i[(fent_i["admin_dttm"] >= fent_i["mv_start"]) & (fent_i["admin_dttm"] <= fent_i["mv_end"])]
    print(f"  Fentanyl boluses during MV: {len(fent_i):,} records")

    return cohort, nvps, fent_c, fent_i


# ══════════════════════════════════════════════
# Analysis 1: Does non-zero NVPS drive clinical action?
# ══════════════════════════════════════════════
def analysis1_nvps_drives_action(nvps, fent_c, fent_i):
    """Check if non-zero NVPS is followed by fentanyl dose change within 2h."""
    print("\n" + "=" * 60)
    print("Analysis 1: Does non-zero NVPS drive fentanyl action?")
    print("=" * 60)

    # Separate non-zero and zero NVPS
    nvps_nonzero = nvps[nvps["numerical_value"] > 0].copy()
    nvps_zero = nvps[nvps["numerical_value"] == 0].copy()

    # Sample zero NVPS to match non-zero count (for tractability)
    n_nonzero = len(nvps_nonzero)
    sample_size = min(n_nonzero * 3, len(nvps_zero))
    nvps_zero_sample = nvps_zero.sample(n=sample_size, random_state=42)

    print(f"  Non-zero NVPS assessments: {n_nonzero:,}")
    print(f"  Zero NVPS sample (3x control): {sample_size:,}")

    # Fentanyl actions: dose changes in continuous + boluses in intermittent
    dose_changes = fent_c[fent_c["mar_action_category"] == "dose_change"][
        ["hospitalization_id", "admin_dttm"]
    ].copy()
    boluses = fent_i[["hospitalization_id", "admin_dttm"]].copy()
    all_actions = pd.concat([dose_changes, boluses]).sort_values(
        ["hospitalization_id", "admin_dttm"]
    )
    print(f"  Total fentanyl actions (dose changes + boluses): {len(all_actions):,}")

    def pct_followed_by_action(nvps_subset, actions, window_hours=2):
        """For each NVPS, check if a fentanyl action occurs within window_hours after."""
        followed = 0
        total = 0
        for hid, grp in nvps_subset.groupby("hospitalization_id"):
            pt_actions = actions[actions["hospitalization_id"] == hid]["admin_dttm"].values
            if len(pt_actions) == 0:
                total += len(grp)
                continue
            for _, row in grp.iterrows():
                nvps_time = row["recorded_dttm"]
                # Check if any action within (0, window_hours] after NVPS
                time_diffs = (pt_actions - np.datetime64(nvps_time)) / np.timedelta64(1, "h")
                if np.any((time_diffs > 0) & (time_diffs <= window_hours)):
                    followed += 1
                total += 1
        return followed, total

    print("  Computing action rates (this may take a few minutes)...")
    f_nz, t_nz = pct_followed_by_action(nvps_nonzero, all_actions)
    f_z, t_z = pct_followed_by_action(nvps_zero_sample, all_actions)

    rate_nz = 100 * f_nz / t_nz if t_nz > 0 else 0
    rate_z = 100 * f_z / t_z if t_z > 0 else 0

    print(f"\n  Results:")
    print(f"  {'─' * 50}")
    print(f"  Non-zero NVPS → fentanyl action within 2h: {f_nz:,}/{t_nz:,} ({rate_nz:.1f}%)")
    print(f"  Zero NVPS → fentanyl action within 2h:     {f_z:,}/{t_z:,} ({rate_z:.1f}%)")
    print(f"  {'─' * 50}")

    # Summary table
    rows = [
        ("Non-zero NVPS (score 1-10)", t_nz, f_nz, f"{rate_nz:.1f}%"),
        ("Zero NVPS (score 0, sampled)", t_z, f_z, f"{rate_z:.1f}%"),
    ]
    table = pd.DataFrame(rows, columns=["NVPS Group", "N Assessments", "Followed by Action", "Action Rate"])
    save_table(table, "table7_nvps_action")

    # Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Non-zero NVPS\n(score 1-10)", "Zero NVPS\n(score 0)"],
        [rate_nz, rate_z],
        color=["#d32f2f", "#78909c"],
        edgecolor="white",
        width=0.5,
    )
    for bar, rate in zip(bars, [rate_nz, rate_z]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel("% Followed by Fentanyl Action Within 2h")
    ax.set_title("Fentanyl Dose Change/Bolus After NVPS Assessment")
    ax.set_ylim(0, max(rate_nz, rate_z) * 1.3)
    fig.tight_layout()
    save_figure(fig, "fig11_nvps_action_comparison")
    plt.close(fig)


# ══════════════════════════════════════════════
# Analysis 2: Is NVPS documented before fentanyl boluses?
# ══════════════════════════════════════════════
def analysis2_nvps_before_bolus(nvps, fent_i):
    """Check if NVPS is documented within 60 min before/after fentanyl boluses."""
    print("\n" + "=" * 60)
    print("Analysis 2: Is NVPS documented before fentanyl boluses?")
    print("=" * 60)

    print(f"  Fentanyl boluses: {len(fent_i):,}")

    time_diffs_all = []  # minutes from nearest NVPS to bolus (negative = NVPS before bolus)

    pre_60 = 0   # NVPS within 60 min BEFORE bolus
    post_60 = 0  # NVPS within 60 min AFTER bolus
    either_60 = 0
    total = 0

    for hid, bolus_grp in fent_i.groupby("hospitalization_id"):
        pt_nvps_times = nvps.loc[
            nvps["hospitalization_id"] == hid, "recorded_dttm"
        ].values
        if len(pt_nvps_times) == 0:
            total += len(bolus_grp)
            continue

        for _, row in bolus_grp.iterrows():
            bolus_time = row["admin_dttm"]
            diffs_min = (pt_nvps_times - np.datetime64(bolus_time)) / np.timedelta64(1, "m")

            # Nearest NVPS overall
            if len(diffs_min) > 0:
                nearest_idx = np.argmin(np.abs(diffs_min))
                time_diffs_all.append(diffs_min[nearest_idx])

            # Pre-bolus: NVPS within -60 to 0 minutes
            has_pre = np.any((diffs_min >= -60) & (diffs_min <= 0))
            # Post-bolus: NVPS within 0 to 60 minutes
            has_post = np.any((diffs_min > 0) & (diffs_min <= 60))

            if has_pre:
                pre_60 += 1
            if has_post:
                post_60 += 1
            if has_pre or has_post:
                either_60 += 1
            total += 1

    pct_pre = 100 * pre_60 / total if total > 0 else 0
    pct_post = 100 * post_60 / total if total > 0 else 0
    pct_either = 100 * either_60 / total if total > 0 else 0

    print(f"\n  Results:")
    print(f"  {'─' * 50}")
    print(f"  Total boluses analyzed: {total:,}")
    print(f"  NVPS within 60 min BEFORE bolus: {pre_60:,} ({pct_pre:.1f}%)")
    print(f"  NVPS within 60 min AFTER bolus:  {post_60:,} ({pct_post:.1f}%)")
    print(f"  NVPS within 60 min of bolus:     {either_60:,} ({pct_either:.1f}%)")
    print(f"  {'─' * 50}")

    # Summary table
    rows = [
        ("NVPS within 60 min before bolus (pre-assessment)", total, pre_60, f"{pct_pre:.1f}%"),
        ("NVPS within 60 min after bolus (reassessment)", total, post_60, f"{pct_post:.1f}%"),
        ("NVPS within 60 min of bolus (either)", total, either_60, f"{pct_either:.1f}%"),
    ]
    table = pd.DataFrame(rows, columns=["Metric", "N Boluses", "N with NVPS", "Rate"])
    save_table(table, "table8_bolus_assessment")

    # Figure: histogram of time from nearest NVPS to bolus
    if len(time_diffs_all) > 0:
        diffs = np.array(time_diffs_all)
        # Clip to ±240 min for visualization
        diffs_clipped = diffs[(diffs >= -240) & (diffs <= 240)]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(diffs_clipped, bins=96, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Bolus time")
        ax.axvline(x=-60, color="orange", linestyle=":", linewidth=1.5, label="60 min before")
        ax.axvline(x=60, color="green", linestyle=":", linewidth=1.5, label="60 min after")
        ax.set_xlabel("Minutes from Nearest NVPS to Fentanyl Bolus\n(negative = NVPS before bolus)")
        ax.set_ylabel("Count")
        ax.set_title("Timing of NVPS Relative to Fentanyl Boluses")
        ax.legend()

        # Annotate rates
        ax.text(0.02, 0.95, f"Pre-bolus NVPS (≤60 min): {pct_pre:.1f}%",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        fig.tight_layout()
        save_figure(fig, "fig12_bolus_nvps_timing")
        plt.close(fig)


# ══════════════════════════════════════════════
# Analysis 3: NVPS-RASS concordance
# ══════════════════════════════════════════════
def analysis3_nvps_rass_concordance(nvps, cohort):
    """Cross-tabulate NVPS scores by RASS sedation depth."""
    print("\n" + "=" * 60)
    print("Analysis 3: NVPS-RASS concordance")
    print("=" * 60)

    mv = cohort[["hospitalization_id", "mv_start", "mv_end"]].copy()

    # Load RASS during MV
    rass = load_clif_table(
        "patient_assessments",
        filters=[("assessment_category", "==", "RASS")],
    )
    rass["recorded_dttm"] = pd.to_datetime(rass["recorded_dttm"])
    rass["rass_value"] = pd.to_numeric(rass["numerical_value"], errors="coerce")
    rass = rass.dropna(subset=["rass_value"])
    rass = rass.merge(mv, on="hospitalization_id")
    rass = rass[(rass["recorded_dttm"] >= rass["mv_start"]) & (rass["recorded_dttm"] <= rass["mv_end"])]
    print(f"  RASS during MV: {len(rass):,}")

    # For each NVPS, find nearest RASS within ±2 hours
    print("  Pairing NVPS with nearest RASS (±2h)... this may take several minutes...")

    nvps_scores = []
    rass_scores = []

    for hid, nvps_grp in nvps.groupby("hospitalization_id"):
        pt_rass = rass[rass["hospitalization_id"] == hid]
        if len(pt_rass) == 0:
            continue

        rass_times = pt_rass["recorded_dttm"].values
        rass_vals = pt_rass["rass_value"].values

        for _, row in nvps_grp.iterrows():
            nvps_time = row["recorded_dttm"]
            diffs_hr = (rass_times - np.datetime64(nvps_time)) / np.timedelta64(1, "h")
            within_window = np.abs(diffs_hr) <= 2

            if np.any(within_window):
                nearest_idx = np.argmin(np.abs(diffs_hr))
                if abs(diffs_hr[nearest_idx]) <= 2:
                    nvps_scores.append(row["numerical_value"])
                    rass_scores.append(rass_vals[nearest_idx])

    pairs = pd.DataFrame({"nvps_score": nvps_scores, "rass_value": rass_scores})
    print(f"  Paired NVPS-RASS assessments: {len(pairs):,}")

    # Categorize RASS
    def rass_category(val):
        if val <= -4:
            return "Deep sedation\n(RASS -5, -4)"
        elif val <= -2:
            return "Moderate sedation\n(RASS -3, -2)"
        elif val <= 0:
            return "Light sedation\n(RASS -1, 0)"
        else:
            return "Agitated\n(RASS +1 to +3)"

    pairs["rass_category"] = pairs["rass_value"].apply(rass_category)

    def nvps_bin(val):
        if val == 0:
            return "NVPS = 0"
        elif val <= 3:
            return "NVPS 1-3"
        elif val <= 6:
            return "NVPS 4-6"
        else:
            return "NVPS 7-10"

    pairs["nvps_label"] = pairs["nvps_score"].apply(nvps_bin)

    # Cross-tabulation
    rass_order = [
        "Deep sedation\n(RASS -5, -4)",
        "Moderate sedation\n(RASS -3, -2)",
        "Light sedation\n(RASS -1, 0)",
        "Agitated\n(RASS +1 to +3)",
    ]
    ct = pd.crosstab(pairs["rass_category"], pairs["nvps_label"], margins=True)
    ct_pct = pd.crosstab(pairs["rass_category"], pairs["nvps_label"], normalize="index") * 100

    print(f"\n  NVPS Score Distribution by RASS Category:")
    print(f"  {'─' * 60}")
    for cat in rass_order:
        if cat in ct_pct.index:
            row = ct_pct.loc[cat]
            n = ct.loc[cat, "All"] if "All" in ct.columns else ct.loc[cat].sum()
            zero_pct = row.get("NVPS = 0", 0)
            print(f"  {cat.replace(chr(10), ' ')}: {zero_pct:.1f}% scored 0 (n={n:,})")
    print(f"  {'─' * 60}")

    # Highlight key finding
    if "Light sedation\n(RASS -1, 0)" in ct_pct.index:
        light_zero = ct_pct.loc["Light sedation\n(RASS -1, 0)"].get("NVPS = 0", 0)
        print(f"\n  KEY FINDING: {light_zero:.1f}% of lightly sedated patients (RASS -1, 0)")
        print(f"  still scored NVPS = 0. These patients should be able to exhibit pain behaviors.")

    # Save cross-tab
    ct_save = ct_pct.reindex(rass_order).round(1)
    ct_save["N"] = [ct.loc[cat, "All"] if cat in ct.index else 0 for cat in rass_order]
    save_table(ct_save.reset_index(), "table9_nvps_rass_concordance")

    # Figure: stacked bar
    fig, ax = plt.subplots(figsize=(10, 6))
    ct_plot = ct_pct.reindex(rass_order)
    nvps_cols = ["NVPS = 0", "NVPS 1-3", "NVPS 4-6", "NVPS 7-10"]
    nvps_cols = [c for c in nvps_cols if c in ct_plot.columns]
    colors = ["#78909c", "#f57c00", "#d32f2f", "#7b1fa2"]

    bottom = np.zeros(len(ct_plot))
    for col, color in zip(nvps_cols, colors):
        vals = ct_plot[col].fillna(0).values
        ax.bar(range(len(ct_plot)), vals, bottom=bottom, color=color,
               label=col, edgecolor="white", width=0.6)
        # Add percentage labels for non-trivial segments
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += vals

    ax.set_xticks(range(len(ct_plot)))
    ax.set_xticklabels([idx.replace("\n", "\n") for idx in ct_plot.index], fontsize=9)
    ax.set_ylabel("Percentage")
    ax.set_title("NVPS Score Distribution by Sedation Depth (RASS)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    # Add N labels at top
    for i, cat in enumerate(rass_order):
        if cat in ct.index:
            n = ct.loc[cat, "All"] if "All" in ct.columns else ct.loc[cat].sum()
            ax.text(i, 102, f"n={n:,}", ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    save_figure(fig, "fig13_nvps_by_rass")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("05_qi_analyses.py — NVPS Clinical Utility")
    print("=" * 60)

    cohort, nvps, fent_c, fent_i = load_shared_data()

    analysis1_nvps_drives_action(nvps, fent_c, fent_i)
    analysis2_nvps_before_bolus(nvps, fent_i)
    analysis3_nvps_rass_concordance(nvps, cohort)

    print("\n" + "=" * 60)
    print("DONE. All QI analyses complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
