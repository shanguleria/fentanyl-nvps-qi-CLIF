"""
04_aim3_association.py
=====================
Aim 3: Association between NVPS documentation regularity and ventilator-free days.

Primary analysis:
  - 72h landmark: measure NVPS compliance in first 72h of MV
  - Outcome: VFD-28
  - Linear regression + cause-specific hazard models (competing risks)

Competing risks:
  - Cause-specific Cox models: separate models for extubation and death
  - Cumulative incidence curves (Aalen-Johansen) by compliance quartile
  - Fine-Gray subdistribution hazard (sensitivity analysis)

Sensitivity analyses:
  - Full MV duration (no landmark)
  - Exclude deaths
  - Alternative exposure definitions
  - Stratify by fentanyl type
  - Alternative landmark windows (48h, 96h)

Requires: 01_build_cohort.py and 03_aim2_nvps.py outputs

Outputs:
  - tables/table4_primary_regression.csv
  - tables/table5_cause_specific_extubation.csv
  - tables/table5b_cause_specific_death.csv
  - tables/table5c_fine_gray.csv
  - tables/table6_sensitivity.csv
  - figures/fig8_cumulative_incidence.pdf/.png
  - figures/fig9_forest_plot.pdf/.png
  - figures/fig10_vfd_by_compliance.pdf/.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter, AalenJohansenFitter
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_intermediate,
    save_table,
    save_figure,
    CONCURRENT_SEDATIVES,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ──────────────────────────────────────────────
# Step 1: Build analysis dataset
# ──────────────────────────────────────────────
def build_analysis_dataset():
    """Merge cohort, NVPS metrics, and prepare for regression."""
    print("Step 1: Building analysis dataset...")

    cohort = load_intermediate("cohort")
    cohort["mv_start"] = pd.to_datetime(cohort["mv_start"])
    nvps_metrics = load_intermediate("nvps_metrics")

    # Merge
    df = cohort.merge(nvps_metrics, on="hospitalization_id", how="left")

    # Fill missing NVPS metrics (patients with no NVPS at all)
    for col in ["compliance_rate", "compliance_72h", "assessments_per_day",
                "assessments_per_day_72h", "n_nvps", "has_nvps"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df["has_nvps"] = df["has_nvps"].astype(bool)

    print(f"  Full dataset: {len(df):,} patients")
    print(f"  With NVPS: {df['has_nvps'].sum():,}")
    return df


def apply_landmark(df, landmark_hours=72):
    """Apply landmark: restrict to patients still intubated at landmark time."""
    print(f"\n  Applying {landmark_hours}h landmark...")
    landmark_df = df[df["mv_duration_hours"] >= landmark_hours].copy()

    # Use compliance measured in first 72h
    if landmark_hours == 72:
        landmark_df["exposure"] = landmark_df["compliance_72h"]
        landmark_df["exposure_label"] = "72h compliance"
    else:
        # Recompute for other landmarks — use overall compliance as approximation
        landmark_df["exposure"] = landmark_df["compliance_rate"]
        landmark_df["exposure_label"] = f"{landmark_hours}h compliance (approx)"

    # Time-to-extubation starts at landmark
    landmark_df["time_to_extubation"] = landmark_df["mv_duration_hours"] - landmark_hours
    landmark_df["time_to_extubation_days"] = landmark_df["time_to_extubation"] / 24

    # Competing risk event coding:
    #   0 = censored (still on vent at day 28, or admin censoring)
    #   1 = extubated alive
    #   2 = died before extubation
    landmark_df["extubated_alive"] = (~landmark_df["death_within_28d"]).astype(int)
    landmark_df["event_type"] = np.where(
        landmark_df["death_within_28d"], 2,  # died
        np.where(landmark_df["extubated_alive"], 1, 0)  # extubated or censored
    )

    n = len(landmark_df)
    n_full = len(df)
    print(f"  Landmark cohort: {n:,} / {n_full:,} ({100*n/n_full:.1f}%)")
    print(f"  Excluded (MV < {landmark_hours}h): {n_full - n:,}")
    print(f"  Exposure (compliance_72h) distribution:")
    print(f"    {landmark_df['exposure'].describe().to_string()}")

    return landmark_df


# ──────────────────────────────────────────────
# Step 2: Prepare covariates
# ──────────────────────────────────────────────
def prepare_covariates(df):
    """Create regression-ready covariate matrix."""
    print("\nStep 2: Preparing covariates...")

    covariates = ["exposure"]

    # Age (continuous)
    if "age_at_admission" in df.columns:
        covariates.append("age_at_admission")

    # Sex (binary)
    if "sex_category" in df.columns:
        df["female"] = (df["sex_category"].str.lower() == "female").astype(int)
        covariates.append("female")

    # SOFA (admission)
    if "sofa_total" in df.columns:
        covariates.append("sofa_total")

    # Fentanyl type (dummies)
    if "fentanyl_type" in df.columns:
        fent_dummies = pd.get_dummies(df["fentanyl_type"], prefix="fent", drop_first=True, dtype=int)
        df = pd.concat([df, fent_dummies], axis=1)
        covariates.extend(fent_dummies.columns.tolist())

    # Concurrent sedatives
    for sed in CONCURRENT_SEDATIVES:
        col = f"received_{sed}"
        if col in df.columns:
            df[col] = df[col].astype(int)
            covariates.append(col)

    # Drop rows with missing covariates
    available_covariates = [c for c in covariates if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=available_covariates).copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with missing covariates")

    print(f"  Covariates: {available_covariates}")
    print(f"  Analysis N: {len(df):,}")

    return df, available_covariates


# ──────────────────────────────────────────────
# Step 3: Primary analysis — Linear regression
# ──────────────────────────────────────────────
def run_linear_regression(df, covariates, outcome="vfd_28", label="Primary"):
    """Run OLS regression of VFD-28 on NVPS compliance + covariates."""
    print(f"\nStep 3: Linear regression ({label})...")

    X = df[covariates].copy()
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.OLS(y, X).fit(cov_type="HC1")  # Robust standard errors

    print(f"\n  {label} Regression Results:")
    print(f"  {'─' * 50}")
    print(f"  N = {int(model.nobs)}")
    print(f"  R² = {model.rsquared:.3f}")
    print(f"  Adj R² = {model.rsquared_adj:.3f}")
    print(f"  {'─' * 50}")

    # Format results
    results = []
    for var in model.params.index:
        results.append({
            "Variable": var,
            "Coefficient": f"{model.params[var]:.3f}",
            "SE": f"{model.bse[var]:.3f}",
            "95% CI": f"[{model.conf_int().loc[var, 0]:.3f}, {model.conf_int().loc[var, 1]:.3f}]",
            "p-value": f"{model.pvalues[var]:.4f}",
        })
        if var == "exposure":
            print(f"  EXPOSURE (NVPS compliance): coef={model.params[var]:.3f}, "
                  f"95% CI [{model.conf_int().loc[var, 0]:.3f}, {model.conf_int().loc[var, 1]:.3f}], "
                  f"p={model.pvalues[var]:.4f}")

    results_df = pd.DataFrame(results)
    return model, results_df


# ──────────────────────────────────────────────
# Step 4: Cause-specific hazard models (competing risks)
# ──────────────────────────────────────────────
def _fit_cause_specific_cox(df, covariates, event_of_interest, event_label):
    """Fit a cause-specific Cox model for one event type.

    event_of_interest: value in 'event_type' column (1=extubation, 2=death)
    Other events are censored.
    """
    cox_df = df[covariates + ["time_to_extubation_days", "event_type"]].copy()
    cox_df = cox_df[cox_df["time_to_extubation_days"] > 0]

    # For cause-specific: event=1 if this cause occurred, 0 otherwise (censored)
    cox_df["event"] = (cox_df["event_type"] == event_of_interest).astype(int)

    cph = CoxPHFitter()
    cph.fit(
        cox_df[covariates + ["time_to_extubation_days", "event"]],
        duration_col="time_to_extubation_days",
        event_col="event",
    )

    results = []
    for var in cph.summary.index:
        hr = np.exp(cph.summary.loc[var, "coef"])
        ci_low = np.exp(cph.summary.loc[var, "coef lower 95%"])
        ci_high = np.exp(cph.summary.loc[var, "coef upper 95%"])
        p = cph.summary.loc[var, "p"]
        results.append({
            "Variable": var,
            "HR": f"{hr:.3f}",
            "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "p-value": f"{p:.4f}",
        })
        if var == "exposure":
            print(f"  EXPOSURE ({event_label}): HR={hr:.3f}, "
                  f"95% CI [{ci_low:.3f}, {ci_high:.3f}], p={p:.4f}")

    return cph, pd.DataFrame(results)


def run_cause_specific_models(df, covariates):
    """Run cause-specific hazard models for extubation and death."""
    print("\nStep 4: Cause-specific hazard models (competing risks)...")

    cox_df = df[df["time_to_extubation_days"] > 0].copy()
    n_extubated = (cox_df["event_type"] == 1).sum()
    n_died = (cox_df["event_type"] == 2).sum()
    n_censored = (cox_df["event_type"] == 0).sum()
    print(f"  Events: {n_extubated:,} extubated, {n_died:,} died, {n_censored:,} censored")

    print(f"\n  {'─' * 50}")
    print(f"  Cause-specific model 1: Extubation (death = censored)")
    print(f"  {'─' * 50}")
    cph_extub, results_extub = _fit_cause_specific_cox(
        df, covariates, event_of_interest=1, event_label="extubation"
    )

    print(f"\n  {'─' * 50}")
    print(f"  Cause-specific model 2: Death (extubation = censored)")
    print(f"  {'─' * 50}")
    cph_death, results_death = _fit_cause_specific_cox(
        df, covariates, event_of_interest=2, event_label="death"
    )

    return cph_extub, results_extub, cph_death, results_death


# ──────────────────────────────────────────────
# Step 4b: Fine-Gray subdistribution hazard (sensitivity)
# ──────────────────────────────────────────────
def run_fine_gray(df, covariates):
    """Fine-Gray subdistribution hazard via IPCW-weighted Cox regression.

    For the extubation endpoint, patients who die (competing event) remain
    in the risk set with IPCW weights rather than being censored.
    """
    print("\nStep 4b: Fine-Gray subdistribution hazard (sensitivity)...")

    fg_df = df[covariates + ["time_to_extubation_days", "event_type"]].copy()
    fg_df = fg_df[fg_df["time_to_extubation_days"] > 0]

    # Fine-Gray for extubation: competing event (death) subjects get
    # weight that increases over time. Approximation using KM of censoring.
    # Standard approach: subjects with competing event are kept in risk set
    # with decreasing weight = G(t)/G(t_competing_event)

    # Step 1: Estimate censoring distribution G(t) via KM on censoring times
    # (censoring event = competing event or admin censoring)
    times = fg_df["time_to_extubation_days"].values
    events = fg_df["event_type"].values

    # For IPCW: we need the KM estimate of the censoring survival function
    # Censoring here means: not experiencing the event of interest (extubation)
    # and not being administratively censored
    kmf_censor = KaplanMeierFitter()
    # Fit KM where "event" = censored/competing (i.e., NOT the event of interest)
    censor_indicator = (events != 1).astype(int)  # 1 if censored or died
    kmf_censor.fit(times, event_observed=censor_indicator)

    # Step 2: Assign IPCW weights
    # Subjects with event of interest (extubation): weight = 1
    # Subjects who are censored: weight = 1 (they leave risk set normally)
    # Subjects with competing event (death): weight = G(t)/G(t_death)
    #   where G(t_death) is the censoring survival probability at their event time

    weights = np.ones(len(fg_df))
    competing_mask = events == 2

    if competing_mask.sum() > 0:
        # Get G(t) at each competing event time
        competing_times = times[competing_mask]
        g_at_competing = kmf_censor.predict(competing_times).values

        # For subjects who died: they stay in risk set with diminishing weight
        # In practice, we create pseudo-observations: keep them with weight
        # G(max_time) / G(their_event_time) and extend their time to max follow-up
        max_time = times.max()
        g_at_max = max(float(kmf_censor.predict(max_time)), 0.01)

        # Modify competing event subjects: extend time, adjust weight
        fg_times = times.copy()
        fg_events = (events == 1).astype(int)  # only extubation is event
        fg_weights = weights.copy()

        # Competing event subjects: extend to max time, weighted
        for i in np.where(competing_mask)[0]:
            g_ti = max(float(kmf_censor.predict(times[i])), 0.01)
            fg_weights[i] = g_at_max / g_ti
            fg_times[i] = max_time
            fg_events[i] = 0  # censored at max time

        fg_df = fg_df.copy()
        fg_df["time_to_extubation_days"] = fg_times
        fg_df["fg_event"] = fg_events
        fg_df["fg_weight"] = fg_weights

    else:
        # No competing events — Fine-Gray = standard Cox
        fg_df["fg_event"] = (fg_df["event_type"] == 1).astype(int)
        fg_df["fg_weight"] = 1.0

    # Fit weighted Cox
    cph_fg = CoxPHFitter()
    try:
        cph_fg.fit(
            fg_df[covariates + ["time_to_extubation_days", "fg_event", "fg_weight"]],
            duration_col="time_to_extubation_days",
            event_col="fg_event",
            weights_col="fg_weight",
        )

        print(f"\n  Fine-Gray Results (extubation, death as competing risk):")
        print(f"  {'─' * 50}")

        results = []
        for var in cph_fg.summary.index:
            hr = np.exp(cph_fg.summary.loc[var, "coef"])
            ci_low = np.exp(cph_fg.summary.loc[var, "coef lower 95%"])
            ci_high = np.exp(cph_fg.summary.loc[var, "coef upper 95%"])
            p = cph_fg.summary.loc[var, "p"]
            results.append({
                "Variable": var,
                "sdHR": f"{hr:.3f}",
                "95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
                "p-value": f"{p:.4f}",
            })
            if var == "exposure":
                print(f"  EXPOSURE: sdHR={hr:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}], p={p:.4f}")

        results_df = pd.DataFrame(results)
        return cph_fg, results_df

    except Exception as e:
        print(f"  Fine-Gray model failed: {e}")
        return None, pd.DataFrame()


# ──────────────────────────────────────────────
# Step 5: Sensitivity analyses
# ──────────────────────────────────────────────
def run_sensitivity_analyses(df_full, covariates):
    """Run sensitivity analyses and collect exposure effect estimates."""
    print("\n" + "=" * 60)
    print("Step 5: Sensitivity analyses")
    print("=" * 60)

    sensitivity_results = []

    # --- S1: Full MV duration (no landmark) ---
    print("\n--- S1: Full MV duration (no landmark) ---")
    df_s1 = df_full.copy()
    df_s1["exposure"] = df_s1["compliance_rate"]
    df_s1, covs_s1 = prepare_covariates(df_s1)
    model_s1, _ = run_linear_regression(df_s1, covs_s1, label="S1: No landmark")
    if "exposure" in model_s1.params.index:
        sensitivity_results.append({
            "Analysis": "No landmark (full MV compliance)",
            "N": int(model_s1.nobs),
            "Coefficient": model_s1.params["exposure"],
            "CI_low": model_s1.conf_int().loc["exposure", 0],
            "CI_high": model_s1.conf_int().loc["exposure", 1],
            "p_value": model_s1.pvalues["exposure"],
        })

    # --- S2: Exclude deaths ---
    print("\n--- S2: Exclude patients who died within 28 days ---")
    df_s2 = df_full[df_full["mv_duration_hours"] >= 72].copy()
    df_s2["exposure"] = df_s2["compliance_72h"]
    df_s2 = df_s2[~df_s2["death_within_28d"]].copy()
    if len(df_s2) > 20:
        df_s2, covs_s2 = prepare_covariates(df_s2)
        model_s2, _ = run_linear_regression(df_s2, covs_s2, label="S2: Exclude deaths")
        if "exposure" in model_s2.params.index:
            sensitivity_results.append({
                "Analysis": "Exclude 28-day deaths",
                "N": int(model_s2.nobs),
                "Coefficient": model_s2.params["exposure"],
                "CI_low": model_s2.conf_int().loc["exposure", 0],
                "CI_high": model_s2.conf_int().loc["exposure", 1],
                "p_value": model_s2.pvalues["exposure"],
            })

    # --- S3: Assessments per day as exposure ---
    print("\n--- S3: Assessments per day as exposure ---")
    df_s3 = df_full[df_full["mv_duration_hours"] >= 72].copy()
    df_s3["exposure"] = df_s3["assessments_per_day_72h"]
    if len(df_s3) > 20:
        df_s3, covs_s3 = prepare_covariates(df_s3)
        model_s3, _ = run_linear_regression(df_s3, covs_s3, label="S3: Assessments/day")
        if "exposure" in model_s3.params.index:
            sensitivity_results.append({
                "Analysis": "Exposure = assessments/day (72h)",
                "N": int(model_s3.nobs),
                "Coefficient": model_s3.params["exposure"],
                "CI_low": model_s3.conf_int().loc["exposure", 0],
                "CI_high": model_s3.conf_int().loc["exposure", 1],
                "p_value": model_s3.pvalues["exposure"],
            })

    # --- S4: 48h landmark ---
    print("\n--- S4: 48h landmark ---")
    df_s4 = df_full[df_full["mv_duration_hours"] >= 48].copy()
    df_s4["exposure"] = df_s4["compliance_rate"]  # Approximate with overall
    if len(df_s4) > 20:
        df_s4, covs_s4 = prepare_covariates(df_s4)
        model_s4, _ = run_linear_regression(df_s4, covs_s4, label="S4: 48h landmark")
        if "exposure" in model_s4.params.index:
            sensitivity_results.append({
                "Analysis": "48h landmark",
                "N": int(model_s4.nobs),
                "Coefficient": model_s4.params["exposure"],
                "CI_low": model_s4.conf_int().loc["exposure", 0],
                "CI_high": model_s4.conf_int().loc["exposure", 1],
                "p_value": model_s4.pvalues["exposure"],
            })

    # --- S5: Continuous fentanyl only ---
    print("\n--- S5: Continuous fentanyl only ---")
    df_s5 = df_full[
        (df_full["mv_duration_hours"] >= 72)
        & (df_full["fentanyl_type"].isin(["continuous", "both"]))
    ].copy()
    df_s5["exposure"] = df_s5["compliance_72h"]
    if len(df_s5) > 20:
        df_s5, covs_s5 = prepare_covariates(df_s5)
        model_s5, _ = run_linear_regression(df_s5, covs_s5, label="S5: Continuous fent only")
        if "exposure" in model_s5.params.index:
            sensitivity_results.append({
                "Analysis": "Continuous fentanyl only",
                "N": int(model_s5.nobs),
                "Coefficient": model_s5.params["exposure"],
                "CI_low": model_s5.conf_int().loc["exposure", 0],
                "CI_high": model_s5.conf_int().loc["exposure", 1],
                "p_value": model_s5.pvalues["exposure"],
            })

    sens_df = pd.DataFrame(sensitivity_results)
    if len(sens_df) > 0:
        sens_df["Coefficient"] = sens_df["Coefficient"].round(3)
        sens_df["CI_low"] = sens_df["CI_low"].round(3)
        sens_df["CI_high"] = sens_df["CI_high"].round(3)
        sens_df["p_value"] = sens_df["p_value"].round(4)
    save_table(sens_df, "table6_sensitivity")
    print("\n  Sensitivity results saved.")

    return sens_df


# ──────────────────────────────────────────────
# Step 6: Figures
# ──────────────────────────────────────────────
def plot_cumulative_incidence(df):
    """Cumulative incidence curves for extubation and death by compliance quartile."""
    print("\nStep 6a: Plotting cumulative incidence by compliance tertile...")

    df = df[df["time_to_extubation_days"] > 0].copy()
    all_labels = ["T1 (lowest)", "T2", "T3 (highest)"]
    all_colors = ["#d32f2f", "#f57c00", "#1976d2"]
    n_bins = pd.qcut(df["exposure"], q=3, duplicates="drop").cat.categories.size
    labels = all_labels[:n_bins]
    colors = all_colors[:n_bins]
    df["compliance_tertile"] = pd.qcut(
        df["exposure"], q=3, labels=labels, duplicates="drop",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xlim = min(df["time_to_extubation_days"].quantile(0.95), 25)

    for event_idx, (event_val, event_label, ax) in enumerate([
        (1, "Extubation", axes[0]),
        (2, "Death", axes[1]),
    ]):
        for tertile, color in zip(
            df["compliance_tertile"].cat.categories, colors
        ):
            mask = df["compliance_tertile"] == tertile
            if mask.sum() == 0:
                continue

            aj = AalenJohansenFitter(calculate_variance=False)
            aj.fit(
                durations=df.loc[mask, "time_to_extubation_days"],
                event_observed=df.loc[mask, "event_type"],
                event_of_interest=event_val,
            )
            aj.plot(ax=ax, color=color, linewidth=2,
                    label=f"{tertile} (n={mask.sum()})")

        ax.set_xlabel("Days from 72h Landmark")
        ax.set_ylabel(f"Cumulative Incidence of {event_label}")
        ax.set_title(f"{event_label}")
        ax.legend(title="NVPS Compliance", fontsize=8)
        ax.set_xlim(0, xlim)

    fig.suptitle("Cumulative Incidence (Competing Risks) by NVPS Compliance Tertile", fontsize=13)
    fig.tight_layout()
    save_figure(fig, "fig8_cumulative_incidence")
    plt.close(fig)


def plot_vfd_by_compliance(df):
    """Box/violin plot of VFD-28 by NVPS compliance tertile."""
    print("Step 6b: Plotting VFD-28 by compliance tertile...")

    df = df.copy()
    all_labels = ["T1\n(lowest)", "T2", "T3\n(highest)"]
    n_bins = pd.qcut(df["exposure"], q=3, duplicates="drop").cat.categories.size
    df["compliance_tertile"] = pd.qcut(
        df["exposure"], q=3, labels=all_labels[:n_bins], duplicates="drop",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df, x="compliance_tertile", y="vfd_28",
        palette="Blues", ax=ax,
    )
    ax.set_xlabel("NVPS 72h Compliance Tertile")
    ax.set_ylabel("Ventilator-Free Days at Day 28")
    ax.set_title("VFD-28 by NVPS Documentation Compliance")

    # Add median annotations
    medians = df.groupby("compliance_tertile")["vfd_28"].median()
    for i, med in enumerate(medians):
        ax.text(i, med + 0.5, f"{med:.1f}", ha="center", fontweight="bold", fontsize=10)

    fig.tight_layout()
    save_figure(fig, "fig10_vfd_by_compliance")
    plt.close(fig)


def plot_forest(primary_result, sensitivity_df):
    """Forest plot of exposure effect across analyses."""
    print("Step 6c: Plotting forest plot...")

    # Combine primary and sensitivity
    all_results = []
    if primary_result is not None:
        all_results.append({
            "Analysis": "Primary (72h landmark)",
            "Coefficient": primary_result.params.get("exposure", np.nan),
            "CI_low": primary_result.conf_int().loc["exposure", 0] if "exposure" in primary_result.params.index else np.nan,
            "CI_high": primary_result.conf_int().loc["exposure", 1] if "exposure" in primary_result.params.index else np.nan,
        })

    for _, row in sensitivity_df.iterrows():
        all_results.append({
            "Analysis": row["Analysis"],
            "Coefficient": row["Coefficient"],
            "CI_low": row["CI_low"],
            "CI_high": row["CI_high"],
        })

    if not all_results:
        print("  No results to plot.")
        return

    forest_df = pd.DataFrame(all_results)

    fig, ax = plt.subplots(figsize=(10, max(4, len(forest_df) * 0.8)))
    y_pos = range(len(forest_df))

    ax.errorbar(
        forest_df["Coefficient"], y_pos,
        xerr=[
            forest_df["Coefficient"] - forest_df["CI_low"],
            forest_df["CI_high"] - forest_df["Coefficient"],
        ],
        fmt="o", color="steelblue", capsize=5, markersize=8, linewidth=2,
    )
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df["Analysis"])
    ax.set_xlabel("Coefficient (change in VFD-28 per unit compliance)")
    ax.set_title("Effect of NVPS Documentation Compliance on VFD-28")
    ax.invert_yaxis()

    fig.tight_layout()
    save_figure(fig, "fig9_forest_plot")
    plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("04_aim3_association.py — NVPS Compliance vs VFD-28")
    print("=" * 60)

    # Build dataset
    df_full = build_analysis_dataset()

    # Apply 72h landmark
    df_landmark = apply_landmark(df_full, landmark_hours=72)

    # Prepare covariates
    df_landmark, covariates = prepare_covariates(df_landmark)

    # Primary linear regression
    primary_model, primary_results = run_linear_regression(
        df_landmark, covariates, outcome="vfd_28", label="Primary"
    )
    save_table(primary_results, "table4_primary_regression")

    # Cause-specific hazard models (competing risks — primary)
    cph_extub, res_extub, cph_death, res_death = run_cause_specific_models(
        df_landmark, covariates
    )
    save_table(res_extub, "table5_cause_specific_extubation")
    save_table(res_death, "table5b_cause_specific_death")

    # Fine-Gray subdistribution hazard (sensitivity)
    cph_fg, res_fg = run_fine_gray(df_landmark, covariates)
    save_table(res_fg, "table5c_fine_gray")

    # Sensitivity analyses
    sens_df = run_sensitivity_analyses(df_full, covariates)

    # Figures
    plot_cumulative_incidence(df_landmark)
    plot_vfd_by_compliance(df_landmark)
    plot_forest(primary_model, sens_df)

    print("\n" + "=" * 60)
    print("DONE. Aim 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
