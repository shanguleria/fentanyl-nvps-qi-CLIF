"""
12_html_dashboard.py
Creates a self-contained HTML dashboard with all figures and tables
organized into tabbed sections by analysis aim.

Input:  output/figures/*.png, output/tables/*.csv
Output: output/figure_dashboard.html
"""

import base64
import json
from pathlib import Path

import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.json") as f:
    config = json.load(f)

FIGURES = Path(config["output_path"]) / "figures"
TABLES = Path(config["output_path"]) / "tables"
OUTPUT = Path(config["output_path"]) / "figure_dashboard.html"

# ── Tab definitions ─────────────────────────────────────────────────────────
# Each tab: (id, label, figure_files, table_files)
# Figure/table files are basenames without extension
TABS = [
    (
        "cohort",
        "Cohort",
        ["fig0_consort_diagram"],
        ["table1_cohort"],
    ),
    (
        "aim1",
        "Aim 1: Fentanyl Dosing",
        ["fig1_fentanyl_trajectory", "fig2_fentanyl_dose_distribution",
         "fig3_fentanyl_bolus_pattern"],
        ["table2_fentanyl_summary"],
    ),
    (
        "aim2",
        "Aim 2: NVPS Patterns",
        ["fig4_nvps_gap_distribution", "fig5_nvps_by_mv_day",
         "fig6_nvps_by_time_of_day", "fig7_nvps_score_distribution"],
        ["table3_nvps_documentation"],
    ),
    (
        "aim3",
        "Aim 3: Association",
        ["fig8_cumulative_incidence", "fig9_forest_plot",
         "fig10_vfd_by_compliance"],
        ["table4_primary_regression", "table5_cause_specific_extubation",
         "table5_cox_model", "table5b_cause_specific_death",
         "table5c_fine_gray", "table6_sensitivity"],
    ),
    (
        "qi",
        "QI Analyses",
        ["fig11_nvps_action_comparison", "fig12_bolus_nvps_timing",
         "fig13_nvps_by_rass"],
        ["table7_nvps_action", "table8_bolus_assessment",
         "table9_nvps_rass_concordance"],
    ),
    (
        "unit",
        "Unit-Level NVPS",
        ["fig14_nonzero_rate_by_icu_type", "fig15_compliance_by_icu_type",
         "fig16_nvps_rass_by_icu_type"],
        ["table10_nvps_by_icu_type", "table11_compliance_by_icu_type",
         "table12_nvps_rass_by_icu_type"],
    ),
    (
        "year",
        "NVPS by Year",
        ["fig17_nvps_scores_by_year"],
        ["table_nvps_scores_by_year"],
    ),
    (
        "fent_just",
        "Fent Dose Justification",
        ["fig18_dose_increase_justification", "fig19_dose_increase_by_icu",
         "fig22_nvps_justification_by_year"],
        ["table13_dose_increase_justification", "table14_dose_increase_by_icu"],
    ),
    (
        "rass_just",
        "RASS Dose Justification",
        ["fig20_dose_increase_rass_justification",
         "fig21_dose_increase_rass_by_icu",
         "fig23_rass_justification_by_year"],
        ["table15_dose_increase_rass_justification",
         "table16_dose_increase_rass_by_icu"],
    ),
    (
        "propofol",
        "Propofol",
        ["fig24_propofol_trajectory", "fig25_propofol_dose_distribution",
         "fig26_propofol_dose_increase_rass",
         "fig27_propofol_dose_increase_rass_by_icu",
         "fig28_propofol_rass_justification_by_year"],
        ["table17_propofol_summary", "table18_propofol_dose_increase_rass",
         "table19_propofol_dose_increase_rass_by_icu"],
    ),
]


# ── Helpers ─────────────────────────────────────────────────────────────────
def img_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _clean_table1(path: Path) -> str:
    """Format tableone output: drop Missing, clean variable names,
    collapse binary True/False rows to show only the True row."""
    df = pd.read_csv(path)
    # Drop the Missing column
    if "Missing" in df.columns:
        df = df.drop(columns=["Missing"])

    # Pretty-print the variable/stat column (level_0)
    label_map = {
        "age_at_admission": "Age at Admission",
        "mv_duration_hours": "MV Duration (hours)",
        "mv_duration_days": "MV Duration (days)",
        "sofa_total": "Admission SOFA",
        "vfd_28": "Ventilator-Free Days (28d)",
        "fentanyl_type": "Fentanyl Type",
        "in_hospital_death": "In-Hospital Death",
        "death_within_28d": "Death Within 28 Days",
        "sex_category": "Sex",
        "race_category": "Race",
        "ethnicity_category": "Ethnicity",
        "received_propofol": "Received Propofol",
        "received_dexmedetomidine": "Received Dexmedetomidine",
        "received_midazolam": "Received Midazolam",
    }

    def clean_level0(val):
        if pd.isna(val):
            return ""
        for raw, nice in label_map.items():
            if raw in str(val):
                return str(val).replace(raw, nice)
        return str(val)

    df["level_0"] = df["level_0"].apply(clean_level0)
    df = df.rename(columns={"level_0": "Variable", "level_1": "Category"})
    df["Category"] = df["Category"].fillna("")

    # For binary True/False variables, keep only the True row and
    # move the value up to the Variable row (drop Category column value)
    binary_vars = [
        "In-Hospital Death",
        "Death Within 28 Days",
        "Received Propofol",
        "Received Dexmedetomidine",
        "Received Midazolam",
    ]
    drop_idx = []
    for bv in binary_vars:
        mask = df["Variable"].str.contains(bv, na=False)
        rows = df[mask]
        false_rows = rows[rows["Category"] == "False"]
        drop_idx.extend(false_rows.index.tolist())
        # Clear the "True" category text — the variable name is sufficient
        true_rows = rows[rows["Category"] == "True"]
        df.loc[true_rows.index, "Category"] = ""

    df = df.drop(index=drop_idx).reset_index(drop=True)

    return df.fillna("").to_html(index=False, classes="results-table", border=0)


ICU_LABELS = {
    "medical_icu": "Medical ICU",
    "surgical_icu": "Surgical ICU",
    "mixed_cardiothoracic_icu": "Cardiothoracic ICU",
    "mixed_neuro_icu": "Neuro ICU",
    "general_icu": "General ICU",
    "burn_icu": "Burn ICU",
}


def _clean_icu_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace snake_case ICU type values with readable labels."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace(ICU_LABELS)
    return df


def csv_to_html(path: Path, name: str = "") -> str:
    if name == "table1_cohort":
        return _clean_table1(path)
    df = pd.read_csv(path)
    df = _clean_icu_names(df)
    return df.fillna("").to_html(index=False, classes="results-table", border=0)


def pretty_name(basename: str) -> str:
    """Convert 'fig1_fentanyl_trajectory' → 'Fig 1 — Fentanyl Trajectory'."""
    parts = basename.split("_", 1)
    if len(parts) == 2:
        prefix = parts[0].replace("fig", "Fig ").replace("table", "Table ")
        label = parts[1].replace("_", " ").title()
        return f"{prefix} — {label}"
    return basename.replace("_", " ").title()


# ── Build HTML sections ────────────────────────────────────────────────────
def build_tab_content(fig_names: list[str], table_names: list[str]) -> str:
    sections = []

    # Figures
    for name in fig_names:
        path = FIGURES / f"{name}.png"
        if path.exists():
            uri = img_to_data_uri(path)
            sections.append(
                f'<div class="figure-block">'
                f'<h3>{pretty_name(name)}</h3>'
                f'<img src="{uri}" alt="{name}">'
                f'</div>'
            )
        else:
            sections.append(
                f'<div class="missing">Missing: {name}.png</div>'
            )

    # Tables
    for name in table_names:
        path = TABLES / f"{name}.csv"
        if path.exists():
            html_table = csv_to_html(path, name=name)
            sections.append(
                f'<div class="section">'
                f'<h3>{pretty_name(name)}</h3>'
                f'{html_table}'
                f'</div>'
            )
        else:
            sections.append(
                f'<div class="missing">Missing: {name}.csv</div>'
            )

    return "\n".join(sections)


# ── Assemble full HTML ──────────────────────────────────────────────────────
def build_dashboard() -> str:
    # Tab buttons
    tab_buttons = []
    for i, (tid, label, _, _) in enumerate(TABS):
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab-btn{active}" '
            f'onclick="switchTab(\'{tid}\')" '
            f'id="btn-{tid}">{label}</button>'
        )
    tabs_bar = "\n".join(tab_buttons)

    # Tab panels
    tab_ids_js = ", ".join(f'"{t[0]}"' for t in TABS)
    panels = []
    for i, (tid, label, figs, tbls) in enumerate(TABS):
        display = "block" if i == 0 else "none"
        content = build_tab_content(figs, tbls)
        panels.append(
            f'<div class="tab-panel" id="panel-{tid}" '
            f'style="display:{display};">'
            f'<h2>{label}</h2>'
            f'{content}'
            f'</div>'
        )
    panels_html = "\n".join(panels)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fentanyl / NVPS CLIF Analysis Dashboard</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: #f0f2f5;
    margin: 0;
    padding: 20px;
  }}
  .container {{
    max-width: 1400px;
    margin: 0 auto;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 24px;
  }}
  h1 {{
    margin: 0 0 4px 0;
    font-size: 1.6em;
    color: #1a1a1a;
  }}
  .subtitle {{
    color: #666;
    margin: 0 0 20px 0;
    font-size: 0.95em;
  }}
  .tab-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0;
    margin-bottom: 24px;
  }}
  .tab-btn {{
    padding: 8px 16px;
    border: none;
    background: #f0f2f5;
    color: #333;
    font-size: 0.85em;
    font-weight: 500;
    cursor: pointer;
    border-radius: 6px 6px 0 0;
    transition: background 0.15s, color 0.15s;
  }}
  .tab-btn:hover {{
    background: #d0d5dd;
  }}
  .tab-btn.active {{
    background: #2563eb;
    color: #fff;
  }}
  .tab-panel {{
    padding: 0;
  }}
  .tab-panel h2 {{
    font-size: 1.3em;
    color: #1a1a1a;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 8px;
    margin-top: 0;
  }}
  .figure-block {{
    margin-bottom: 32px;
  }}
  .figure-block h3 {{
    font-size: 1em;
    color: #444;
    margin-bottom: 8px;
  }}
  .figure-block img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
  }}
  .section {{
    margin-bottom: 32px;
  }}
  .section h3 {{
    font-size: 1em;
    color: #444;
    margin-bottom: 8px;
  }}
  .missing {{
    color: #999;
    font-style: italic;
    padding: 12px;
    background: #fafafa;
    border-radius: 4px;
    margin-bottom: 16px;
  }}
  table.results-table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85em;
    margin-bottom: 8px;
  }}
  table.results-table thead th {{
    background: #2563eb;
    color: #fff;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }}
  table.results-table tbody td {{
    padding: 6px 12px;
    border-bottom: 1px solid #e8e8e8;
  }}
  table.results-table tbody tr:nth-child(even) {{
    background: #f9f9f9;
  }}
  table.results-table tbody tr:hover {{
    background: #eef2ff;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Fentanyl / NVPS CLIF Analysis Dashboard</h1>
  <p class="subtitle">29 figures &middot; 24 tables &middot; UCMC 2018-2024 &middot; N = 12,620</p>
  <div class="tab-bar">
    {tabs_bar}
  </div>
  {panels_html}
</div>
<script>
const TABS = [{tab_ids_js}];
function switchTab(id) {{
  TABS.forEach(function(t) {{
    document.getElementById('panel-' + t).style.display = 'none';
    document.getElementById('btn-' + t).classList.remove('active');
  }});
  document.getElementById('panel-' + id).style.display = 'block';
  document.getElementById('btn-' + id).classList.add('active');
}}
</script>
</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building HTML dashboard...")
    html = build_dashboard()
    OUTPUT.write_text(html)
    print(f"Dashboard saved to: {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024 / 1024:.1f} MB")
