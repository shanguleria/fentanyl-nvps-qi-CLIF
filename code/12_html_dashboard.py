"""
12_html_dashboard.py
Creates a self-contained HTML dashboard with all figures and tables
organized into grouped tabbed sections by analysis aim.

Styled per ~/.claude/templates/dashboard_design_guide.md (2026-04-15).

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
# Each tab: (id, label, group, figure_files, table_files)
# Tabs with the same group key render under a shared uppercase group label.
TABS = [
    (
        "cohort", "Cohort", "COHORT",
        ["fig0_consort_diagram"],
        ["table1_cohort"],
    ),
    (
        "aim1", "Fentanyl Dosing", "DOSING PATTERNS",
        ["fig1_fentanyl_trajectory", "fig2_fentanyl_dose_distribution",
         "fig3_fentanyl_bolus_pattern"],
        ["table2_fentanyl_summary"],
    ),
    (
        "propofol_dosing", "Propofol Dosing", "DOSING PATTERNS",
        ["fig24_propofol_trajectory", "fig25_propofol_dose_distribution"],
        ["table17_propofol_summary"],
    ),
    (
        "aim2", "NVPS Patterns", "SCORING PATTERNS",
        ["fig4_nvps_gap_distribution", "fig5_nvps_by_mv_day",
         "fig6_nvps_by_time_of_day", "fig7_nvps_score_distribution"],
        ["table3_nvps_documentation"],
    ),
    (
        "unit", "Unit-Level NVPS", "SCORING PATTERNS",
        ["fig14_nonzero_rate_by_icu_type", "fig15_compliance_by_icu_type",
         "fig16_nvps_rass_by_icu_type"],
        ["table10_nvps_by_icu_type", "table11_compliance_by_icu_type",
         "table12_nvps_rass_by_icu_type"],
    ),
    (
        "year", "NVPS by Year", "SCORING PATTERNS",
        ["fig17_nvps_scores_by_year"],
        ["table_nvps_scores_by_year"],
    ),
    (
        "qi", "NVPS Assessments", "QI",
        ["fig11_nvps_action_comparison", "fig12_bolus_nvps_timing",
         "fig13_nvps_by_rass"],
        ["table7_nvps_action", "table8_bolus_assessment",
         "table9_nvps_rass_concordance"],
    ),
    (
        "fent_just", "Fentanyl (NVPS)", "QI",
        ["fig18_dose_increase_justification", "fig19_dose_increase_by_icu",
         "fig22_nvps_justification_by_year"],
        ["table13_dose_increase_justification", "table14_dose_increase_by_icu"],
    ),
    (
        "rass_just", "Fentanyl (RASS)", "QI",
        ["fig20_dose_increase_rass_justification",
         "fig21_dose_increase_rass_by_icu",
         "fig23_rass_justification_by_year"],
        ["table15_dose_increase_rass_justification",
         "table16_dose_increase_rass_by_icu"],
    ),
    (
        "propofol_just", "Propofol (RASS)", "QI",
        ["fig26_propofol_dose_increase_rass",
         "fig27_propofol_dose_increase_rass_by_icu",
         "fig28_propofol_rass_justification_by_year"],
        ["table18_propofol_dose_increase_rass",
         "table19_propofol_dose_increase_rass_by_icu"],
    ),
    (
        "aim3", "NVPS Compliance \u2192 VFD-28", "ASSOCIATION",
        ["fig8_cumulative_incidence", "fig9_forest_plot",
         "fig10_vfd_by_compliance"],
        ["table4_primary_regression", "table5_cause_specific_extubation",
         "table5_cox_model", "table5b_cause_specific_death",
         "table5c_fine_gray", "table6_sensitivity"],
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
    if "Missing" in df.columns:
        df = df.drop(columns=["Missing"])

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


def build_tab_bar() -> str:
    """Render grouped tab bar: one column per group, preserving TABS order."""
    groups: list[tuple[str, list]] = []
    for tab in TABS:
        group_label = tab[2]
        if groups and groups[-1][0] == group_label:
            groups[-1][1].append(tab)
        else:
            groups.append((group_label, [tab]))

    first_tab_id = TABS[0][0]
    group_html = []
    for group_label, group_tabs in groups:
        btn_html = []
        for tid, label, _group, _f, _t in group_tabs:
            active = " active" if tid == first_tab_id else ""
            btn_html.append(
                f'<button class="tab-btn{active}" '
                f'onclick="switchTab(\'{tid}\')" '
                f'id="btn-{tid}">{label}</button>'
            )
        group_html.append(
            '<div class="tab-group">'
            f'<div class="tab-group-label">{group_label}</div>'
            f'<div class="tab-group-buttons">{"".join(btn_html)}</div>'
            '</div>'
        )
    return "\n".join(group_html)


# ── Assemble full HTML ──────────────────────────────────────────────────────
def build_dashboard() -> str:
    tabs_bar = build_tab_bar()

    tab_ids_js = ", ".join(f'"{t[0]}"' for t in TABS)
    panels = []
    first_tab_id = TABS[0][0]
    for tid, label, _group, figs, tbls in TABS:
        display = "block" if tid == first_tab_id else "none"
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
  :root {{
    --text-primary: #1e293b;
    --text-heading: #0f172a;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --accent: #0f766e;
    --accent-light: #f0fdfa;
    --accent-border: #ccfbf1;
    --header-bg: #f8fafc;
    --header-rule: #334155;
    --row-border: #f1f5f9;
    --row-alt: #fafbfc;
    --row-hover: #f1f5f9;
    --divider: #e2e8f0;
    --missing-bg: #fffbeb;
    --missing-border: #fde68a;
    --missing-text: #92400e;
    --page-bg: #f8f9fa;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Inter, -apple-system, "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.55;
    color: var(--text-primary);
    background: var(--page-bg);
    margin: 0;
    padding: 40px 20px;
  }}
  .container {{
    max-width: 1600px;
    margin: 0 auto;
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06),
                0 1px 2px rgba(15, 23, 42, 0.04);
    padding: 40px 48px;
  }}
  h1 {{
    margin: 0 0 6px 0;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-heading);
  }}
  .subtitle {{
    color: var(--text-secondary);
    margin: 0 0 32px 0;
    font-size: 14px;
  }}
  .tab-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    padding-bottom: 20px;
    margin-bottom: 32px;
    border-bottom: 1px solid var(--divider);
  }}
  .tab-group {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .tab-group-label {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--text-muted);
  }}
  .tab-group-buttons {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }}
  .tab-btn {{
    padding: 8px 16px;
    border: 1px solid var(--divider);
    background: #fff;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border-radius: 6px;
    transition: background 0.12s, color 0.12s, border-color 0.12s,
                box-shadow 0.12s;
  }}
  .tab-btn:hover {{
    background: var(--row-hover);
    border-color: var(--text-muted);
  }}
  .tab-btn.active {{
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
    box-shadow: 0 1px 2px rgba(15, 118, 110, 0.3);
  }}
  .tab-panel h2 {{
    font-size: 19px;
    font-weight: 600;
    color: var(--text-heading);
    padding-bottom: 10px;
    margin: 0 0 28px 0;
    border-bottom: 1px solid var(--divider);
  }}
  .figure-block {{
    margin-bottom: 36px;
  }}
  .figure-block h3,
  .section h3 {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text-heading);
    margin: 0 0 12px 0;
    text-align: left;
  }}
  .figure-block img {{
    display: block;
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08),
                0 1px 2px rgba(15, 23, 42, 0.04);
  }}
  .section {{
    padding: 28px;
    margin-bottom: 48px;
    background: #fff;
    border: 1px solid var(--row-border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
  }}
  .missing {{
    background: var(--missing-bg);
    border: 1px solid var(--missing-border);
    color: var(--missing-text);
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 13px;
    margin-bottom: 16px;
  }}
  .missing::before {{
    content: "— ";
    font-weight: 600;
  }}
  table.results-table {{
    border-collapse: collapse;
    width: auto;
    font-size: 13px;
    color: var(--text-primary);
  }}
  table.results-table thead th {{
    background: var(--header-bg);
    color: var(--text-heading);
    padding: 9px 12px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid var(--header-rule);
    border-top: none;
    border-left: none;
    border-right: none;
  }}
  table.results-table tbody td {{
    padding: 9px 12px;
    text-align: left;
    border-bottom: 1px solid var(--row-border);
    border-left: none;
    border-right: none;
    border-top: none;
  }}
  table.results-table tbody tr:nth-child(even) td {{
    background: var(--row-alt);
  }}
  table.results-table tbody tr:hover td {{
    background: var(--row-hover);
  }}
  table.results-table tbody tr:last-child td {{
    border-bottom: 1px solid var(--text-muted);
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Fentanyl / NVPS CLIF Analysis Dashboard</h1>
  <p class="subtitle">29 figures &middot; 24 tables &middot; UCMC 2018&ndash;2024 &middot; N = 12,620</p>
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
