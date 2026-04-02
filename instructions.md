# How to Run the Fentanyl/NVPS Analysis

This guide walks you through running each analysis script step by step.

## Prerequisites

- macOS with Python 3 installed
- VS Code or Terminal access
- CLIF 2.1.0 data at the path specified in `config.json`

## One-Time Setup

### 1. Open Terminal

In VS Code: **Terminal → New Terminal** (or press `` Ctrl+` ``).

Make sure you're in the project directory:

```bash
cd /Users/shanguleria/Desktop/Research/CLIF/Sedation/patrick_fent/fent_claude
```

### 2. Activate the Python environment

Every time you open a new terminal, run:

```bash
source .venv/bin/activate
```

You'll see `(.venv)` appear at the start of your prompt. This means you're using the project's Python environment.

### 3. Install additional packages

Run this once (copy the whole block):

```bash
pip install matplotlib seaborn lifelines statsmodels scipy tableone
```

This may take a minute. You'll see installation progress.

### 4. Verify everything is installed

```bash
python -c "import clifpy, pandas, matplotlib, seaborn, lifelines, statsmodels, tableone; print('All packages installed!')"
```

If you see `All packages installed!` you're good to go.

---

## Running the Analysis

**Important**: Scripts must be run in order — each one depends on the previous.

### Script 1: Build the Cohort

This identifies mechanically ventilated patients who received fentanyl and computes demographics, comorbidities, and outcomes.

```bash
cd /Users/shanguleria/Desktop/Research/CLIF/Sedation/patrick_fent/fent_claude
python code/01_build_cohort.py
```

**What to expect**:
- Takes 3-10 minutes depending on your machine
- Prints progress at each step (MV episodes, fentanyl filter, demographics, etc.)
- Ends with "DONE. Final cohort: N hospitalizations"

**What it creates**:
- `output/intermediate/cohort.parquet` — the full cohort dataset
- `output/intermediate/mv_episodes.parquet` — MV episode timing
- `output/tables/table1_cohort.csv` — Table 1 (demographics summary)
- `output/tables/cohort_flow.csv` — exclusion counts at each step

**Check it worked**: Open `output/tables/cohort_flow.csv` in Excel or VS Code to see patient counts at each step.

---

### Script 2: Fentanyl Dosing (Aim 1)

This characterizes fentanyl dosing patterns over the course of mechanical ventilation.

```bash
python code/02_aim1_fentanyl.py
```

**What to expect**:
- Takes 5-15 minutes (the hourly trajectory computation is the slow part)
- Prints fentanyl dose statistics and patient counts

**What it creates**:
- `output/tables/table2_fentanyl_summary.csv` — dosing summary statistics
- `output/figures/fig1_fentanyl_trajectory.pdf` — dose over time (median + IQR)
- `output/figures/fig2_fentanyl_dose_distribution.pdf` — starting and peak dose histograms
- `output/figures/fig3_fentanyl_bolus_pattern.pdf` — intermittent bolus timing and doses

---

### Script 3: NVPS Documentation Patterns (Aim 2)

This analyzes when and how regularly NVPS pain scores are documented.

```bash
python code/03_aim2_nvps.py
```

**What to expect**:
- Takes 5-15 minutes
- The per-patient metrics loop may take a few minutes for ~17K patients
- Prints documentation compliance rates and score distribution

**What it creates**:
- `output/tables/table3_nvps_documentation.csv` — documentation metrics summary
- `output/figures/fig4_nvps_gap_distribution.pdf` — gap between assessments + compliance
- `output/figures/fig5_nvps_by_mv_day.pdf` — documentation rate over MV course
- `output/figures/fig6_nvps_by_time_of_day.pdf` — documentation by hour (shift patterns)
- `output/figures/fig7_nvps_score_distribution.pdf` — NVPS score distribution (spoiler: mostly 0)
- `output/intermediate/nvps_metrics.parquet` — per-patient documentation metrics (used by Aim 3)

---

### Script 4: Association Analysis (Aim 3)

This tests whether better NVPS documentation is associated with more ventilator-free days.

```bash
python code/04_aim3_association.py
```

**What to expect**:
- Takes 2-5 minutes
- Prints regression coefficients and hazard ratios
- The key result is the "EXPOSURE" line — the effect of NVPS compliance on VFD-28

**What it creates**:
- `output/tables/table4_primary_regression.csv` — main regression results
- `output/tables/table5_cox_model.csv` — Cox model results (hazard ratios)
- `output/tables/table6_sensitivity.csv` — sensitivity analyses
- `output/figures/fig8_km_by_nvps_quartile.pdf` — Kaplan-Meier by compliance group
- `output/figures/fig9_forest_plot.pdf` — forest plot of all analyses
- `output/figures/fig10_vfd_by_compliance.pdf` — VFD-28 box plot by compliance quartile

---

## Viewing Results

### Tables
All tables are saved as CSV files in `output/tables/`. You can open them in:
- **Excel**: Double-click the file
- **VS Code**: Click the file (install "Rainbow CSV" extension for nicer formatting)

### Figures
All figures are saved as both PDF and PNG in `output/figures/`. You can:
- **Preview in VS Code**: Click any PNG file
- **Open in Preview**: Double-click any PDF file in Finder

---

## Troubleshooting

### "ModuleNotFoundError: No module named '...'"
You forgot to activate the virtual environment. Run:
```bash
source .venv/bin/activate
```
Then try again.

### "FileNotFoundError: ... cohort.parquet"
You need to run the scripts in order. Script 2, 3, and 4 depend on Script 1's output.

### Script seems stuck / taking very long
Some scripts process millions of rows. If it's been printing progress messages, it's still working. Give it up to 20 minutes.

### Memory error
Close other applications. The scripts load large parquet files. If problems persist, let me know and we can optimize to process in chunks.

### "No such file or directory: .../clif_..."
Check that `config.json` has the correct path to your CLIF data directory.

---

## Quick Reference

| Script | Purpose | Runtime | Key Output |
|--------|---------|---------|------------|
| `01_build_cohort.py` | Build cohort | 3-10 min | `table1_cohort.csv` |
| `02_aim1_fentanyl.py` | Fentanyl dosing | 5-15 min | `table2_fentanyl_summary.csv`, trajectory figures |
| `03_aim2_nvps.py` | NVPS patterns | 5-15 min | `table3_nvps_documentation.csv`, pattern figures |
| `04_aim3_association.py` | Association | 2-5 min | `table4-6_*.csv`, KM + forest figures |
