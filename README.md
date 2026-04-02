# Fentanyl Sedation & NVPS Documentation in Mechanically Ventilated ICU Patients

Analysis of fentanyl use for sedation and pain assessment documentation patterns in mechanically ventilated ICU patients using the [Common Longitudinal ICU Format (CLIF)](https://clif-consortium.github.io/website/).

## Study Overview

This project examines fentanyl dosing practices and Nonverbal Pain Scale (NVPS) documentation in adult ICU patients on invasive mechanical ventilation at the University of Chicago Medical Center (UCMC).

### Aims

1. **Fentanyl dosing characterization** — Describe continuous and intermittent fentanyl use patterns during mechanical ventilation
2. **NVPS documentation patterns** — Evaluate NVPS scoring frequency, compliance, and the prevalence of zero scores
3. **Association with outcomes** — Assess the relationship between NVPS documentation compliance and ventilator-free days (VFD-28) using a competing risks framework

### QI Analyses

- NVPS score impact on fentanyl dose changes
- NVPS documentation before fentanyl boluses
- NVPS-RASS concordance (are zero pain scores plausible given sedation depth?)
- Unit-level NVPS patterns by ICU type
- NVPS score distribution trends by year
- Fentanyl dose increase justification by NVPS and RASS scores

## CLIF Tables Used

- `clif_medication_admin_continuous` — fentanyl infusion data
- `clif_medication_admin_intermittent` — fentanyl bolus data
- `clif_respiratory_support` — mechanical ventilation episodes
- `clif_patient_assessments` — NVPS and RASS scores
- `clif_hospitalization` — admission/discharge info
- `clif_patient` — demographics
- `clif_adt` — ICU location/type
- `clif_vitals` — vital signs (weight for dose conversion)
- `clif_labs` — laboratory values (SOFA computation)

## Scripts

| Script | Description |
|--------|-------------|
| `code/01_build_cohort.py` | Build cohort: adult MV patients with fentanyl, compute SOFA and VFD-28 |
| `code/02_aim1_fentanyl.py` | Aim 1: Fentanyl dosing characterization |
| `code/03_aim2_nvps.py` | Aim 2: NVPS documentation patterns and compliance |
| `code/04_aim3_association.py` | Aim 3: NVPS compliance vs VFD-28 (competing risks) |
| `code/05_qi_analyses.py` | QI: NVPS-action linkage, bolus assessment, NVPS-RASS concordance |
| `code/06_unit_level_nvps.py` | QI: NVPS patterns stratified by ICU type |
| `code/07_nvps_by_year.py` | QI: NVPS score distribution by year |
| `code/08_dose_increase_justification.py` | QI: Dose increase justification by NVPS score |
| `code/09_dose_increase_rass.py` | QI: Dose increase justification by RASS score |
| `code/utils.py` | Shared utilities for data loading and output |

## Setup

1. Clone the repo
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install clifpy pandas pyarrow matplotlib seaborn lifelines statsmodels scipy tableone
   ```
4. Update `config.json` with your local CLIF data path and desired output path
5. Run scripts in order (`01` through `09`)

## Configuration

Edit `config.json` to set:
- `clif_data_path` — path to your CLIF 2.1.0 parquet files
- `output_path` — where tables and figures are saved
