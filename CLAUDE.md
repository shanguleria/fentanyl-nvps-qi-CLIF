# Fentanyl Sedation CLIF Project

## Project Overview
Analysis of fentanyl use for sedation in mechanically ventilated ICU patients using CLIF (Common Longitudinal ICU Format) data.

## Data
- **CLIF Version**: 2.1.0
- **Data Path**: Configured in `config.json` (do NOT hardcode paths in scripts)
- **Site**: UCMC (University of Chicago Medical Center)
- **Protected Data**: Never read raw parquet files directly — always use Python code to read and print aggregated summaries only

## Key CLIF Tables
- `clif_medication_admin_continuous` — fentanyl infusion data
- `clif_medication_admin_intermittent` — fentanyl bolus data
- `clif_respiratory_support` — mechanical ventilation episodes
- `clif_vitals` — vital signs
- `clif_labs` — laboratory values
- `clif_patient_assessments` — sedation scores (RASS, CAM-ICU, etc.)
- `clif_hospitalization` — admission/discharge info
- `clif_patient` — demographics
- `clif_adt` — ADT (admit/discharge/transfer) events

## Environment
- Python venv at `.venv/`
- Key packages: `clifpy`, `pandas`, `pyarrow`
- Activate: `source .venv/bin/activate`

## Workflow
1. Load config from `config.json`
2. Use `clifpy` to load CLIF tables
3. Filter to relevant cohort (mechanically ventilated patients receiving fentanyl)
4. Perform analysis

## Commands
```bash
source .venv/bin/activate
python <script>.py
```
