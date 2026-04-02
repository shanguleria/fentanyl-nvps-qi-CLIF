"""Shared utilities for fentanyl/NVPS analysis."""
import json
import os
from pathlib import Path

import pandas as pd

# Project root is one level up from code/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """Load config.json and return data_path, output_path."""
    config_path = PROJECT_ROOT / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config["clif_data_path"], config["output_path"]


def load_clif_table(table_name, columns=None, filters=None):
    """Load a CLIF parquet table with optional column selection and row filters.

    Parameters
    ----------
    table_name : str
        Table name without prefix/suffix, e.g. 'respiratory_support'.
    columns : list[str], optional
        Columns to read. None reads all.
    filters : list[tuple], optional
        PyArrow filter pushdown, e.g. [('med_category', '==', 'fentanyl')].

    Returns
    -------
    pd.DataFrame
    """
    data_path, _ = load_config()
    file_path = os.path.join(data_path, f"clif_{table_name}.parquet")
    return pd.read_parquet(file_path, columns=columns, filters=filters)


def output_dir(subdir=""):
    """Return output directory path, creating it if needed."""
    _, out = load_config()
    d = Path(out) / subdir if subdir else Path(out)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(df, name):
    """Save a DataFrame as CSV to output/tables/."""
    path = output_dir("tables") / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved table: {path}")


def save_intermediate(df, name):
    """Save a DataFrame as parquet to output/intermediate/."""
    path = output_dir("intermediate") / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved intermediate: {path}")


def load_intermediate(name):
    """Load a parquet file from output/intermediate/."""
    path = output_dir("intermediate") / f"{name}.parquet"
    return pd.read_parquet(path)


def save_figure(fig, name):
    """Save a matplotlib figure to output/figures/ as PDF and PNG."""
    for ext in ["pdf", "png"]:
        path = output_dir("figures") / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved figure: {name}.pdf/.png")


# --- Constants ---

# Fentanyl med_name patterns to EXCLUDE (not true fentanyl)
FENTANYL_EXCLUDE_NAMES = [
    "SUFENTANIL",
    "REMIFENTANIL",
]

# Concurrent sedatives to track
CONCURRENT_SEDATIVES = ["propofol", "dexmedetomidine", "midazolam"]

# MV episode gap threshold (hours) — consecutive IMV rows within this gap
# are considered part of the same episode
MV_GAP_THRESHOLD_HOURS = 24

# Minimum MV duration to include (hours)
MV_MIN_DURATION_HOURS = 4
