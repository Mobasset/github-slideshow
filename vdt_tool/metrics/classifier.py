"""
Signal quality classifier: bins RSRP, RSRQ, SINR into quality categories
with associated colors for map rendering.
"""

import numpy as np
import pandas as pd
from typing import Union

# ─── RSRP bins (3GPP TS 36.133 / field engineering thresholds) ────────────────
RSRP_BINS = [
    (-80.0, float("inf"), "Excellent", "#2ecc71"),
    (-90.0, -80.0, "Good", "#a8d08d"),
    (-100.0, -90.0, "Fair", "#f0a500"),
    (-110.0, -100.0, "Poor", "#e06000"),
    (float("-inf"), -110.0, "Bad", "#c0392b"),
]

# ─── RSRQ bins ────────────────────────────────────────────────────────────────
RSRQ_BINS = [
    (-3.0, float("inf"), "Excellent", "#2ecc71"),
    (-6.0, -3.0, "Good", "#a8d08d"),
    (-9.0, -6.0, "Fair", "#f0a500"),
    (-12.0, -9.0, "Poor", "#e06000"),
    (float("-inf"), -12.0, "Bad", "#c0392b"),
]

# ─── SINR bins ────────────────────────────────────────────────────────────────
SINR_BINS = [
    (20.0, float("inf"), "Excellent", "#2ecc71"),
    (13.0, 20.0, "Good", "#a8d08d"),
    (0.0, 13.0, "Fair", "#f0a500"),
    (-6.0, 0.0, "Poor", "#e06000"),
    (float("-inf"), -6.0, "Bad", "#c0392b"),
]

# Map bin label → integer rank for sorting
BIN_RANK = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1, "Bad": 0, "N/A": -1}

# Folium/CSS color for each bin label
BIN_COLOR: dict = {}
for bins in [RSRP_BINS, RSRQ_BINS, SINR_BINS]:
    for _, _, label, color in bins:
        BIN_COLOR[label] = color


def classify_value(value: float, bins: list) -> tuple:
    """Return (label, color) for a scalar value against a bin list."""
    if np.isnan(value):
        return "N/A", "#888888"
    for low, high, label, color in bins:
        if low <= value < high:
            return label, color
    return "N/A", "#888888"


def classify_rsrp(value: float) -> tuple:
    return classify_value(value, RSRP_BINS)


def classify_rsrq(value: float) -> tuple:
    return classify_value(value, RSRQ_BINS)


def classify_sinr(value: float) -> tuple:
    return classify_value(value, SINR_BINS)


def add_rsrp_class(df: pd.DataFrame, col: str = "rsrp_dbm") -> pd.DataFrame:
    """Add rsrp_class and rsrp_color columns to a DataFrame."""
    if col not in df.columns:
        df["rsrp_class"] = "N/A"
        df["rsrp_color"] = "#888888"
        return df

    def _classify_row(v):
        label, color = classify_rsrp(v)
        return pd.Series({"rsrp_class": label, "rsrp_color": color})

    classified = df[col].apply(classify_rsrp).apply(pd.Series)
    classified.columns = ["rsrp_class", "rsrp_color"]
    return pd.concat([df, classified], axis=1)


def add_all_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Add classification columns for RSRP, RSRQ, SINR."""
    df = add_rsrp_class(df)

    if "rsrq_db" in df.columns:
        clas = df["rsrq_db"].apply(lambda v: classify_rsrq(v)[0])
        df["rsrq_class"] = clas
    else:
        df["rsrq_class"] = "N/A"

    if "sinr_db" in df.columns:
        clas = df["sinr_db"].apply(lambda v: classify_sinr(v)[0])
        df["sinr_class"] = clas
    else:
        df["sinr_class"] = "N/A"

    return df


def coverage_bin_distribution(df: pd.DataFrame) -> dict:
    """Return dict of {bin_label: count} for RSRP bins."""
    if "rsrp_class" not in df.columns:
        df = add_rsrp_class(df)
    counts = df["rsrp_class"].value_counts().to_dict()
    # Ensure all bins present
    for _, _, label, _ in RSRP_BINS:
        if label not in counts:
            counts[label] = 0
    return counts


def rsrp_to_normalized(rsrp_values: pd.Series) -> pd.Series:
    """Normalize RSRP values to 0–1 range for heatmap weighting."""
    # Clamp to -140 to -40 dBm
    clamped = rsrp_values.clip(-140, -40)
    return (clamped - (-140)) / ((-40) - (-140))


def cdf_series(values: pd.Series, n_points: int = 200) -> tuple:
    """Compute CDF of a numeric series. Returns (sorted_values, cdf_probs)."""
    clean = values.dropna().sort_values()
    if clean.empty:
        return np.array([]), np.array([])
    probs = np.linspace(0, 1, len(clean))
    # Downsample to n_points if needed
    if len(clean) > n_points:
        idx = np.round(np.linspace(0, len(clean) - 1, n_points)).astype(int)
        clean = clean.iloc[idx]
        probs = probs[idx]
    return clean.values, probs
