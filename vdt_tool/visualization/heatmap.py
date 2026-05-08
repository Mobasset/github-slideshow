"""
Weighted heatmap layer builder.

Generates heatmap data arrays weighted by RSRP/RSRQ/SINR/Throughput
for use with Plotly Densitymapbox or Folium HeatMap plugin.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from metrics.classifier import rsrp_to_normalized

logger = logging.getLogger(__name__)


METRIC_WEIGHT_FUNCS = {
    "rsrp_dbm": rsrp_to_normalized,
    "rsrq_db": lambda s: (s.clip(-19.5, 0) - (-19.5)) / 19.5,
    "sinr_db": lambda s: (s.clip(-23, 40) - (-23)) / 63.0,
    "throughput_dl_mbps": lambda s: (s.clip(0, 150) / 150.0),
}

METRIC_DISPLAY_NAMES = {
    "rsrp_dbm": "RSRP (dBm)",
    "rsrq_db": "RSRQ (dB)",
    "sinr_db": "SINR (dB)",
    "throughput_dl_mbps": "DL Throughput (Mbps)",
}


def build_heatmap_data(
    df: pd.DataFrame,
    metric: str = "rsrp_dbm",
    max_points: int = 10_000,
) -> List[List[float]]:
    """
    Build [[lat, lon, weight], ...] list for Folium HeatMap plugin.

    Args:
        df: Unified GeoDataFrame
        metric: Column name to use for weighting
        max_points: Downsample if more points exist

    Returns:
        List of [lat, lon, weight] where weight ∈ [0, 1]
    """
    if df.empty:
        return []

    # Filter valid GPS and metric
    mask = df["latitude"].notna() & df["longitude"].notna()
    if metric in df.columns:
        mask &= df[metric].notna()
    df_valid = df[mask].copy()

    if df_valid.empty:
        return []

    # Downsample
    if len(df_valid) > max_points:
        df_valid = df_valid.sample(n=max_points, random_state=42)

    # Compute weights
    weight_func = METRIC_WEIGHT_FUNCS.get(metric)
    if weight_func and metric in df_valid.columns:
        weights = weight_func(df_valid[metric]).fillna(0).clip(0, 1)
    else:
        weights = pd.Series(np.ones(len(df_valid)), index=df_valid.index)

    result = []
    for (_, row), w in zip(df_valid.iterrows(), weights):
        result.append([float(row["latitude"]), float(row["longitude"]), float(w)])

    return result


def build_plotly_density(
    df: pd.DataFrame,
    metric: str = "rsrp_dbm",
    radius: int = 15,
    max_points: int = 15_000,
) -> dict:
    """
    Build Plotly densitymapbox trace dict.

    Returns dict ready for go.Densitymapbox.
    """
    if df.empty:
        return {}

    mask = df["latitude"].notna() & df["longitude"].notna()
    if metric in df.columns:
        mask &= df[metric].notna()
    df_valid = df[mask].copy()

    if df_valid.empty:
        return {}

    if len(df_valid) > max_points:
        df_valid = df_valid.sample(n=max_points, random_state=42)

    weight_func = METRIC_WEIGHT_FUNCS.get(metric)
    if weight_func and metric in df_valid.columns:
        z = weight_func(df_valid[metric]).fillna(0).clip(0, 1).tolist()
    else:
        z = [1.0] * len(df_valid)

    colorscale_map = {
        "rsrp_dbm": "RdYlGn",
        "rsrq_db": "RdYlGn",
        "sinr_db": "RdBu",
        "throughput_dl_mbps": "plasma",
    }
    colorscale = colorscale_map.get(metric, "RdYlGn")

    return {
        "type": "densitymapbox",
        "lat": df_valid["latitude"].tolist(),
        "lon": df_valid["longitude"].tolist(),
        "z": z,
        "radius": radius,
        "colorscale": colorscale,
        "showscale": True,
        "colorbar": {
            "title": METRIC_DISPLAY_NAMES.get(metric, metric),
            "bgcolor": "#16213e",
            "tickfont": {"color": "#ffffff"},
            "titlefont": {"color": "#ffffff"},
        },
        "name": f"Heatmap: {METRIC_DISPLAY_NAMES.get(metric, metric)}",
        "hoverinfo": "skip",
    }


def get_map_center(df: pd.DataFrame) -> Tuple[float, float]:
    """Return (lat, lon) center of the GPS track."""
    mask = df["latitude"].notna() & df["longitude"].notna()
    if not mask.any():
        return (23.5880, 58.3829)  # Default: Muscat, Oman
    return (
        float(df.loc[mask, "latitude"].median()),
        float(df.loc[mask, "longitude"].median()),
    )
