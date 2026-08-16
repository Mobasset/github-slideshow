"""
H3 hexagonal binning layer.

Aggregates signal samples into Uber H3 hex cells and builds a
GeoJSON choropleth for Plotly Choroplethmapbox rendering.
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

H3_AVAILABLE = False
try:
    import h3  # type: ignore
    H3_AVAILABLE = True
except ImportError:
    logger.warning("h3-py not installed — hexbin layer unavailable")

METRIC_LABELS = {
    "rsrp_dbm": "RSRP (dBm)",
    "rsrq_db": "RSRQ (dB)",
    "sinr_db": "SINR (dB)",
    "throughput_dl_mbps": "DL Mbps",
}

COLORSCALE_MAP = {
    "rsrp_dbm": "RdYlGn",
    "rsrq_db": "RdYlGn",
    "sinr_db": "RdBu",
    "throughput_dl_mbps": "Plasma",
}

VRANGE_MAP = {
    "rsrp_dbm": (-130, -60),
    "rsrq_db": (-15, -3),
    "sinr_db": (-10, 30),
    "throughput_dl_mbps": (0, 100),
}


def _hex_boundary_to_geojson(hex_id: str) -> dict:
    """Convert H3 hex ID to GeoJSON polygon."""
    boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
    return {
        "type": "Feature",
        "id": hex_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": [list(boundary)],
        },
        "properties": {"hex_id": hex_id},
    }


def build_hexbin_layer(
    df: pd.DataFrame,
    metric: str = "rsrp_dbm",
    resolution: int = 8,
    agg: str = "median",
    min_samples: int = 3,
) -> Optional[dict]:
    """
    Aggregate samples into H3 hexagons and build a Plotly choropleth trace.

    Args:
        df: Unified GeoDataFrame
        metric: Signal metric column to aggregate
        resolution: H3 resolution (7=~1.2km², 8=~0.46km², 9=~0.17km², 10=~0.07km²)
        agg: Aggregation function ('median', 'mean', 'p5', 'count')
        min_samples: Minimum samples per hex to include

    Returns:
        dict with 'trace' (Plotly Choroplethmapbox) and 'geojson' (FeatureCollection)
    """
    if not H3_AVAILABLE:
        logger.warning("h3-py required for hexbin layer")
        return None

    if df.empty or metric not in df.columns:
        return None

    mask = df["latitude"].notna() & df["longitude"].notna() & df[metric].notna()
    df_valid = df[mask].copy()

    if df_valid.empty:
        return None

    # Assign H3 index to each point
    try:
        df_valid["hex_id"] = df_valid.apply(
            lambda r: h3.geo_to_h3(float(r["latitude"]), float(r["longitude"]), resolution),
            axis=1,
        )
    except Exception as exc:
        logger.warning("H3 geo_to_h3 failed: %s", exc)
        return None

    # Aggregate by hex
    def _agg_func(series: pd.Series) -> float:
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        if agg == "median":
            return float(clean.median())
        elif agg == "mean":
            return float(clean.mean())
        elif agg == "p5":
            return float(np.percentile(clean, 5))
        elif agg == "count":
            return float(len(clean))
        return float(clean.median())

    grouped = df_valid.groupby("hex_id")[metric].agg(_agg_func).reset_index()
    counts = df_valid.groupby("hex_id").size().reset_index(name="sample_count")
    grouped = grouped.merge(counts, on="hex_id")
    grouped = grouped[grouped["sample_count"] >= min_samples]

    if grouped.empty:
        return None

    # Build GeoJSON feature collection
    features = []
    for hex_id in grouped["hex_id"]:
        try:
            features.append(_hex_boundary_to_geojson(hex_id))
        except Exception:
            continue

    geojson = {"type": "FeatureCollection", "features": features}

    # Build Plotly trace
    vmin, vmax = VRANGE_MAP.get(metric, (grouped[metric].min(), grouped[metric].max()))
    colorscale = COLORSCALE_MAP.get(metric, "RdYlGn")
    label = METRIC_LABELS.get(metric, metric)

    trace = {
        "type": "choroplethmapbox",
        "geojson": geojson,
        "locations": grouped["hex_id"].tolist(),
        "z": grouped[metric].tolist(),
        "featureidkey": "id",
        "colorscale": colorscale,
        "zmin": vmin,
        "zmax": vmax,
        "marker": {"opacity": 0.65, "line": {"width": 0.5, "color": "#444"}},
        "colorbar": {
            "title": label,
            "bgcolor": "#16213e",
            "tickfont": {"color": "#ffffff"},
            "titlefont": {"color": "#ffffff"},
            "len": 0.5,
        },
        "text": [
            f"Hex: {row['hex_id']}<br>{label}: {row[metric]:.1f}<br>Samples: {row['sample_count']}"
            for _, row in grouped.iterrows()
        ],
        "hovertemplate": "%{text}<extra></extra>",
        "name": f"HexBin: {label} ({agg})",
    }

    return {"trace": trace, "geojson": geojson, "stats": grouped}


def get_hex_stats(
    df: pd.DataFrame,
    resolution: int = 8,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return per-hex statistics DataFrame for display in UI."""
    if not H3_AVAILABLE or df.empty:
        return pd.DataFrame()

    if metrics is None:
        metrics = ["rsrp_dbm", "rsrq_db", "sinr_db", "throughput_dl_mbps"]

    mask = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[mask].copy()

    if df_valid.empty:
        return pd.DataFrame()

    try:
        df_valid["hex_id"] = df_valid.apply(
            lambda r: h3.geo_to_h3(float(r["latitude"]), float(r["longitude"]), resolution),
            axis=1,
        )
    except Exception as exc:
        logger.warning("H3 index error: %s", exc)
        return pd.DataFrame()

    agg_dict: Dict[str, Any] = {"sample_count": ("hex_id", "count")}
    for m in metrics:
        if m in df_valid.columns:
            agg_dict[f"{m}_median"] = (m, "median")

    result = df_valid.groupby("hex_id").agg(**{k: pd.NamedAgg(*v) for k, v in agg_dict.items()})
    return result.reset_index()
