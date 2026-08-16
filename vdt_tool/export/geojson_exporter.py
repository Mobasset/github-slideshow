"""
GeoJSON exporter: outputs all samples as a FeatureCollection with
all metrics as feature properties.
"""

import io
import json
import logging
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns to include as GeoJSON properties
PROPERTY_COLUMNS = [
    "timestamp_ms", "cell_id", "pci", "earfcn", "rsrp_dbm", "rsrq_db",
    "sinr_db", "cqi", "ta_us", "throughput_dl_mbps", "throughput_ul_mbps",
    "ho_attempt", "ho_success", "ho_failure", "rlf_flag", "scg_failure",
    "a3_trigger", "a5_trigger", "event_type", "event_cause", "rsrp_class",
    "source_file",
]


def _safe_val(v):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def export_geojson(df: pd.DataFrame) -> bytes:
    """
    Export unified DataFrame as GeoJSON FeatureCollection.

    Only rows with valid latitude/longitude are included.
    Returns UTF-8 JSON bytes.
    """
    if df.empty:
        return json.dumps({"type": "FeatureCollection", "features": []}).encode("utf-8")

    mask = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[mask]

    features = []
    for _, row in df_valid.iterrows():
        properties = {}
        for col in PROPERTY_COLUMNS:
            if col in row.index:
                properties[col] = _safe_val(row[col])

        # Timestamp as ISO string for readability
        ts_ms = row.get("timestamp_ms", 0)
        if ts_ms and ts_ms > 0:
            properties["timestamp_iso"] = str(
                pd.Timestamp(int(ts_ms), unit="ms", tz="UTC")
            )

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    round(float(row["longitude"]), 6),
                    round(float(row["latitude"]), 6),
                ],
            },
            "properties": properties,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_features": len(features),
            "generator": "VDT Tool — Virtual Drive Test",
        },
    }

    return json.dumps(geojson, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def export_csv(df: pd.DataFrame) -> bytes:
    """Export unified DataFrame as UTF-8 CSV bytes."""
    if df.empty:
        return b""

    # Drop geometry column if present (geopandas)
    export_df = df.copy()
    if "geometry" in export_df.columns:
        export_df = export_df.drop(columns=["geometry"])

    # Convert timestamp to ISO string
    if "timestamp_utc" in export_df.columns:
        export_df["timestamp_utc"] = export_df["timestamp_utc"].astype(str)

    buf = io.StringIO()
    export_df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue().encode("utf-8")
