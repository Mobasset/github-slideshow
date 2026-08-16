"""
Colored dot map layer: each GPS sample = circle marker colored by RSRP bin.
Supports Douglas-Peucker decimation above 50k samples for performance.
"""

import logging
import math
from typing import Optional, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DECIMATION_THRESHOLD = 50_000


def _douglas_peucker_indices(points: np.ndarray, epsilon: float) -> List[int]:
    """
    Return indices of points that survive Douglas-Peucker simplification.
    points: Nx2 array of (lat, lon).
    """
    if len(points) <= 2:
        return list(range(len(points)))

    def _rdp(start: int, end: int) -> List[int]:
        if end - start <= 1:
            return [start, end]
        # Find the point with max distance from the line start→end
        p1 = points[start]
        p2 = points[end]
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            dists = np.linalg.norm(points[start:end+1] - p1, axis=1)
        else:
            # Perpendicular distance
            t = np.dot(points[start:end+1] - p1, line_vec) / (line_len ** 2)
            proj = np.outer(t, line_vec) + p1
            dists = np.linalg.norm(points[start:end+1] - proj, axis=1)

        max_idx = np.argmax(dists) + start
        max_dist = dists[max_idx - start]

        if max_dist > epsilon:
            left = _rdp(start, max_idx)
            right = _rdp(max_idx, end)
            return left[:-1] + right
        else:
            return [start, end]

    return _rdp(0, len(points) - 1)


def decimate_samples(df: pd.DataFrame, threshold: int = DECIMATION_THRESHOLD) -> pd.DataFrame:
    """
    Reduce sample count to threshold using Douglas-Peucker simplification
    on the lat/lon trajectory.
    """
    if len(df) <= threshold:
        return df

    # Only decimate rows that have valid GPS
    has_gps = df["latitude"].notna() & df["longitude"].notna()
    df_gps = df[has_gps].copy()
    df_no_gps = df[~has_gps]

    if len(df_gps) <= threshold:
        return df

    points = df_gps[["latitude", "longitude"]].values.astype(float)

    # Scale epsilon proportionally to achieve ~threshold points
    # Start with small epsilon and scale up
    target = int(threshold * 0.9)
    epsilon = 0.00001
    for _ in range(20):
        indices = _douglas_peucker_indices(points, epsilon)
        if len(indices) <= target:
            break
        epsilon *= 2.0

    df_decimated = df_gps.iloc[indices]
    result = pd.concat([df_decimated, df_no_gps]).sort_values("timestamp_ms")
    logger.info(
        "Decimated %d → %d samples (%.1f%%)",
        len(df),
        len(result),
        100 * len(result) / len(df),
    )
    return result.reset_index(drop=True)


def build_dot_layer_data(
    df: pd.DataFrame,
    radius: int = 5,
    opacity: float = 0.75,
    decimate: bool = True,
) -> List[dict]:
    """
    Build a list of marker dicts suitable for Plotly scattermapbox or
    Folium CircleMarker rendering.

    Each dict has:
      lat, lon, color, rsrp, rsrq, sinr, cqi, cell_id, timestamp, event_type
    """
    if df.empty:
        return []

    # Filter to rows with valid GPS
    mask = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[mask].copy()

    if decimate and len(df_valid) > DECIMATION_THRESHOLD:
        df_valid = decimate_samples(df_valid)

    markers = []
    for _, row in df_valid.iterrows():
        color = row.get("rsrp_color", "#888888")
        rsrp = row.get("rsrp_dbm", float("nan"))
        rsrq = row.get("rsrq_db", float("nan"))
        sinr = row.get("sinr_db", float("nan"))
        cqi = row.get("cqi", float("nan"))
        cell_id = row.get("cell_id", "")
        pci = row.get("pci", "")
        event_type = row.get("event_type", "")

        ts = row.get("timestamp_utc", "")
        if hasattr(ts, "strftime"):
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            ts_str = str(ts)

        tooltip = (
            f"<b>{ts_str}</b><br>"
            f"RSRP: {rsrp:.1f} dBm | RSRQ: {rsrq:.1f} dB<br>"
            f"SINR: {sinr:.1f} dB | CQI: {cqi}<br>"
            f"Cell: {cell_id} | PCI: {pci}<br>"
            f"Event: {event_type}"
        ) if not (isinstance(rsrp, float) and math.isnan(rsrp)) else (
            f"<b>{ts_str}</b><br>Cell: {cell_id} | Event: {event_type}"
        )

        markers.append(
            {
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "color": color,
                "rsrp": float(rsrp) if not (isinstance(rsrp, float) and math.isnan(rsrp)) else None,
                "rsrq": float(rsrq) if not (isinstance(rsrq, float) and math.isnan(rsrq)) else None,
                "sinr": float(sinr) if not (isinstance(sinr, float) and math.isnan(sinr)) else None,
                "cqi": float(cqi) if not (isinstance(cqi, float) and math.isnan(cqi)) else None,
                "cell_id": str(cell_id),
                "pci": str(pci),
                "timestamp": ts_str,
                "event_type": str(event_type),
                "tooltip": tooltip,
                "radius": radius,
                "opacity": opacity,
            }
        )

    return markers


def build_plotly_scatter(df: pd.DataFrame, mapbox_token: str = "") -> dict:
    """
    Build Plotly scattermapbox trace dict from the sample DataFrame.
    Returns a dict ready to pass to go.Scattermapbox.
    """
    if df.empty:
        return {}

    markers = build_dot_layer_data(df)
    if not markers:
        return {}

    lats = [m["lat"] for m in markers]
    lons = [m["lon"] for m in markers]
    colors = [m["color"] for m in markers]
    texts = [m["tooltip"] for m in markers]
    custom = [
        [m["rsrp"], m["rsrq"], m["sinr"], m["cell_id"], m["event_type"]]
        for m in markers
    ]

    return {
        "type": "scattermapbox",
        "lat": lats,
        "lon": lons,
        "mode": "markers",
        "marker": {
            "size": 7,
            "color": colors,
            "opacity": 0.75,
        },
        "text": texts,
        "hovertemplate": "%{text}<extra></extra>",
        "customdata": custom,
        "name": "RSRP Samples",
    }
