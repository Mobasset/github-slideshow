"""
Antenna sector wedge overlay builder.

Draws filled polygon wedges from cell metadata (azimuth, HPBW, ISD radius)
and colors them by signal metric aggregated within each wedge.
"""

import logging
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# WGS84 Earth radius in meters
EARTH_RADIUS_M = 6_371_000.0

DEFAULT_HPBW_DEG = 60.0       # Half-power beamwidth (±30° each side)
DEFAULT_ISD_M = 300.0          # Inter-site distance / coverage radius


def _offset_latlon(lat: float, lon: float, bearing_deg: float, dist_m: float) -> Tuple[float, float]:
    """
    Calculate destination lat/lon given origin, bearing (degrees CW from N), and distance.
    Uses spherical Earth approximation.
    """
    d = dist_m / EARTH_RADIUS_M
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _build_wedge_polygon(
    lat: float,
    lon: float,
    azimuth_deg: float,
    hpbw_deg: float = DEFAULT_HPBW_DEG,
    radius_m: float = DEFAULT_ISD_M,
    n_arc_points: int = 16,
) -> List[Tuple[float, float]]:
    """
    Build wedge polygon as list of (lat, lon) points.

    The wedge starts at (lat, lon), fans out to radius_m,
    spanning azimuth ± hpbw_deg/2.
    """
    half_bw = hpbw_deg / 2.0
    start_bearing = azimuth_deg - half_bw
    end_bearing = azimuth_deg + half_bw

    points = [(lat, lon)]

    for i in range(n_arc_points + 1):
        bearing = start_bearing + (end_bearing - start_bearing) * i / n_arc_points
        pt = _offset_latlon(lat, lon, bearing, radius_m)
        points.append(pt)

    points.append((lat, lon))  # Close the polygon
    return points


def _point_in_wedge(
    pt_lat: float,
    pt_lon: float,
    site_lat: float,
    site_lon: float,
    azimuth_deg: float,
    hpbw_deg: float,
    radius_m: float,
) -> bool:
    """Test if a GPS point falls within a sector wedge."""
    # Distance check
    dlat = math.radians(pt_lat - site_lat)
    dlon = math.radians(pt_lon - site_lon)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(site_lat)) * math.cos(
        math.radians(pt_lat)
    ) * math.sin(dlon / 2) ** 2
    dist = 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))
    if dist > radius_m:
        return False

    # Bearing check
    y = math.sin(math.radians(pt_lon - site_lon)) * math.cos(math.radians(pt_lat))
    x = math.cos(math.radians(site_lat)) * math.sin(math.radians(pt_lat)) - math.sin(
        math.radians(site_lat)
    ) * math.cos(math.radians(pt_lat)) * math.cos(math.radians(pt_lon - site_lon))
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    half_bw = hpbw_deg / 2.0
    diff = abs(((bearing - azimuth_deg) + 180) % 360 - 180)
    return diff <= half_bw


def build_sector_traces(
    cell_meta_df: pd.DataFrame,
    samples_df: Optional[pd.DataFrame] = None,
    color_by: str = "rsrp_dbm",
    hpbw_deg: float = DEFAULT_HPBW_DEG,
    radius_m: float = DEFAULT_ISD_M,
) -> List[dict]:
    """
    Build Plotly Scattermapbox traces for sector wedge overlays.

    Args:
        cell_meta_df: Cell metadata with columns CellID/cell_id, Lat/lat, Lon/lon,
                      Azimuth/azimuth, PCI/pci, Site_Name/site_name, Band/band
        samples_df: Optional signal samples for per-sector metric aggregation
        color_by: Metric column to use for wedge color
        hpbw_deg: Half-power beamwidth in degrees
        radius_m: Coverage radius in meters

    Returns:
        List of Plotly trace dicts (one per sector)
    """
    if cell_meta_df.empty:
        return []

    # Normalize column names
    col_map = {}
    for c in cell_meta_df.columns:
        cl = c.lower().strip()
        if cl in ("cellid", "cell_id"):
            col_map[c] = "cell_id"
        elif cl in ("lat", "latitude"):
            col_map[c] = "lat"
        elif cl in ("lon", "longitude", "lng"):
            col_map[c] = "lon"
        elif cl in ("azimuth", "az"):
            col_map[c] = "azimuth"
        elif cl in ("pci",):
            col_map[c] = "pci"
        elif cl in ("site_name", "sitename", "site"):
            col_map[c] = "site_name"
        elif cl in ("band",):
            col_map[c] = "band"
        elif cl in ("tilt",):
            col_map[c] = "tilt"
        elif cl in ("tech", "technology"):
            col_map[c] = "tech"

    meta = cell_meta_df.rename(columns=col_map)

    required = ["cell_id", "lat", "lon", "azimuth"]
    for r in required:
        if r not in meta.columns:
            logger.warning("Missing required column '%s' in cell metadata", r)
            return []

    meta["lat"] = pd.to_numeric(meta["lat"], errors="coerce")
    meta["lon"] = pd.to_numeric(meta["lon"], errors="coerce")
    meta["azimuth"] = pd.to_numeric(meta["azimuth"], errors="coerce")
    meta = meta.dropna(subset=["lat", "lon", "azimuth"])

    # Compute per-sector metric if samples available
    sector_metric: Dict[Any, float] = {}
    if samples_df is not None and not samples_df.empty and color_by in samples_df.columns:
        mask = samples_df["latitude"].notna() & samples_df["longitude"].notna()
        samples_valid = samples_df[mask].copy()
        for _, row in meta.iterrows():
            cell_id = row["cell_id"]
            site_lat = float(row["lat"])
            site_lon = float(row["lon"])
            az = float(row["azimuth"])
            in_wedge = samples_valid.apply(
                lambda r: _point_in_wedge(
                    float(r["latitude"]),
                    float(r["longitude"]),
                    site_lat,
                    site_lon,
                    az,
                    hpbw_deg,
                    radius_m,
                ),
                axis=1,
            )
            wedge_samples = samples_valid.loc[in_wedge, color_by].dropna()
            if not wedge_samples.empty:
                sector_metric[cell_id] = float(wedge_samples.median())

    # Build wedge traces
    vranges = {
        "rsrp_dbm": (-130, -60),
        "rsrq_db": (-15, -3),
        "sinr_db": (-10, 30),
        "throughput_dl_mbps": (0, 100),
    }
    vmin, vmax = vranges.get(color_by, (-130, -60))

    def _metric_to_color(val: float) -> str:
        if color_by == "rsrp_dbm":
            from metrics.classifier import classify_rsrp as _cr
            _, color = _cr(val)
            return color
        if np.isnan(val):
            return "#888888"
        norm = (val - vmin) / max(vmax - vmin, 1)
        norm = max(0.0, min(1.0, norm))
        r = int(255 * (1 - norm))
        g = int(255 * norm)
        return f"#{r:02x}{g:02x}00"

    traces = []
    for _, row in meta.iterrows():
        cell_id = row["cell_id"]
        site_lat = float(row["lat"])
        site_lon = float(row["lon"])
        az = float(row["azimuth"])
        pci = row.get("pci", "")
        site_name = row.get("site_name", str(cell_id))
        band = row.get("band", "")
        tilt = row.get("tilt", "")
        tech = row.get("tech", "LTE")

        polygon = _build_wedge_polygon(site_lat, site_lon, az, hpbw_deg, radius_m)
        lats = [p[0] for p in polygon] + [None]
        lons = [p[1] for p in polygon] + [None]

        metric_val = sector_metric.get(cell_id, float("nan"))
        fill_color = _metric_to_color(metric_val)  # noqa

        n_samples = 0
        if samples_df is not None and "cell_id" in samples_df.columns:
            n_samples = int((samples_df["cell_id"] == cell_id).sum())

        tooltip = (
            f"<b>{site_name}</b> Cell {cell_id}<br>"
            f"PCI: {pci} | Band: {band} | Tech: {tech}<br>"
            f"Az: {az}° | Tilt: {tilt}°<br>"
            f"Samples: {n_samples}<br>"
            f"Median {color_by}: {metric_val:.1f}"
            if not np.isnan(metric_val)
            else f"<b>{site_name}</b> Cell {cell_id}<br>Az: {az}° | Samples: {n_samples}"
        )

        traces.append(
            {
                "type": "scattermapbox",
                "lat": lats,
                "lon": lons,
                "mode": "lines",
                "fill": "toself",
                "fillcolor": fill_color + "55",  # semi-transparent
                "line": {"color": fill_color, "width": 1.5},
                "text": [tooltip] * len(lats),
                "hovertemplate": "%{text}<extra></extra>",
                "name": f"Sector {cell_id}",
                "showlegend": False,
            }
        )

    # Site marker traces
    site_lats = meta["lat"].tolist()
    site_lons = meta["lon"].tolist()
    site_texts = [
        f"<b>{row.get('site_name', str(row['cell_id']))}</b><br>"
        f"Cell: {row['cell_id']} | Az: {row['azimuth']}°"
        for _, row in meta.iterrows()
    ]

    traces.append(
        {
            "type": "scattermapbox",
            "lat": site_lats,
            "lon": site_lons,
            "mode": "markers+text",
            "marker": {"size": 12, "color": "#e94560", "symbol": "triangle"},
            "text": [str(row.get("site_name", row["cell_id"])) for _, row in meta.iterrows()],
            "textposition": "top right",
            "hovertext": site_texts,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "name": "eNB/gNB Sites",
            "showlegend": True,
        }
    )

    return traces
