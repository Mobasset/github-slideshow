"""
KMZ exporter: builds a Google Earth-compatible .kmz file with
color-coded sample dots and sector wedge overlays.
"""

import io
import logging
import math
import zipfile
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

KML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document>
  <name>VDT Export</name>
  <open>1</open>
"""

KML_FOOTER = """</Document>
</kml>"""

# RSRP bin → KML color (aabbggrr format for KML)
RSRP_KML_COLORS = {
    "Excellent": "ff71ec2e",
    "Good":      "ff8dd0a8",
    "Fair":      "ff00a5f0",
    "Poor":      "ff0060e0",
    "Bad":       "ff2b39c0",
    "N/A":       "ff888888",
}

EVENT_KML_COLORS = {
    "INTERNAL_HANDOVER_FAILURE": "ff0000ff",
    "RRC_RLF":                   "ff0000aa",
    "NR_SCG_FAILURE":            "ff0080ff",
    "INTERNAL_HANDOVER_SUCCESS": "ff00ff00",
    "A3_TRIGGER":                "ffff8800",
    "A5_TRIGGER":                "ffff4400",
}


def _kml_style(style_id: str, color: str, scale: float = 0.6) -> str:
    return f"""  <Style id="{style_id}">
    <IconStyle>
      <color>{color}</color>
      <scale>{scale}</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/circle.png</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0</scale></LabelStyle>
  </Style>
"""


def _kml_placemark(lat: float, lon: float, name: str, style_id: str, desc: str) -> str:
    return f"""  <Placemark>
    <name>{name}</name>
    <description><![CDATA[{desc}]]></description>
    <styleUrl>#{style_id}</styleUrl>
    <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
  </Placemark>
"""


def _kml_linestring(coords: list, name: str, color: str = "ff00ffff", width: int = 2) -> str:
    coord_str = " ".join(f"{lon:.6f},{lat:.6f},0" for lat, lon in coords)
    return f"""  <Placemark>
    <name>{name}</name>
    <Style><LineStyle><color>{color}</color><width>{width}</width></LineStyle></Style>
    <LineString><coordinates>{coord_str}</coordinates></LineString>
  </Placemark>
"""


def _kml_polygon(coords: list, name: str, line_color: str, fill_color: str) -> str:
    coord_str = " ".join(f"{lon:.6f},{lat:.6f},0" for lat, lon in coords)
    return f"""  <Placemark>
    <name>{name}</name>
    <Style>
      <LineStyle><color>{line_color}</color><width>1</width></LineStyle>
      <PolyStyle><color>{fill_color}</color></PolyStyle>
    </Style>
    <Polygon><outerBoundaryIs><LinearRing>
      <coordinates>{coord_str}</coordinates>
    </LinearRing></outerBoundaryIs></Polygon>
  </Placemark>
"""


def export_kmz(
    df: pd.DataFrame,
    cell_meta_df: Optional[pd.DataFrame] = None,
    include_track: bool = True,
    include_sectors: bool = True,
    include_events: bool = True,
) -> bytes:
    """
    Build a KMZ file from the unified DataFrame.

    Returns raw bytes of the .kmz (zip containing doc.kml).
    """
    kml_parts = [KML_HEADER]

    # ── Styles ────────────────────────────────────────────────────────────────
    for bin_label, color in RSRP_KML_COLORS.items():
        style_id = f"rsrp_{bin_label.lower()}"
        kml_parts.append(_kml_style(style_id, color, scale=0.5))

    for etype, color in EVENT_KML_COLORS.items():
        kml_parts.append(_kml_style(f"evt_{etype.lower()}", color, scale=0.8))

    # ── GPS track polyline ─────────────────────────────────────────────────────
    if include_track and not df.empty and "latitude" in df.columns:
        mask = df["latitude"].notna() & df["longitude"].notna()
        track_df = df[mask].sort_values("timestamp_ms")
        if len(track_df) >= 2:
            coords = list(zip(track_df["latitude"], track_df["longitude"]))
            # Downsample track to max 2000 points
            if len(coords) > 2000:
                step = len(coords) // 2000
                coords = coords[::step]
            kml_parts.append(
                '<Folder><name>GPS Track</name>\n'
                + _kml_linestring(coords, "Drive Route", color="ffffaa00", width=3)
                + '</Folder>\n'
            )

    # ── RSRP sample dots ──────────────────────────────────────────────────────
    if not df.empty and "latitude" in df.columns:
        mask = df["latitude"].notna() & df["longitude"].notna()
        df_valid = df[mask]

        # Downsample to max 5000 markers for KMZ performance
        if len(df_valid) > 5000:
            df_valid = df_valid.sample(n=5000, random_state=42)

        kml_parts.append('<Folder><name>RSRP Samples</name>\n')
        for _, row in df_valid.iterrows():
            rsrp_class = row.get("rsrp_class", "N/A")
            style_id = f"rsrp_{rsrp_class.lower()}"
            rsrp = row.get("rsrp_dbm", float("nan"))
            rsrq = row.get("rsrq_db", float("nan"))
            sinr = row.get("sinr_db", float("nan"))
            cell_id = row.get("cell_id", "")
            event_type = row.get("event_type", "")

            desc = (
                f"RSRP: {rsrp:.1f} dBm<br>"
                f"RSRQ: {rsrq:.1f} dB<br>"
                f"SINR: {sinr:.1f} dB<br>"
                f"Cell: {cell_id}<br>"
                f"Event: {event_type}"
            )
            kml_parts.append(
                _kml_placemark(
                    float(row["latitude"]),
                    float(row["longitude"]),
                    rsrp_class,
                    style_id,
                    desc,
                )
            )
        kml_parts.append('</Folder>\n')

    # ── Event markers ─────────────────────────────────────────────────────────
    if include_events and not df.empty and "event_type" in df.columns:
        event_mask = (
            df["latitude"].notna()
            & df["longitude"].notna()
            & df["event_type"].isin(EVENT_KML_COLORS.keys())
        )
        df_events = df[event_mask]
        if not df_events.empty:
            kml_parts.append('<Folder><name>Events</name>\n')
            for _, row in df_events.iterrows():
                etype = row["event_type"]
                style_id = f"evt_{etype.lower()}"
                cause = row.get("event_cause", "")
                desc = f"Event: {etype}<br>Cause: {cause}"
                kml_parts.append(
                    _kml_placemark(
                        float(row["latitude"]),
                        float(row["longitude"]),
                        etype.replace("INTERNAL_", ""),
                        style_id,
                        desc,
                    )
                )
            kml_parts.append('</Folder>\n')

    # ── Sector wedges ─────────────────────────────────────────────────────────
    if include_sectors and cell_meta_df is not None and not cell_meta_df.empty:
        from visualization.sector_overlay import _build_wedge_polygon, DEFAULT_HPBW_DEG, DEFAULT_ISD_M

        kml_parts.append('<Folder><name>Sector Wedges</name>\n')
        col_map = {}
        for c in cell_meta_df.columns:
            cl = c.lower().strip()
            if cl in ("cellid", "cell_id"):
                col_map[c] = "cell_id"
            elif cl in ("lat", "latitude"):
                col_map[c] = "lat"
            elif cl in ("lon", "longitude"):
                col_map[c] = "lon"
            elif cl in ("azimuth", "az"):
                col_map[c] = "azimuth"
            elif cl in ("site_name", "sitename"):
                col_map[c] = "site_name"

        meta = cell_meta_df.rename(columns=col_map)
        if all(c in meta.columns for c in ["lat", "lon", "azimuth"]):
            for _, row in meta.iterrows():
                polygon = _build_wedge_polygon(
                    float(row["lat"]),
                    float(row["lon"]),
                    float(row["azimuth"]),
                    DEFAULT_HPBW_DEG,
                    DEFAULT_ISD_M,
                )
                site = row.get("site_name", str(row.get("cell_id", "")))
                kml_parts.append(
                    _kml_polygon(
                        polygon,
                        f"Sector {site}",
                        line_color="ffe94560",
                        fill_color="330f3460",
                    )
                )
        kml_parts.append('</Folder>\n')

    kml_parts.append(KML_FOOTER)
    kml_content = "".join(kml_parts).encode("utf-8")

    # Pack into .kmz (zip)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_content)
    buf.seek(0)
    return buf.read()
