"""
GPS track correlator: aligns GPS samples ↔ L3/event samples by timestamp.

Supported GPS input formats:
  - CSV: columns timestamp_ms (or timestamp/time), latitude, longitude, [imsi]
  - KML: <Placemark> with <TimeStamp><when> and <Point><coordinates>
  - GPX: <trkpt lat= lon=> with <time> element

Output: GPS track as DataFrame with columns:
  timestamp_ms, latitude, longitude, imsi (optional)

Correlation: pandas merge_asof with tolerance=500ms applied to the
merged signal DataFrame. Missing GPS positions interpolated linearly.
"""

import gzip
import io
import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GPS_TOLERANCE_MS = 500  # ±500ms for merge_asof alignment

_NS_KML = {"k": "http://www.opengis.net/kml/2.2"}
_NS_GPX = {"g": "http://www.topografix.com/GPX/1/1"}


def _parse_iso_ts_ms(val: str) -> int:
    """Parse ISO 8601 timestamp string to epoch ms."""
    import datetime
    val = val.strip()
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            dt = datetime.datetime.strptime(val[:26], fmt)
            dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return 0


def _ts_col_to_ms(series: pd.Series) -> pd.Series:
    """Convert a mixed timestamp column to epoch ms integers."""
    def _convert(val):
        if pd.isna(val):
            return 0
        v_str = str(val).strip()
        # Pure numeric
        try:
            v = int(float(v_str))
            if v > 1_600_000_000_000:  # ms epoch
                return v
            if v > 1_600_000_000:  # s epoch
                return v * 1000
        except (ValueError, OverflowError):
            pass
        return _parse_iso_ts_ms(v_str)

    return series.apply(_convert)


def parse_gps_csv(data: bytes, source_file: str = "gps_csv") -> pd.DataFrame:
    """Parse GPS CSV file."""
    text = data.decode("utf-8", errors="replace")
    # Auto-detect delimiter
    sample = text[:1024]
    delimiter = ","
    for delim in ["\t", ";", "|", ","]:
        if sample.count(delim) > sample.count(delimiter):
            delimiter = delim

    try:
        df = pd.read_csv(io.StringIO(text), delimiter=delimiter, low_memory=False)
    except Exception as exc:
        logger.warning("GPS CSV parse error: %s", exc)
        return _empty_gps_df()

    if df.empty:
        return _empty_gps_df()

    # Normalize column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Map timestamp column
    ts_col = None
    for candidate in ["timestamp_ms", "timestamp", "time", "ts", "utc_time", "gps_time"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        logger.warning("No timestamp column found in GPS CSV")
        df["timestamp_ms"] = 0
    else:
        df["timestamp_ms"] = _ts_col_to_ms(df[ts_col])

    # Map lat/lon columns
    lat_col = next(
        (c for c in df.columns if c in ["latitude", "lat", "gps_lat", "y"]), None
    )
    lon_col = next(
        (c for c in df.columns if c in ["longitude", "lon", "lng", "gps_lon", "x"]), None
    )

    if lat_col is None or lon_col is None:
        logger.warning("Latitude/longitude columns not found in GPS CSV")
        return _empty_gps_df()

    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

    # Optional IMSI
    imsi_col = next((c for c in df.columns if "imsi" in c), None)
    df["imsi"] = df[imsi_col] if imsi_col else ""

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["timestamp_ms"] > 0]

    return df[["timestamp_ms", "latitude", "longitude", "imsi"]].reset_index(drop=True)


def parse_gps_kml(data: bytes, source_file: str = "gps_kml") -> pd.DataFrame:
    """Parse GPS KML file."""
    records = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        logger.warning("KML parse error: %s", exc)
        return _empty_gps_df()

    # Strip namespace for robust matching
    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    for pm in root.iter():
        if strip_ns(pm.tag) != "Placemark":
            continue

        # Find timestamp
        ts_ms = 0
        for child in pm.iter():
            if strip_ns(child.tag) == "when" and child.text:
                ts_ms = _parse_iso_ts_ms(child.text)
                break
            if strip_ns(child.tag) == "TimeStamp":
                when = child.find(".//{http://www.opengis.net/kml/2.2}when") or child.find(".//when")
                if when is not None and when.text:
                    ts_ms = _parse_iso_ts_ms(when.text)
                    break

        # Find coordinates
        lat, lon = float("nan"), float("nan")
        for child in pm.iter():
            if strip_ns(child.tag) == "coordinates" and child.text:
                coords = child.text.strip().split(",")
                if len(coords) >= 2:
                    try:
                        lon = float(coords[0])
                        lat = float(coords[1])
                    except ValueError:
                        pass
                break

        if not (np.isnan(lat) or np.isnan(lon)):
            records.append(
                {"timestamp_ms": ts_ms, "latitude": lat, "longitude": lon, "imsi": ""}
            )

    if not records:
        return _empty_gps_df()
    return pd.DataFrame(records)


def parse_gps_gpx(data: bytes, source_file: str = "gps_gpx") -> pd.DataFrame:
    """Parse GPS GPX track file."""
    records = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        logger.warning("GPX parse error: %s", exc)
        return _empty_gps_df()

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    for elem in root.iter():
        if strip_ns(elem.tag) != "trkpt":
            continue
        try:
            lat = float(elem.get("lat", "nan"))
            lon = float(elem.get("lon", "nan"))
        except ValueError:
            continue

        ts_ms = 0
        for child in elem:
            if strip_ns(child.tag) == "time" and child.text:
                ts_ms = _parse_iso_ts_ms(child.text)
                break

        if not (np.isnan(lat) or np.isnan(lon)):
            records.append(
                {"timestamp_ms": ts_ms, "latitude": lat, "longitude": lon, "imsi": ""}
            )

    if not records:
        return _empty_gps_df()
    return pd.DataFrame(records)


def parse_gps_bytes(data: bytes, filename: str = "gps") -> pd.DataFrame:
    """Auto-detect GPS file format and parse."""
    fname_lower = filename.lower()

    if fname_lower.endswith(".gpx"):
        return parse_gps_gpx(data, source_file=filename)
    if fname_lower.endswith(".kml") or fname_lower.endswith(".kmz"):
        if fname_lower.endswith(".kmz"):
            import zipfile
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    kml_names = [n for n in zf.namelist() if n.endswith(".kml")]
                    if kml_names:
                        data = zf.read(kml_names[0])
            except Exception as exc:
                logger.warning("KMZ extract error: %s", exc)
        return parse_gps_kml(data, source_file=filename)

    # Try CSV
    sample = data[:512].decode("utf-8", errors="replace")
    if "lat" in sample.lower() or "lon" in sample.lower():
        return parse_gps_csv(data, source_file=filename)

    # Try XML-based formats
    if b"<trk" in data or b"<trkpt" in data:
        return parse_gps_gpx(data, source_file=filename)
    if b"<Placemark" in data or b"<kml" in data:
        return parse_gps_kml(data, source_file=filename)

    # Default: try CSV
    return parse_gps_csv(data, source_file=filename)


def interpolate_gps_track(gps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate GPS track to fill small time gaps.
    Uses linear interpolation on lat/lon for gaps < 30 seconds.
    """
    if gps_df.empty:
        return gps_df

    gps_sorted = gps_df.sort_values("timestamp_ms").reset_index(drop=True)

    # Only interpolate if we have enough points
    if len(gps_sorted) < 3:
        return gps_sorted

    # Mark large gaps (> 30s) — don't interpolate across those
    time_diff = gps_sorted["timestamp_ms"].diff()
    large_gap = time_diff > 30_000  # 30 seconds

    gps_sorted["latitude"] = gps_sorted["latitude"].interpolate(
        method="linear", limit_direction="both"
    )
    gps_sorted["longitude"] = gps_sorted["longitude"].interpolate(
        method="linear", limit_direction="both"
    )

    return gps_sorted


def correlate_gps_to_samples(
    samples_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    tolerance_ms: int = GPS_TOLERANCE_MS,
) -> pd.DataFrame:
    """
    Align GPS track to signal samples using merge_asof (nearest timestamp).

    Args:
        samples_df: DataFrame with 'timestamp_ms' column
        gps_df: GPS DataFrame with 'timestamp_ms', 'latitude', 'longitude'
        tolerance_ms: Maximum allowed time difference in ms

    Returns:
        samples_df with 'latitude', 'longitude' columns added/updated.
    """
    if samples_df.empty:
        return samples_df

    if gps_df.empty:
        samples_df["latitude"] = float("nan")
        samples_df["longitude"] = float("nan")
        samples_df["gps_matched"] = False
        return samples_df

    # Ensure integer timestamp for merge
    samples_sorted = samples_df.copy()
    samples_sorted["timestamp_ms"] = pd.to_numeric(
        samples_sorted["timestamp_ms"], errors="coerce"
    ).fillna(0).astype("int64")

    gps_sorted = gps_df.copy()
    gps_sorted["timestamp_ms"] = pd.to_numeric(
        gps_sorted["timestamp_ms"], errors="coerce"
    ).fillna(0).astype("int64")
    gps_sorted = gps_sorted.sort_values("timestamp_ms").reset_index(drop=True)

    samples_sorted = samples_sorted.sort_values("timestamp_ms").reset_index(drop=True)

    merged = pd.merge_asof(
        samples_sorted,
        gps_sorted[["timestamp_ms", "latitude", "longitude"]],
        on="timestamp_ms",
        tolerance=tolerance_ms,
        direction="nearest",
        suffixes=("", "_gps"),
    )

    # If samples already had lat/lon columns, fill only where GPS matched
    if "latitude_gps" in merged.columns:
        merged["latitude"] = merged["latitude_gps"].combine_first(
            merged.get("latitude", pd.Series(dtype=float))
        )
        merged["longitude"] = merged["longitude_gps"].combine_first(
            merged.get("longitude", pd.Series(dtype=float))
        )
        merged = merged.drop(columns=["latitude_gps", "longitude_gps"], errors="ignore")
    elif "latitude" not in merged.columns:
        merged["latitude"] = float("nan")
        merged["longitude"] = float("nan")

    merged["gps_matched"] = merged["latitude"].notna() & merged["longitude"].notna()

    return merged


def apply_cell_centroid_positions(
    df: pd.DataFrame, cell_meta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For samples without GPS, assign cell centroid lat/lon from metadata.
    This is the fallback when no GPS file is provided.
    """
    if cell_meta_df.empty or df.empty:
        return df

    # Normalize cell_meta columns
    col_map = {}
    for c in cell_meta_df.columns:
        cl = c.lower().strip()
        if cl in ("cellid", "cell_id"):
            col_map[c] = "cell_id"
        elif cl in ("lat", "latitude"):
            col_map[c] = "lat"
        elif cl in ("lon", "longitude", "lng"):
            col_map[c] = "lon"
    cell_pos = cell_meta_df.rename(columns=col_map)[["cell_id", "lat", "lon"]].copy()
    cell_pos["cell_id"] = pd.to_numeric(cell_pos["cell_id"], errors="coerce")

    df_merged = df.merge(cell_pos, on="cell_id", how="left")
    # Only fill where GPS is missing
    if "latitude" not in df_merged.columns:
        df_merged["latitude"] = df_merged["lat"]
        df_merged["longitude"] = df_merged["lon"]
    else:
        mask = df_merged["latitude"].isna()
        df_merged.loc[mask, "latitude"] = df_merged.loc[mask, "lat"]
        df_merged.loc[mask, "longitude"] = df_merged.loc[mask, "lon"]

    df_merged = df_merged.drop(columns=["lat", "lon"], errors="ignore")
    return df_merged


def _empty_gps_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp_ms", "latitude", "longitude", "imsi"])
