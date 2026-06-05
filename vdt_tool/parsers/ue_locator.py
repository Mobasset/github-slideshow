"""
UE position estimation from RSRP + cell metadata.

Estimates UE distance from the serving cell using a log-distance path loss
model, then projects the position along the cell azimuth at that distance.

Path loss model:  PL(d) = PL(d0) + 10·n·log10(d/d0)
Typical macro LTE: n=3.5, RSRP_ref=-65 dBm at d0=100 m (46 dBm TX, 1800 MHz)
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

_N = 3.5          # path loss exponent (urban/suburban macro)
_RSRP_REF = -65.0  # dBm at d_ref
_D_REF = 100.0     # metres
_D_MIN = 50.0      # clamp min
_D_MAX = 15_000.0  # clamp max
_R_EARTH = 6_371_000.0


def rsrp_to_distance(rsrp_dbm: float) -> float:
    """Estimate distance in metres from RSRP (dBm) via log-distance path loss."""
    if not math.isfinite(rsrp_dbm):
        return _D_MIN
    d = _D_REF * 10 ** ((_RSRP_REF - rsrp_dbm) / (10.0 * _N))
    return float(np.clip(d, _D_MIN, _D_MAX))


def offset_position(lat: float, lon: float, bearing_deg: float, dist_m: float) -> Tuple[float, float]:
    """
    Return (lat, lon) of a point dist_m metres from (lat, lon)
    along bearing_deg (degrees clockwise from North).
    """
    b = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dr = dist_m / _R_EARTH
    lat2 = math.asin(
        math.sin(lat1) * math.cos(dr)
        + math.cos(lat1) * math.sin(dr) * math.cos(b)
    )
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def build_cell_db(cell_meta_df: pd.DataFrame) -> Dict[int, dict]:
    """
    Build {cell_id: {lat, lon, azimuth, site}} from a cell metadata DataFrame.

    Supports the Ericsson DB.csv column layout:
        eNBId, Cellid1, Latitude, Longitude, Azimuth, Site

    cell_id key = eNBId * 1000 + Cellid1  (matches binary CTR parser encoding)
    """
    db: Dict[int, dict] = {}
    if cell_meta_df is None or cell_meta_df.empty:
        return db

    rename: Dict[str, str] = {}
    for col in cell_meta_df.columns:
        cl = col.lower().strip()
        if cl == 'enbid':
            rename[col] = 'enb_id'
        elif cl == 'cellid1':
            rename[col] = 'local_cell'
        elif cl in ('latitude', 'lat'):
            rename[col] = 'lat'
        elif cl in ('longitude', 'lon'):
            rename[col] = 'lon'
        elif cl in ('azimuth', 'az'):
            rename[col] = 'azimuth'
        elif cl == 'site':
            rename[col] = 'site'

    df = cell_meta_df.rename(columns=rename)

    if not {'enb_id', 'local_cell', 'lat', 'lon', 'azimuth'}.issubset(df.columns):
        return db

    for _, row in df.iterrows():
        try:
            cid = int(row['enb_id']) * 1000 + int(row['local_cell'])
            db[cid] = {
                'lat':     float(row['lat']),
                'lon':     float(row['lon']),
                'azimuth': float(row['azimuth']),
                'site':    str(row.get('site', '')),
            }
        except (ValueError, KeyError, TypeError):
            continue

    return db


def apply_ue_localization(
    df: pd.DataFrame,
    cell_db: Dict[int, dict],
    default_dist_m: float = 500.0,
) -> pd.DataFrame:
    """
    Add or fill 'latitude' / 'longitude' columns using RSRP-based positioning.

    Algorithm per row:
      1. Look up serving cell in cell_db by cell_id
      2. If RSRP available  → d = rsrp_to_distance(rsrp_dbm)
         Else (RLF / events) → d = default_dist_m
      3. lat, lon = offset_position(cell_lat, cell_lon, azimuth, d)

    Rows already having a non-NaN lat/lon are left unchanged.
    """
    if df.empty or not cell_db:
        return df

    df = df.copy()
    if 'latitude' not in df.columns:
        df['latitude'] = np.nan
    if 'longitude' not in df.columns:
        df['longitude'] = np.nan

    needs_pos = df['latitude'].isna()
    if not needs_pos.any():
        return df

    lats = df['latitude'].to_numpy(dtype=float, na_value=np.nan).copy()
    lons = df['longitude'].to_numpy(dtype=float, na_value=np.nan).copy()
    cell_ids = df['cell_id'].to_numpy()
    rsrps = df['rsrp_dbm'].to_numpy(dtype=float, na_value=np.nan).copy() if 'rsrp_dbm' in df.columns else np.full(len(df), np.nan)

    for i, need in enumerate(needs_pos):
        if not need:
            continue
        try:
            cid = int(cell_ids[i])
        except (ValueError, TypeError):
            continue
        cell = cell_db.get(cid)
        if cell is None:
            continue
        rsrp = rsrps[i]
        dist = rsrp_to_distance(float(rsrp)) if math.isfinite(rsrp) else default_dist_m
        lat, lon = offset_position(cell['lat'], cell['lon'], cell['azimuth'], dist)
        lats[i] = lat
        lons[i] = lon

    df['latitude'] = lats
    df['longitude'] = lons
    return df
