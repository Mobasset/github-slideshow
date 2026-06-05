"""
UE position estimation — weighted centroid of observed cells.

Algorithm
---------
For each sample belonging to a known UE (RNTI ≠ 0):
  1. Collect every serving-cell measurement from the same RNTI within
     ±window_ms of this sample's timestamp.
  2. For each unique cell seen in that window, keep the strongest RSRP.
  3. Convert RSRP to linear power:  w = 10^(RSRP_dBm / 10)
  4. UE position = weighted centroid of those cell tower positions.

Why this is correct
-------------------
RSRP gives *distance* but no *direction*. The azimuth of the serving
sector is irrelevant — the UE can be anywhere around the tower.
The weighted centroid places the UE between all measured cells, pulled
toward whichever tower is strongest at that moment. When only one cell
is visible the position lands at that tower; across a handover it
smoothly transitions between the two towers.

Fallback
--------
Samples with RNTI=0 (RLF, SCG events, per-radio measurements) have no
UE context, so they are placed at the serving cell site (tower lat/lon).
"""

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

_R_EARTH = 6_371_000.0          # metres


# ── Cell DB ──────────────────────────────────────────────────────────────────

def build_cell_db(cell_meta_df: pd.DataFrame) -> Dict[int, dict]:
    """
    Build {cell_id: {lat, lon, azimuth, site}} from a cell metadata DataFrame.

    Supports the Ericsson DB.csv layout:
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
    if not {'enb_id', 'local_cell', 'lat', 'lon'}.issubset(df.columns):
        return db

    for _, row in df.iterrows():
        try:
            cid = int(row['enb_id']) * 1000 + int(row['local_cell'])
            db[cid] = {
                'lat':     float(row['lat']),
                'lon':     float(row['lon']),
                'azimuth': float(row.get('azimuth', 0)),
                'site':    str(row.get('site', '')),
            }
        except (ValueError, KeyError, TypeError):
            continue

    return db


# ── Weighted centroid ────────────────────────────────────────────────────────

def _weighted_centroid(
    cell_rsrp: Dict[int, float],
    cell_db: Dict[int, dict],
) -> Tuple[float, float]:
    """
    Return (lat, lon) as a linear-power-weighted centroid of the given cells.

    cell_rsrp: {cell_id → best RSRP in dBm}
    """
    lat_acc = lon_acc = w_acc = 0.0
    for cid, rsrp in cell_rsrp.items():
        cell = cell_db.get(cid)
        if cell is None:
            continue
        w = 10 ** (rsrp / 10.0)          # dBm → linear power
        lat_acc += w * cell['lat']
        lon_acc += w * cell['lon']
        w_acc   += w
    if w_acc == 0:
        return math.nan, math.nan
    return lat_acc / w_acc, lon_acc / w_acc


def apply_ue_localization(
    df: pd.DataFrame,
    cell_db: Dict[int, dict],
    window_ms: int = 15_000,
) -> pd.DataFrame:
    """
    Add 'latitude' and 'longitude' to df using RSRP weighted-centroid positioning.

    Parameters
    ----------
    df         : DataFrame from parse_ctrace_binary / parse_ctrace
    cell_db    : lookup built by build_cell_db()
    window_ms  : half-width of the sliding time window (default 15 s)

    Returns
    -------
    df with 'latitude' and 'longitude' columns populated.
    """
    if df.empty or not cell_db:
        return df

    df = df.copy()
    df['latitude']  = np.nan
    df['longitude'] = np.nan

    ts_arr    = df['timestamp_ms'].to_numpy(dtype=float)
    cell_arr  = df['cell_id'].to_numpy()
    rsrp_arr  = df['rsrp_dbm'].to_numpy(dtype=float) if 'rsrp_dbm' in df.columns else np.full(len(df), np.nan)
    rnti_arr  = df['rnti'].to_numpy()   if 'rnti'   in df.columns else np.zeros(len(df))
    lats      = np.full(len(df), np.nan)
    lons      = np.full(len(df), np.nan)

    # ── Pass 1: RNTI-aware windowed centroid ─────────────────────────────────
    # Group indices by RNTI (skip RNTI=0)
    rnti_groups: Dict[int, list] = {}
    for i, r in enumerate(rnti_arr):
        ri = int(r)
        if ri == 0:
            continue
        rnti_groups.setdefault(ri, []).append(i)

    for rnti, indices in rnti_groups.items():
        indices.sort(key=lambda i: ts_arr[i])
        ts_rnti = [ts_arr[i] for i in indices]

        # Sliding window: for each sample find all indices within ±window_ms
        lo = hi = 0
        for pos, i in enumerate(indices):
            t = ts_rnti[pos]

            # Advance lo until ts >= t - window_ms
            while lo < len(indices) and ts_rnti[lo] < t - window_ms:
                lo += 1
            # Advance hi until ts > t + window_ms
            while hi < len(indices) and ts_rnti[hi] <= t + window_ms:
                hi += 1

            # Collect best RSRP per cell in window
            cell_rsrp: Dict[int, float] = {}
            for j in indices[lo:hi]:
                try:
                    cid = int(cell_arr[j])
                except (ValueError, TypeError):
                    continue
                if cid not in cell_db:
                    continue
                rsrp = rsrp_arr[j]
                if math.isfinite(rsrp):
                    if cid not in cell_rsrp or rsrp > cell_rsrp[cid]:
                        cell_rsrp[cid] = rsrp

            if not cell_rsrp:
                continue

            lat, lon = _weighted_centroid(cell_rsrp, cell_db)
            lats[i] = lat
            lons[i] = lon

    # ── Pass 2: fallback — serving cell site for remaining samples ───────────
    for i in range(len(df)):
        if math.isfinite(lats[i]):
            continue
        try:
            cid = int(cell_arr[i])
        except (ValueError, TypeError):
            continue
        cell = cell_db.get(cid)
        if cell is None:
            continue
        lats[i] = cell['lat']
        lons[i] = cell['lon']

    df['latitude']  = lats
    df['longitude'] = lons
    return df
