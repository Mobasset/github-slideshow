"""
Ericsson CTR (Cell Trace Recording) CSV/CSV.GZ parser for the VDT Tool.

Handles Ericsson CTR files in the format:
  ERBS_Name,ID,Time,Source,Destination,RBS_Module_ID,Global_Cell_ID,MMES1APID,
  ENBS1APID,Gummei,Type,Description[,field_name,value,field_name,value,...]

Column indices (0-based):
  0  ERBS_Name
  1  ID
  2  Time (HH:MM:SS:mmm)
  3  Source
  4  Destination
  5  RBS_Module_ID
  6  Global_Cell_ID  (enb_id-local_cell_id)
  7  MMES1APID       (used as rnti)
  8  ENBS1APID
  9  Gummei
  10 Type
  11 Description     (event name)
  12+  alternating key, value pairs

Date is extracted from the filename (e.g. A20260508).
"""

import gzip
import io
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Events we care about ──────────────────────────────────────────────────────

RELEVANT_EVENTS = {
    "RRC_MEASUREMENT_REPORT",
    "INTERNAL_EVENT_UE_MOBILITY_EVAL",
    "INTERNAL_PROC_HO_EXEC_X2_OUT",
    "INTERNAL_PROC_HO_PREP_X2_OUT",
    "INTERNAL_PROC_HO_EXEC_X2_IN",
    "INTERNAL_PROC_HO_PREP_X2_IN",
    "RRC_SCG_FAILURE_INFORMATION_NR",
    "RRC_RRC_CONNECTION_RE_ESTABLISHMENT_REQUEST",
    "INTERNAL_PROC_RRC_CONNECTION_RE_ESTABLISHMENT",
    "INTERNAL_PER_RADIO_UE_MEASUREMENT",
}

# SINR distribution bin midpoints for INTERNAL_PER_RADIO_UE_MEASUREMENT
SINR_BIN_MIDPOINTS = [-7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5, 14.0, 20.0]

# Regex for extracting date from filename (e.g. A20260508)
_DATE_RE = re.compile(r"A(\d{8})")


def _extract_date_str(filename: str) -> Optional[str]:
    """Extract YYYYMMDD date string from filename."""
    m = _DATE_RE.search(filename)
    if m:
        raw = m.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return None


def _parse_time(date_str: str, time_col: str) -> int:
    """
    Parse timestamp from date_str (YYYY-MM-DD) and time_col (HH:MM:SS:mmm).
    Returns milliseconds since epoch (UTC).
    """
    # time_col format: HH:MM:SS:mmm
    parts = time_col.split(":")
    if len(parts) < 3:
        return 0
    h = parts[0].zfill(2)
    m = parts[1].zfill(2)
    s = parts[2].zfill(2)
    ms = parts[3].zfill(3) if len(parts) > 3 else "000"
    iso = f"{date_str}T{h}:{m}:{s}.{ms}Z"
    try:
        ts = pd.Timestamp(iso, tz="UTC")
        return int(ts.timestamp() * 1000)
    except Exception:
        return 0


def _parse_timestamp_utc(date_str: str, time_col: str) -> pd.Timestamp:
    """Return a timezone-aware UTC Timestamp (or NaT on failure)."""
    parts = time_col.split(":")
    if len(parts) < 3:
        return pd.NaT
    h = parts[0].zfill(2)
    m = parts[1].zfill(2)
    s = parts[2].zfill(2)
    ms = parts[3].zfill(3) if len(parts) > 3 else "000"
    iso = f"{date_str}T{h}:{m}:{s}.{ms}Z"
    try:
        return pd.Timestamp(iso, tz="UTC")
    except Exception:
        return pd.NaT


def _parse_cell_id(global_cell_id: str) -> int:
    """Parse Global_Cell_ID 'enb_id-local_cell_id' → enb_id * 1000 + local_cell_id."""
    parts = global_cell_id.split("-")
    try:
        enb_id = int(parts[0])
        local_id = int(parts[1]) if len(parts) > 1 else 0
        return enb_id * 1000 + local_id
    except (ValueError, IndexError):
        return 0


def _parse_kv(cols: list) -> dict:
    """
    Parse alternating key-value pairs from cols[12:].
    Skips keys that start with '[' (list/array markers).
    """
    kv = {}
    params = cols[12:]
    i = 0
    while i + 1 < len(params):
        key = params[i].strip()
        val = params[i + 1].strip()
        if key and not key.startswith("["):
            kv[key] = val
        i += 2
    return kv


def _sinr_from_bins(kv: dict) -> float:
    """
    Compute weighted-average SINR from SINR_MEAS_PUSCH_0..8 distribution bins.
    Returns NaN if no bins present.
    """
    weights = []
    for idx in range(9):
        key = f"SINR_MEAS_PUSCH_{idx}"
        try:
            weights.append(float(kv.get(key, 0) or 0))
        except (ValueError, TypeError):
            weights.append(0.0)
    total = sum(weights)
    if total <= 0:
        return np.nan
    sinr = sum(w * m for w, m in zip(weights, SINR_BIN_MIDPOINTS)) / total
    return round(sinr, 2)


def _empty_df() -> pd.DataFrame:
    """Return empty DataFrame with the canonical VDT column set."""
    cols = [
        "timestamp_utc", "timestamp_ms", "cell_id", "rnti", "pci", "earfcn",
        "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ta_us",
        "throughput_dl_mbps", "throughput_ul_mbps",
        "ho_attempt", "ho_success", "ho_failure", "rlf_flag", "rab_setup",
        "scg_failure", "a3_trigger", "a5_trigger",
        "event_type", "event_cause", "source_file",
    ]
    return pd.DataFrame(columns=cols)


def parse_ctrace(data: bytes, filename: str = "ctrace") -> pd.DataFrame:
    """
    Parse an Ericsson CTR file (CSV or CSV.GZ) into a VDT samples DataFrame.

    Parameters
    ----------
    data : bytes
        Raw file contents (gzip-compressed or plain CSV bytes).
    filename : str
        Original filename, used to extract the measurement date (A20260508…).

    Returns
    -------
    pd.DataFrame
        Columns matching events_parser._empty_df() column order.
    """
    # 1. Decompress if needed
    if filename.lower().endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except Exception as exc:
            logger.warning("CTR gzip decompress failed for %s: %s", filename, exc)
            return _empty_df()

    # 2. Extract date from filename
    date_str = _extract_date_str(filename)
    if date_str is None:
        # Fall back to today's date so timestamps are at least internally consistent
        date_str = str(pd.Timestamp.now(tz="UTC").date())
        logger.warning("Could not extract date from filename '%s'; using %s", filename, date_str)

    # 3. Decode text
    text = data.decode("utf-8", errors="replace")

    records = []

    for lineno, line in enumerate(text.splitlines()):
        # Skip header line
        if lineno == 0:
            continue

        line = line.strip()
        if not line:
            continue

        cols = line.split(",")
        if len(cols) < 12:
            continue

        event_desc = cols[11].strip()
        if event_desc not in RELEVANT_EVENTS:
            continue

        # 4. Parse timestamp
        ts_ms = _parse_time(date_str, cols[2].strip())
        ts_utc = _parse_timestamp_utc(date_str, cols[2].strip())

        # 5. Parse cell_id from Global_Cell_ID (col 6)
        cell_id = _parse_cell_id(cols[6].strip())

        # 6. RNTI from MMES1APID (col 7)
        try:
            rnti = int(cols[7].strip())
        except ValueError:
            rnti = 0

        # 7. Parse key-value pairs from cols[12:]
        kv = _parse_kv(cols)

        # 8. Build record with defaults
        rec = {
            "timestamp_utc":       ts_utc,
            "timestamp_ms":        ts_ms,
            "cell_id":             cell_id,
            "rnti":                rnti,
            "pci":                 0,
            "earfcn":              0,
            "rsrp_dbm":            np.nan,
            "rsrq_db":             np.nan,
            "sinr_db":             np.nan,
            "cqi":                 np.nan,
            "ta_us":               np.nan,
            "throughput_dl_mbps":  np.nan,
            "throughput_ul_mbps":  np.nan,
            "ho_attempt":          0,
            "ho_success":          0,
            "ho_failure":          0,
            "rlf_flag":            0,
            "rab_setup":           0,
            "scg_failure":         0,
            "a3_trigger":          0,
            "a5_trigger":          0,
            "event_type":          event_desc,
            "event_cause":         "",
            "source_file":         filename,
        }

        # 9. Event-specific field extraction
        if event_desc == "RRC_MEASUREMENT_REPORT":
            # Use regex on the raw row string; first match = serving cell
            row_str = ",".join(cols[12:])
            rsrp_m = re.search(r"rsrpResult,(\d+)", row_str)
            rsrq_m = re.search(r"rsrqResult,(\d+)", row_str)
            if rsrp_m:
                ie = int(rsrp_m.group(1))
                rec["rsrp_dbm"] = ie - 140
            if rsrq_m:
                ie = int(rsrq_m.group(1))
                rec["rsrq_db"] = ie / 2 - 19.5

        elif event_desc == "INTERNAL_EVENT_UE_MOBILITY_EVAL":
            serving_rsrp = kv.get("SERVING_RSRP")
            serving_rsrq = kv.get("SERVING_RSRQ")
            trigger = kv.get("MOBILITY_TRIGGER", "")
            if serving_rsrp:
                try:
                    rec["rsrp_dbm"] = float(serving_rsrp)
                except ValueError:
                    pass
            if serving_rsrq:
                try:
                    rec["rsrq_db"] = float(serving_rsrq)
                except ValueError:
                    pass
            if "A3" in trigger.upper():
                rec["a3_trigger"] = 1
                rec["event_type"] = "A3_TRIGGER"
                rec["event_cause"] = trigger
            elif "A5" in trigger.upper():
                rec["a5_trigger"] = 1
                rec["event_type"] = "A5_TRIGGER"
                rec["event_cause"] = trigger

        elif event_desc == "INTERNAL_PROC_HO_EXEC_X2_OUT":
            result = kv.get("PROC_HO_EXEC_OUT_RESULT", "")
            rec["ho_attempt"] = 1
            if "SUCCESSFUL" in result.upper():
                rec["ho_success"] = 1
                rec["event_type"] = "INTERNAL_HANDOVER_SUCCESS"
            else:
                rec["ho_failure"] = 1
                rec["event_type"] = "INTERNAL_HANDOVER_FAILURE"
            rec["event_cause"] = result

        elif event_desc in (
            "INTERNAL_PROC_HO_PREP_X2_OUT",
            "INTERNAL_PROC_HO_EXEC_X2_IN",
            "INTERNAL_PROC_HO_PREP_X2_IN",
        ):
            rec["ho_attempt"] = 1
            rec["event_type"] = "INTERNAL_HANDOVER_ATTEMPT"

        elif event_desc == "RRC_SCG_FAILURE_INFORMATION_NR":
            failure_type = kv.get("failureType_r15", "")
            rec["scg_failure"] = 1
            rec["event_type"] = "NR_SCG_FAILURE"
            rec["event_cause"] = failure_type

        elif event_desc in (
            "RRC_RRC_CONNECTION_RE_ESTABLISHMENT_REQUEST",
            "INTERNAL_PROC_RRC_CONNECTION_RE_ESTABLISHMENT",
        ):
            rec["rlf_flag"] = 1
            rec["event_type"] = "RRC_RLF"

        elif event_desc == "INTERNAL_PER_RADIO_UE_MEASUREMENT":
            cqi_val = kv.get("LAST_CQI_1_REPORTED")
            if cqi_val:
                try:
                    rec["cqi"] = float(cqi_val)
                except ValueError:
                    pass
            rec["sinr_db"] = _sinr_from_bins(kv)

        records.append(rec)

    if not records:
        logger.info("No relevant CTR events found in %s", filename)
        return _empty_df()

    df = pd.DataFrame(records)

    # Ensure all required columns are present and in the right order
    for col in _empty_df().columns:
        if col not in df.columns:
            df[col] = np.nan if col not in {"event_type", "event_cause", "source_file"} else ""

    return df[_empty_df().columns]
