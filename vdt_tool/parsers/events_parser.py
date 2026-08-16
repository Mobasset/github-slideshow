"""
4G/5G event file parser.

Handles structured event exports from Ericsson OSS for LTE and NR:
  - CSV event exports (Ericsson OSS "event export" tool output)
  - Flat text event logs

Event types decoded:
  INTERNAL_HANDOVER_ATTEMPT, INTERNAL_HANDOVER_SUCCESS, INTERNAL_HANDOVER_FAILURE,
  RRC_RLF, INTERNAL_RAB_ESTABLISHMENT, INTERNAL_SYSTEM_RELEASE,
  INTERNAL_ERAB_SETUP, INTERNAL_UE_CONTEXT_RELEASE,
  NR_SCG_FAILURE, A3_TRIGGER, A5_TRIGGER
"""

import io
import logging
import re
from typing import List, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVENT_NAMES = {
    "INTERNAL_HANDOVER_ATTEMPT",
    "INTERNAL_HANDOVER_SUCCESS",
    "INTERNAL_HANDOVER_FAILURE",
    "RRC_RLF",
    "INTERNAL_RAB_ESTABLISHMENT",
    "INTERNAL_SYSTEM_RELEASE",
    "INTERNAL_ERAB_SETUP",
    "INTERNAL_UE_CONTEXT_RELEASE",
    "NR_SCG_FAILURE",
    "A3_TRIGGER",
    "A5_TRIGGER",
}

EVENT_ABBREV: Dict[str, str] = {
    "HO_ATT":  "INTERNAL_HANDOVER_ATTEMPT",
    "HO_SUCC": "INTERNAL_HANDOVER_SUCCESS",
    "HO_FAIL": "INTERNAL_HANDOVER_FAILURE",
    "RLF":     "RRC_RLF",
    "RAB_EST": "INTERNAL_RAB_ESTABLISHMENT",
    "SYS_REL": "INTERNAL_SYSTEM_RELEASE",
    "ERAB":    "INTERNAL_ERAB_SETUP",
    "UE_REL":  "INTERNAL_UE_CONTEXT_RELEASE",
    "SCG_FAIL":"NR_SCG_FAILURE",
    "A3":      "A3_TRIGGER",
    "A5":      "A5_TRIGGER",
}

COLUMN_ALIASES: Dict[str, str] = {
    "event_type":     "event_type",
    "eventtype":      "event_type",
    "event":          "event_type",
    "timestamp":      "timestamp_ms",
    "timestamp_ms":   "timestamp_ms",
    "timestamp_utc":  "timestamp_ms",
    "time":           "timestamp_ms",
    "cell_id":        "cell_id",
    "cellid":         "cell_id",
    "enb_id":         "cell_id",
    "source_cell":    "cell_id",
    "rnti":           "rnti",
    "ue_id":          "rnti",
    "pci":            "pci",
    "target_pci":     "pci",
    "earfcn":         "earfcn",
    "dl_earfcn":      "earfcn",
    "rsrp":           "rsrp_dbm",
    "rsrp_dbm":       "rsrp_dbm",
    "rsrq":           "rsrq_db",
    "rsrq_db":        "rsrq_db",
    "sinr":           "sinr_db",
    "sinr_db":        "sinr_db",
    "cause_code":     "cause_code",
    "causecode":      "cause_code",
    "cause":          "cause_code",
    "ho_type":        "ho_type",
    "target_cell_id": "target_cell_id",
    "targetcellid":   "target_cell_id",
}

_TS_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_EVENT_RE = re.compile(
    r"\b(" + "|".join(re.escape(e) for e in EVENT_NAMES) + r")\b", re.IGNORECASE
)


def _parse_ts(val: Any) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    val_str = str(val).strip()
    try:
        v = int(float(val_str))
        if v > 1_000_000_000_000:
            return v
        if v > 1_000_000_000:
            return v * 1000
    except (ValueError, OverflowError):
        pass
    import datetime
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            dt = datetime.datetime.strptime(val_str[:26], fmt)
            return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return 0


def _normalize_event_type(raw: str) -> str:
    raw = raw.strip().upper()
    if raw in EVENT_NAMES:
        return raw
    if raw in EVENT_ABBREV:
        return EVENT_ABBREV[raw]
    for canonical in EVENT_NAMES:
        if canonical in raw:
            return canonical
    return raw


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = col.lower().strip().replace(" ", "_").replace("-", "_")
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def parse_csv_events(data: bytes, source_file: str = "events_csv") -> pd.DataFrame:
    text = data.decode("utf-8", errors="replace")
    sample = text[:2048]
    delimiter = ","
    for delim in ["\t", ";", "|", ","]:
        if sample.count(delim) > sample.count(delimiter):
            delimiter = delim

    try:
        df = pd.read_csv(io.StringIO(text), delimiter=delimiter, low_memory=False)
    except Exception as exc:
        logger.warning("CSV parse error: %s", exc)
        return _empty_df()

    if df.empty:
        return _empty_df()

    df = _normalize_columns(df)

    if "timestamp_ms" in df.columns:
        df["timestamp_ms"] = df["timestamp_ms"].apply(_parse_ts)
    else:
        for col in df.columns:
            if re.search(r"time|ts|date", col, re.IGNORECASE):
                df["timestamp_ms"] = df[col].apply(_parse_ts)
                break
        else:
            df["timestamp_ms"] = 0

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).apply(_normalize_event_type)
    else:
        df["event_type"] = "UNKNOWN"

    for col in ["cell_id", "rnti", "pci", "earfcn"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[col] = 0

    for col in ["rsrp_dbm", "rsrq_db", "sinr_db"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["ho_attempt"]  = (df["event_type"] == "INTERNAL_HANDOVER_ATTEMPT").astype(int)
    df["ho_success"]  = (df["event_type"] == "INTERNAL_HANDOVER_SUCCESS").astype(int)
    df["ho_failure"]  = (df["event_type"] == "INTERNAL_HANDOVER_FAILURE").astype(int)
    df["rlf_flag"]    = (df["event_type"] == "RRC_RLF").astype(int)
    df["rab_setup"]   = df["event_type"].isin(
        ["INTERNAL_RAB_ESTABLISHMENT", "INTERNAL_ERAB_SETUP"]
    ).astype(int)
    df["scg_failure"] = (df["event_type"] == "NR_SCG_FAILURE").astype(int)
    df["a3_trigger"]  = (df["event_type"] == "A3_TRIGGER").astype(int)
    df["a5_trigger"]  = (df["event_type"] == "A5_TRIGGER").astype(int)
    df["event_cause"] = df["cause_code"].astype(str) if "cause_code" in df.columns else ""
    df["source_file"] = source_file

    for col in _empty_df().columns:
        if col not in df.columns:
            df[col] = np.nan if col not in {"event_type", "event_cause", "source_file"} else ""

    return df[_empty_df().columns]


def parse_text_events(data: bytes, source_file: str = "events_log") -> pd.DataFrame:
    text = data.decode("utf-8", errors="replace")
    records = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        ts_m = _TS_RE.search(line)
        if not ts_m:
            nums = re.findall(r"\b(\d{13})\b", line)
            ts_ms = int(nums[0]) if nums else 0
        else:
            ts_ms = _parse_ts(ts_m.group(1))

        etype_m = _EVENT_RE.search(line)
        if not etype_m:
            continue
        event_type = _normalize_event_type(etype_m.group(1))

        cell_m = re.search(r"cell[_\s]?(?:id)?[:\s]+(\d+)", line, re.IGNORECASE)
        cell_id = int(cell_m.group(1)) if cell_m else 0
        cause_m = re.search(r"cause[:\s]+(\w+)", line, re.IGNORECASE)
        cause = cause_m.group(1) if cause_m else ""

        records.append({
            "timestamp_ms": ts_ms,
            "timestamp_utc": pd.Timestamp(ts_ms, unit="ms", tz="UTC") if ts_ms else pd.NaT,
            "cell_id": cell_id, "rnti": 0, "pci": 0, "earfcn": 0,
            "rsrp_dbm": np.nan, "rsrq_db": np.nan, "sinr_db": np.nan,
            "cqi": np.nan, "ta_us": np.nan,
            "throughput_dl_mbps": np.nan, "throughput_ul_mbps": np.nan,
            "ho_attempt":  int(event_type == "INTERNAL_HANDOVER_ATTEMPT"),
            "ho_success":  int(event_type == "INTERNAL_HANDOVER_SUCCESS"),
            "ho_failure":  int(event_type == "INTERNAL_HANDOVER_FAILURE"),
            "rlf_flag":    int(event_type == "RRC_RLF"),
            "rab_setup":   int(event_type in ("INTERNAL_RAB_ESTABLISHMENT", "INTERNAL_ERAB_SETUP")),
            "scg_failure": int(event_type == "NR_SCG_FAILURE"),
            "a3_trigger":  int(event_type == "A3_TRIGGER"),
            "a5_trigger":  int(event_type == "A5_TRIGGER"),
            "event_type":  event_type,
            "event_cause": cause,
            "source_file": source_file,
        })

    if not records:
        return _empty_df()
    return pd.DataFrame(records)


def parse_events_bytes(data: bytes, filename: str = "events") -> pd.DataFrame:
    """Auto-detect 4G/5G event file format (CSV or text log) and parse."""
    # Sniff first line for CSV structure
    try:
        first_line = data[:256].decode("utf-8", errors="replace").split("\n")[0]
        if "," in first_line or "\t" in first_line or ";" in first_line:
            return parse_csv_events(data, source_file=filename)
    except Exception:
        pass
    return parse_text_events(data, source_file=filename)


def _empty_df() -> pd.DataFrame:
    cols = [
        "timestamp_utc", "timestamp_ms", "cell_id", "rnti", "pci", "earfcn",
        "rsrp_dbm", "rsrq_db", "sinr_db", "cqi", "ta_us",
        "throughput_dl_mbps", "throughput_ul_mbps",
        "ho_attempt", "ho_success", "ho_failure", "rlf_flag", "rab_setup",
        "scg_failure", "a3_trigger", "a5_trigger",
        "event_type", "event_cause", "source_file",
    ]
    return pd.DataFrame(columns=cols)
